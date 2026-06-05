"""
SongUNetEmbBranch: SongUNetPosEmbd with a parallel CNN branch for satellite embeddings.

The satellite embeddings are processed by a separate shallow CNN in parallel with
the UNet's full-resolution (level-0) blocks; the resulting features are concatenated
and fused via a 1x1 conv before the first downsampling step, giving the embeddings a
dedicated processing pathway.

The embedding can reach the branch in two ways (the fusion is identical for both):
  * LEGACY concat (N1, ``embedding_separate=False``): the last `alpha_earth_channels`
    input channels of `x` are the embedding; they are split off here. The embedding
    grid matches the weather grid (448x448).
  * SEPARATE tensor (N8, ``embedding_separate=True``): the embedding arrives as a
    dedicated `embedding=` kwarg at a higher resolution
    (``emb_downscale_factor`` x the weather grid, e.g. 3584=8x448). A
    ``pixel_unshuffle(factor)`` front-end folds the 8x8 sub-pixels of each weather
    cell into channels (C -> C*factor**2) on the 448x448 grid, so a conv can LEARN
    the intra-cell variation instead of averaging it away. ``factor=1`` reproduces
    the N1 branch exactly (unshuffle is a no-op).

Architecture at level-0 (full resolution):
  Main path:  Conv2d(era5_channels → model_channels) + 4x UNetBlocks
  Emb branch: pixel_unshuffle(factor) → 2x Conv2d(C*factor**2 → emb_branch_channels)
  Fusion:     cat([main, emb]) → Conv2d(model_channels + emb_branch_channels → model_channels)
  Then:       standard downsampling and decoder unchanged

Usage:
  In the corrdiff model config set:
    model_args:
      model_type: "SongUNetEmbBranch"
      alpha_earth_channels: 64        # 64 for Alpha Earth, 128 for OLMO
      emb_branch_channels: 64
      emb_downscale_factor: 8         # 1 for N1 (default), 8 for N8
      embedding_separate: true        # true for N8 (separate tensor), false for N1
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import silu
from torch.utils.checkpoint import checkpoint

from physicsnemo.models.diffusion import Conv2d, UNetBlock
from physicsnemo.models.diffusion.song_unet import SongUNetPosEmbd


class SongUNetEmbBranch(SongUNetPosEmbd):
    """SongUNetPosEmbd extended with a parallel CNN branch for alpha earth embeddings.

    The LAST `alpha_earth_channels` channels of the input tensor are treated as
    satellite embeddings. They are split off before the UNet encoder and processed
    by a dedicated 2-layer CNN (`emb_branch`). After the level-0 (full-resolution)
    UNet blocks complete, the two feature maps are concatenated and fused back to
    `model_channels` via a 1×1 conv (`fusion_conv`) before the first downsampling.

    Parameters
    ----------
    img_resolution : int or (int, int)
        Spatial resolution of input/output images.
    in_channels : int
        TOTAL number of input channels, including alpha_earth_channels.
        (e.g. 80 = 4 zeros + 12 ERA5 + 64 alpha_earth, before grid channels are appended
        internally by SongUNetPosEmbd)
    out_channels : int
        Number of output channels.
    alpha_earth_channels : int, default=64
        Number of embedding channels (64 for Alpha Earth, 128 for OLMO). In the
        legacy concat path these are the trailing input channels of `x`, split off
        and NOT passed to the main UNet encoder. In ``embedding_separate`` mode they
        are the channel count of the separate `embedding=` tensor.
    emb_branch_channels : int, default=64
        Number of output channels of the parallel CNN branch.
    emb_downscale_factor : int, default=1
        Per-axis ratio of the embedding grid to the weather grid. 1 = N1 (same
        grid); 8 = N8 (3584=8x448). A ``pixel_unshuffle(factor)`` front-end folds
        the factor**2 sub-pixels of each cell into channels before the branch convs.
    embedding_separate : bool, default=False
        If True, the embedding arrives via the `embedding=` forward kwarg and the
        full `x` is the (weather + grid) main input (no trailing-channel split).
        If False (legacy N1), the embedding is the last `alpha_earth_channels`
        channels of `x`.
    **kwargs
        All other keyword arguments are forwarded to SongUNetPosEmbd / SongUNet.
    """

    def __init__(
        self,
        img_resolution,
        in_channels: int,
        out_channels: int,
        alpha_earth_channels: int = 64,
        emb_branch_channels: int = 64,
        emb_downscale_factor: int = 1,
        embedding_separate: bool = False,
        **kwargs,
    ):
        # Parent (SongUNetPosEmbd → SongUNet) sees the MAIN input channels only.
        # Legacy concat: `x` includes the trailing embedding channels, so subtract
        # them. Separate: `x` is already weather+grid only, so pass through as-is.
        main_in_channels = (
            in_channels if embedding_separate else in_channels - alpha_earth_channels
        )
        super().__init__(
            img_resolution=img_resolution,
            in_channels=main_in_channels,
            out_channels=out_channels,
            **kwargs,
        )
        self.alpha_earth_channels = alpha_earth_channels
        self.emb_downscale_factor = int(emb_downscale_factor)
        self.embedding_separate = bool(embedding_separate)
        # A >1 factor means the embedding is at a higher resolution than the
        # weather grid, so it CANNOT be concatenated into the input — it must be
        # delivered as a separate tensor. Guard against an inconsistent config.
        if self.emb_downscale_factor > 1 and not self.embedding_separate:
            raise ValueError(
                f"emb_downscale_factor={self.emb_downscale_factor} requires "
                "embedding_separate=True (a higher-resolution embedding cannot be "
                "concatenated onto the weather input)."
            )
        model_channels = kwargs.get("model_channels", 128)
        num_groups = min(32, emb_branch_channels)

        # pixel_unshuffle(factor) folds the factor**2 sub-pixels of each weather cell
        # into channels, so the first conv sees every sub-pixel value and can learn
        # the intra-cell variation. factor=1 is a no-op (identity), so the N1 branch
        # is unchanged: first conv = Conv2d(alpha_earth_channels, ...).
        branch_in_channels = alpha_earth_channels * self.emb_downscale_factor**2

        # Parallel CNN branch — plain torch layers with same-padding convolutions.
        # GroupNorm handles mixed-precision autocast without special setup.
        self.emb_branch = torch.nn.Sequential(
            torch.nn.Conv2d(branch_in_channels, emb_branch_channels, 3, padding=1),
            torch.nn.GroupNorm(num_groups, emb_branch_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(emb_branch_channels, emb_branch_channels, 3, padding=1),
            torch.nn.GroupNorm(num_groups, emb_branch_channels),
            torch.nn.SiLU(),
        )

        # Fusion 1×1 conv: (model_channels + emb_branch_channels) → model_channels.
        # Near-zero weight init so the branch starts as a small perturbation,
        # keeping training close to the baseline at initialization.
        self.fusion_conv = Conv2d(
            in_channels=model_channels + emb_branch_channels,
            out_channels=model_channels,
            kernel=1,
            fused_conv_bias=True,
            amp_mode=kwargs.get("amp_mode", False),
            init_mode="xavier_uniform",
            init_weight=1e-5,
        )

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        global_index=None,
        embedding_selector=None,
        augment_labels=None,
        lead_time_label=None,
        embedding=None,
    ):
        """Forward pass with parallel satellite embedding branch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W). In the legacy concat path
            the LAST `alpha_earth_channels` channels are the embeddings; the rest are
            ERA5 + zero-filled output placeholder. In ``embedding_separate`` mode the
            whole tensor is the (weather + grid) main input.
        embedding : torch.Tensor, optional
            Embedding tensor of shape (B, alpha_earth_channels, factor*H, factor*W),
            used in ``embedding_separate`` mode. Reduced to (B, .., H, W) via
            ``pixel_unshuffle(emb_downscale_factor)`` before the branch convs.
        noise_labels, class_labels, global_index, embedding_selector,
        augment_labels, lead_time_label : see SongUNetPosEmbd
        """
        # 1. Obtain the embedding tensor x_emb and the main input x_main.
        if self.embedding_separate or embedding is not None:
            if embedding is None:
                raise ValueError(
                    "SongUNetEmbBranch was built with embedding_separate=True but "
                    "forward() received embedding=None. Ensure the loss/precond "
                    "forwards the embedding kwarg through to the model."
                )
            x_main = x  # already weather + grid only
            x_emb = embedding  # (B, C_emb, factor*H, factor*W)
        else:
            # Legacy concat: split alpha_earth off the trailing channels of x.
            x_main = x[:, : -self.alpha_earth_channels]  # zeros + ERA5
            x_emb = x[:, -self.alpha_earth_channels :]   # embedding (same grid)

        # 2. Reduce the embedding to the weather grid by folding sub-pixels into
        #    channels (no-op when emb_downscale_factor == 1), then run the branch
        #    (concurrently with the level-0 UNet blocks).
        if self.emb_downscale_factor > 1:
            x_emb = F.pixel_unshuffle(x_emb, self.emb_downscale_factor)
        emb_feat = self.emb_branch(x_emb.to(x_main.dtype))  # (B, emb_branch_channels, H, W)

        # 3. Append positional grid embeddings to x_main (from SongUNetPosEmbd)
        if (self.pos_embd is not None) or (self.lt_embd is not None):
            if embedding_selector is not None:
                selected_pos_embd = self.positional_embedding_selector(
                    x_main, embedding_selector, lead_time_label=lead_time_label
                )
            else:
                selected_pos_embd = self.positional_embedding_indexing(
                    x_main, global_index=global_index, lead_time_label=lead_time_label
                )
            x_main = torch.cat((x_main, selected_pos_embd.to(x_main.dtype)), dim=1)
        # x_main is now (B, 20, H, W) = zeros(4) + ERA5(12) + grid(4)

        # 4. SongUNet encoder loop with injection before the first downsampling.
        #    This replicates SongUNet.forward's encoder/decoder logic, adding the
        #    fusion step right before the "_down" block at level-1.
        x = x_main

        # Noise / time-step embedding (embedding_type="zero" for regression)
        if self.embedding_type != "zero":
            emb = self.map_noise(noise_labels)
            emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (
                        torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                    ).to(tmp.dtype)
                emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = silu(self.map_layer0(emb))
            emb = silu(self.map_layer1(emb))
        else:
            emb = torch.zeros(
                (noise_labels.shape[0], self.emb_channels), device=x.device, dtype=x.dtype
            )

        # Encoder — inject emb_feat just before the first downsampling block
        skips = []
        aux = x
        injected = False
        for name, block in self.enc.items():
            # Injection point: fuse main path with satellite embedding features
            # right before the first "_down" block (level-0 → level-1 transition)
            if "_down" in name and not injected:
                x = self.fusion_conv(torch.cat([x, emb_feat], dim=1))
                injected = True

            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            elif "_conv" in name:
                x = block(x)
                if self.additive_pos_embed:
                    x = x + self.spatial_emb.to(dtype=x.dtype)
                skips.append(x)
            else:
                # UNetBlocks (and down/up blocks): use gradient checkpointing if applicable
                if isinstance(block, UNetBlock):
                    if (
                        math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                        > self.checkpoint_threshold
                    ):
                        x = checkpoint(block, x, emb, use_reentrant=False)
                    else:
                        x = block(x, emb)
                else:
                    x = block(x)
                skips.append(x)

        # Decoder — identical to SongUNet.forward decoder loop
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                if (
                    math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                    > self.checkpoint_threshold
                    and "_block" in name
                ) or (
                    math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                    > (self.checkpoint_threshold / 2)
                    and "_up" in name
                ):
                    x = checkpoint(block, x, emb, use_reentrant=False)
                else:
                    x = block(x, emb)

        return aux
