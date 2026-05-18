"""
SongUNetEmbBranch: SongUNetPosEmbd with a parallel CNN branch for satellite embeddings.

The last `alpha_earth_channels` input channels are split off and processed by a
separate shallow CNN in parallel with the UNet's full-resolution (level-0) blocks.
The resulting features are concatenated and fused via a 1x1 conv before the first
downsampling step, giving the satellite embeddings a dedicated processing pathway.

Architecture at level-0 (full resolution):
  Main path:  Conv2d(era5_channels → model_channels) + 4x UNetBlocks
  Emb branch: 2x Conv2d(alpha_earth_channels → emb_branch_channels)
  Fusion:     cat([main, emb]) → Conv2d(model_channels + emb_branch_channels → model_channels)
  Then:       standard downsampling and decoder unchanged

Usage:
  In the corrdiff model config set:
    model_args:
      model_type: "SongUNetEmbBranch"
      alpha_earth_channels: 64
      emb_branch_channels: 64
"""

import math

import numpy as np
import torch
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
        Number of trailing input channels treated as alpha-earth embeddings.
        These are split off and NOT passed to the main UNet encoder.
    emb_branch_channels : int, default=64
        Number of output channels of the parallel CNN branch.
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
        **kwargs,
    ):
        # Parent (SongUNetPosEmbd → SongUNet) sees in_channels WITHOUT alpha_earth
        super().__init__(
            img_resolution=img_resolution,
            in_channels=in_channels - alpha_earth_channels,
            out_channels=out_channels,
            **kwargs,
        )
        self.alpha_earth_channels = alpha_earth_channels
        model_channels = kwargs.get("model_channels", 128)
        num_groups = min(32, emb_branch_channels)

        # Parallel CNN branch — plain torch layers with same-padding convolutions.
        # GroupNorm handles mixed-precision autocast without special setup.
        self.emb_branch = torch.nn.Sequential(
            torch.nn.Conv2d(alpha_earth_channels, emb_branch_channels, 3, padding=1),
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
    ):
        """Forward pass with parallel satellite embedding branch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W).
            The LAST `alpha_earth_channels` channels are the satellite embeddings;
            the preceding channels are ERA5 + zero-filled output placeholder.
        noise_labels, class_labels, global_index, embedding_selector,
        augment_labels, lead_time_label : see SongUNetPosEmbd
        """
        # 1. Split: separate alpha_earth from the main input
        x_main = x[:, : -self.alpha_earth_channels]  # (B, 16, H, W) — zeros + ERA5
        x_emb = x[:, -self.alpha_earth_channels :]   # (B, 64, H, W) — alpha_earth

        # 2. Parallel CNN branch (runs concurrently with level-0 UNet blocks)
        emb_feat = self.emb_branch(x_emb)  # (B, emb_branch_channels, H, W)

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
