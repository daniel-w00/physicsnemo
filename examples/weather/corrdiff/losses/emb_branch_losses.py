"""
Embedding-aware loss wrappers for the separate-tensor emb-branch path.

The physicsnemo losses (`RegressionLoss`, `ResidualLoss`) have fixed `__call__`
signatures and call ``net(x, img_lr, ...)`` with no way to pass an extra tensor.
For N1 that is fine — the embedding rides inside ``img_lr``. For N8 the embedding
is a SEPARATE, higher-resolution tensor (see ``datasets/cwb.py`` ``embedding_separate``
and ``models/song_unet_emb_branch.py``), so it must reach the model as an
``embedding=`` kwarg instead.

Rather than copy (and risk drift from) the physicsnemo loss bodies, these wrappers
inject the embedding by wrapping the ``net`` (and, for the diffusion loss, the frozen
``regression_net``) in a thin callable that adds ``embedding=`` to every forward call.
Both corrdiff preconditioners (``UNet`` for regression, ``EDMPrecondSuperResolution``
for diffusion) forward ``**model_kwargs`` to the inner model, so the kwarg reaches
``SongUNetEmbBranch.forward`` unchanged.

When ``embedding`` is None these behave exactly like the physicsnemo base classes,
so they are safe drop-in replacements for the N1 / no-embedding paths too.
"""

from physicsnemo.metrics.diffusion import RegressionLoss, ResidualLoss


class _EmbInjector:
    """Wrap a precond/model callable so a fixed ``embedding`` kwarg is injected
    into every forward call. Attribute access is delegated to the wrapped object,
    so the loss bodies (which only *call* the net) see no difference otherwise."""

    def __init__(self, net, embedding):
        # Use __dict__ directly so __getattr__ never recurses on these names.
        self.__dict__["_net"] = net
        self.__dict__["_embedding"] = embedding

    def __call__(self, *args, **kwargs):
        if self._embedding is not None:
            kwargs.setdefault("embedding", self._embedding)
        return self._net(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__dict__["_net"], name)


class EmbRegressionLoss(RegressionLoss):
    """`RegressionLoss` that accepts a separate `embedding` tensor and forwards it
    to the model via the precond's `**model_kwargs` passthrough."""

    def __call__(
        self,
        net,
        img_clean,
        img_lr,
        augment_pipe=None,
        lead_time_label=None,
        embedding=None,
    ):
        if embedding is not None:
            net = _EmbInjector(net, embedding)
        return super().__call__(
            net=net,
            img_clean=img_clean,
            img_lr=img_lr,
            augment_pipe=augment_pipe,
            lead_time_label=lead_time_label,
        )


class EmbResidualLoss(ResidualLoss):
    """`ResidualLoss` (diffusion) that accepts a separate `embedding` tensor.

    The embedding is injected into BOTH the diffusion `net` and the frozen
    `regression_net` (which forms the conditioning mean) — this assumes the
    regression net is itself an emb-branch model trained with the same embedding,
    i.e. a consistent emb-branch regression+diffusion pair.

    Patching is not yet supported together with a separate high-res embedding: the
    embedding would need patch-aligned cropping at `emb_downscale_factor` resolution.
    """

    def __call__(
        self,
        net,
        img_clean,
        img_lr,
        patching=None,
        lead_time_label=None,
        augment_pipe=None,
        use_patch_grad_acc=False,
        embedding=None,
    ):
        if embedding is None:
            return super().__call__(
                net=net,
                img_clean=img_clean,
                img_lr=img_lr,
                patching=patching,
                lead_time_label=lead_time_label,
                augment_pipe=augment_pipe,
                use_patch_grad_acc=use_patch_grad_acc,
            )

        if patching is not None:
            raise NotImplementedError(
                "Separate high-res embeddings (embedding_separate=True) are not yet "
                "supported with patching; the embedding needs patch-aligned cropping "
                "at emb_downscale_factor resolution. Use the N1 concat path with "
                "patching, or run the emb-branch diffusion on the full domain."
            )

        net = _EmbInjector(net, embedding)
        orig_regression_net = self.regression_net
        self.regression_net = _EmbInjector(orig_regression_net, embedding)
        try:
            return super().__call__(
                net=net,
                img_clean=img_clean,
                img_lr=img_lr,
                patching=patching,
                lead_time_label=lead_time_label,
                augment_pipe=augment_pipe,
                use_patch_grad_acc=use_patch_grad_acc,
            )
        finally:
            self.regression_net = orig_regression_net
