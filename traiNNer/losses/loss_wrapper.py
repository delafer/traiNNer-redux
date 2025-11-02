from torch import nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ModelAsLoss(nn.Module):
    """
    A wrapper that treats the model's forward pass as the loss function itself.
    This is required for models like ParagonDiffusion that compute their own loss.
    """

    def forward(self, net_g, gt, lr):
        """
        The trainer will pass the generator network (net_g), ground-truth image (gt),
        and low-resolution image (lr). We then call the generator with the
        arguments it expects for its training forward pass.
        """
        return net_g(gt_image=gt, lr_image=lr)
