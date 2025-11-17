import importlib
from copy import deepcopy
from os import path as osp
from typing import Any

from torch import nn

from traiNNer.losses.gan_loss import (
    g_path_regularize,
    gradient_penalty_loss,
    r1_penalty,
)
from traiNNer.losses.iterative_loss_wrapper import (
    IterativeLossWrapper,
    create_iterative_loss,
)
from traiNNer.losses.loss_wrapper import ModelAsLoss

# Import the new R3GAN loss classes for direct access (registered as "r3gan_loss" and "multi_scale_r3gan_loss")
from traiNNer.losses.r3gan_loss import MultiScaleR3GANLoss, R3GANLoss
from traiNNer.utils import get_root_logger, scandir
from traiNNer.utils.registry import LOSS_REGISTRY

__all__ = [
    "ModelAsLoss",
    "MultiScaleR3GANLoss",
    "R3GANLoss",
    "build_loss",
    "g_path_regularize",
    "gradient_penalty_loss",
    "r1_penalty",
]

# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with '_loss.py'
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(loss_folder)
    if v.endswith("_loss.py")
]
# import all the loss modules
_model_modules = [
    importlib.import_module(f"traiNNer.losses.{file_name}")
    for file_name in loss_filenames
]


def build_loss(loss_opt: dict[str, Any]) -> nn.Module:
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.

    Returns:
        nn.Module: Built loss, potentially wrapped with IterativeLossWrapper
    """
    opt = deepcopy(loss_opt)
    loss_type = opt.pop("type")

    # Special handling for GAN losses with R3GAN
    if loss_type.lower() == "ganloss" and opt.get("gan_type", "").lower() == "r3gan":
        # Use R3GANLoss instead of GANLoss for R3GAN configurations
        loss_type = "r3ganloss"
        logger = get_root_logger()
        logger.info(
            "Using R3GANLoss for gan_type: r3gan configuration.",
            extra={"markup": True},
        )

    # Check if this loss needs iteration-based scheduling BEFORE creating the loss
    schedule_params = [
        "start_iter",
        "target_iter",
        "target_weight",
        "disable_after",
        "schedule_type",
        "warn_on_unused",
    ]
    has_schedule = any(param in loss_opt for param in schedule_params)

    # Extract scheduling parameters to avoid passing them to loss constructor
    schedule_config = {}
    if has_schedule:
        for param in schedule_params:
            if param in opt:
                schedule_config[param] = opt.pop(param)

    # Create the loss with only the parameters it expects
    loss = LOSS_REGISTRY.get(loss_type)(**opt)

    if has_schedule:
        # Wrap with IterativeLossWrapper for iteration-based scheduling
        wrapped_loss = create_iterative_loss(loss, {**loss_opt, **schedule_config})
        logger = get_root_logger()
        logger.info(
            "Loss [bold]%s[/bold] wrapped with IterativeLossWrapper for iteration-based scheduling.",
            loss.__class__.__name__,
            extra={"markup": True},
        )
        return wrapped_loss

    logger = get_root_logger()
    logger.info(
        "Loss [bold]%s[/bold](%s) is created.",
        loss.__class__.__name__,
        opt,
        extra={"markup": True},
    )
    return loss
