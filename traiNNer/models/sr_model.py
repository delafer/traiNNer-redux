import os
import shutil
import warnings
from collections import OrderedDict
from os import path as osp
from typing import Any

import cv2
import torch
from ema_pytorch import EMA
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F  # noqa: N812
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from traiNNer.archs import build_network
from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.losses import build_loss
from traiNNer.losses.contrastive_loss import ContrastiveLoss
from traiNNer.losses.dynamic_loss_scheduling import DynamicLossScheduler
from traiNNer.losses.feature_matching_loss import FeatureMatchingLoss
from traiNNer.losses.r3gan_loss import R3GANLoss
from traiNNer.metrics import calculate_metric
from traiNNer.models.base_model import BaseModel
from traiNNer.utils import get_root_logger, imwrite, tensor2img
from traiNNer.utils.color_util import pixelformat2rgb_pt, rgb2pixelformat_pt
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import loss_type_to_label
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed


class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

        # use amp
        self.use_amp = self.opt.use_amp
        self.use_channels_last = self.opt.use_channels_last
        self.memory_format = (
            torch.channels_last
            if self.use_amp and self.use_channels_last
            else torch.preserve_format
        )
        self.amp_dtype = torch.bfloat16 if self.opt.amp_bf16 else torch.float16
        self.use_compile = self.opt.use_compile

        # define network
        assert opt.network_g is not None, "network_g must be defined"
        self.net_g = build_network({**opt.network_g, "scale": opt.scale})

        # load pretrained models
        if self.opt.path.pretrain_network_g is not None:
            self.load_network(
                self.net_g,
                self.opt.path.pretrain_network_g,
                self.opt.path.strict_load_g,
                self.opt.path.param_key_g,
            )

        self.net_g = self.model_to_device(self.net_g, compile=self.opt.use_compile)

        self.lq: Tensor | None = None
        self.gt: Tensor | None = None
        self.output: Tensor | None = None
        logger = get_root_logger()

        if self.use_amp:
            if self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.warning(
                    "bf16 was enabled for AMP but the current GPU does not support bf16. Falling back to float16 for AMP. Disable bf16 to hide this warning (amp_bf16: false)."
                )
                self.amp_dtype = torch.float16

            network_g_name = opt.network_g["type"]
            if (
                self.amp_dtype == torch.float16
                and network_g_name.lower() in ARCHS_WITHOUT_FP16
            ):
                if torch.cuda.is_bf16_supported():
                    logger.warning(
                        "AMP with fp16 was enabled but network_g [bold]%s[/bold] does not support fp16. Falling back to bf16.",
                        network_g_name,
                        extra={"markup": True},
                    )
                    self.amp_dtype = torch.bfloat16
                else:
                    logger.warning(
                        "AMP with fp16 was enabled but network_g [bold]%s[/bold] does not support fp16. Disabling AMP.",
                        network_g_name,
                        extra={"markup": True},
                    )
                    self.use_amp = False
        elif self.amp_dtype == torch.bfloat16:
            logger.warning(
                "bf16 was enabled without AMP and will have no effect. Enable AMP to use bf16 (use_amp: true)."
            )

        if self.use_amp:
            logger.info(
                "Using Automatic Mixed Precision (AMP) with fp32 and %s.",
                "bf16" if self.amp_dtype == torch.bfloat16 else "fp16",
            )

            if self.use_channels_last:
                logger.info("Using channels last memory format.")

        if self.opt.fast_matmul:
            logger.info(
                "Fast matrix multiplication and convolution operations (fast_matmul) enabled, trading precision for performance."
            )

        if self.is_train and self.opt.train:
            # define network net_d if GAN is enabled
            self.has_gan = False
            gan_opt = self.opt.train.gan_opt

            if not gan_opt:
                if self.opt.train.losses:
                    for loss in self.opt.train.losses:
                        loss_type = loss["type"].lower()
                        if loss_type in [
                            "ganloss",
                            "r3ganloss",
                            "multiscaleganloss",
                            "featurematchingloss",
                        ]:
                            self.has_gan = True
                            break

            if gan_opt:
                if gan_opt.get("loss_weight", 0) != 0:
                    self.has_gan = True

            self.net_d = None
            if self.has_gan:
                if self.opt.train.optim_d is None:
                    raise ValueError(
                        "GAN loss requires discriminator optimizer (optim_d). Define optim_d or disable GAN loss."
                    )
                if self.opt.network_d is None:
                    raise ValueError(
                        "GAN loss requires discriminator network (network_d). Define network_d or disable GAN loss."
                    )
                else:
                    self.net_d = build_network(self.opt.network_d)
                    # load pretrained models
                    if self.opt.path.pretrain_network_d is not None:
                        self.load_network(
                            self.net_d,
                            self.opt.path.pretrain_network_d,
                            self.opt.path.strict_load_d,
                            self.opt.path.param_key_d,
                        )
                    self.net_d = self.model_to_device(self.net_d)

            self.losses = {}

            self.ema_decay = 0
            self.net_g_ema: EMA | None = None

            self.optimizer_g: Optimizer | None = None
            self.optimizer_d: Optimizer | None = None
            self.dynamic_loss_scheduler: DynamicLossScheduler | None = None

            self.init_training_settings()

    def init_training_settings(self) -> None:
        self.net_g.train()
        if self.net_d is not None:
            self.net_d.train()

        train_opt = self.opt.train
        assert train_opt is not None

        logger = get_root_logger()

        enable_gradscaler = self.use_amp and not self.opt.amp_bf16

        self.scaler_g = GradScaler(enabled=enable_gradscaler, device="cuda")
        self.scaler_d = GradScaler(enabled=enable_gradscaler, device="cuda")

        self.accum_iters = self.opt.datasets["train"].accum_iter

        self.adaptive_d = train_opt.adaptive_d
        self.adaptive_d_ema_decay = train_opt.adaptive_d_ema_decay
        self.adaptive_d_threshold = train_opt.adaptive_d_threshold
        self.ema_decay = train_opt.ema_decay

        if self.ema_decay > 0:
            logger.info(
                "Using Exponential Moving Average (EMA) with decay: %s.", self.ema_decay
            )
            assert self.opt.network_g is not None, "network_g must be defined"

            init_net_g_ema = None

            # load pretrained model
            if self.opt.path.pretrain_network_g_ema is not None:
                init_net_g_ema = build_network(
                    {**self.opt.network_g, "scale": self.opt.scale}
                )
                self.load_network(
                    init_net_g_ema,
                    self.opt.path.pretrain_network_g_ema,
                    self.opt.path.strict_load_g,
                    "params_ema",
                )

            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            switch_iter = train_opt.ema_switch_iter
            if switch_iter == 0:
                switch_iter = None
            self.net_g_ema = EMA(
                self.get_bare_model(self.net_g),
                ema_model=init_net_g_ema,
                beta=self.ema_decay,
                allow_different_devices=True,
                update_after_step=train_opt.ema_update_after_step,
                update_every=1,
                power=train_opt.ema_power,
                update_model_with_ema_every=switch_iter,
            ).to(device=self.device, memory_format=self.memory_format)  # pyright: ignore[reportCallIssue]

            assert self.net_g_ema is not None
            self.net_g_ema.step = self.net_g_ema.step.to(device=torch.device("cpu"))

        self.grad_clip = train_opt.grad_clip
        if self.grad_clip:
            logger.info("Gradient clipping is enabled.")

        # define losses

        if train_opt.losses is None:
            train_opt.losses = []
            # old loss format
            old_loss_opts = [
                "pixel_opt",
                "mssim_opt",
                "ms_ssim_l1_opt",
                "perceptual_opt",
                "contextual_opt",
                "dists_opt",
                "hr_inversion_opt",
                "dinov2_opt",
                "topiq_opt",
                "pd_opt",
                "fd_opt",
                "ldl_opt",
                "hsluv_opt",
                "gan_opt",
                "color_opt",
                "luma_opt",
                "avg_opt",
                "bicubic_opt",
            ]
            for opt in old_loss_opts:
                loss = getattr(train_opt, opt)
                if loss is not None:
                    train_opt.losses.append(loss)

        for loss in train_opt.losses:
            assert "type" in loss, "all losses must define type"
            assert "loss_weight" in loss, f"{loss['type']} must define loss_weight"
            if float(loss["loss_weight"]) != 0:
                label = loss_type_to_label(loss["type"])
                if label in {"l_g_gan", "l_g_featurematching"}:
                    self.has_gan = True
                self.losses[label] = build_loss(loss).to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765

                # if self.use_compile:
                #     logger.info(
                #         "Loss %s will be compiled. The first iteration may take several minutes.",
                #         label,
                #     )
                #     self.losses[label] = torch.compile(self.losses[label])

        assert self.losses, "At least one loss must be defined."

        # Initialize dynamic loss scheduler if configured
        if (
            hasattr(train_opt, "dynamic_loss_scheduling")
            and train_opt.dynamic_loss_scheduling is not None
            and train_opt.dynamic_loss_scheduling.get("enabled", False)
        ):
            scheduler_config = (
                train_opt.dynamic_loss_scheduling.copy()
            )  # Make a copy to avoid modifying original

            # Add intelligent context for auto-calibration
            if scheduler_config.get("auto_calibrate", False):
                # Get architecture type
                architecture_type = "unknown"
                if self.opt.network_g and "type" in self.opt.network_g:
                    architecture_type = self.opt.network_g["type"]

                # Get training configuration context
                training_config = {
                    "total_iterations": train_opt.total_iter,
                    "dataset_info": getattr(
                        self.opt.datasets.get("train"), "dataset_info", {}
                    ),
                }

                # Use default dataset complexity values for auto-calibration
                # Note: Automatic dataset analysis is disabled due to method availability issues
                if not training_config["dataset_info"]:
                    logger.info(
                        "📊 Using default dataset complexity for auto-calibration",
                        extra={"markup": True},
                    )
                    training_config["dataset_info"] = {
                        "texture_variance": 0.5,
                        "edge_density": 0.5,
                        "color_variation": 0.5,
                        "overall_complexity": 0.5,
                    }

                # Add context to scheduler config
                scheduler_config["architecture_type"] = architecture_type
                scheduler_config["training_config"] = training_config

                logger.info(
                    f"🧠 Auto-calibration mode enabled for {architecture_type} architecture",
                    extra={"markup": True},
                )

                # Log the detected/used dataset info
                dataset_info = training_config["dataset_info"]
                if isinstance(dataset_info, dict):
                    logger.info(
                        f"📊 Using detected dataset complexity: "
                        f"texture={dataset_info.get('texture_variance', 0.5):.2f}, "
                        f"edges={dataset_info.get('edge_density', 0.5):.2f}, "
                        f"colors={dataset_info.get('color_variation', 0.5):.2f}",
                        extra={"markup": True},
                    )

            try:
                from traiNNer.losses.dynamic_loss_scheduling import (
                    create_dynamic_loss_scheduler,
                )

                self.dynamic_loss_scheduler = create_dynamic_loss_scheduler(
                    self.losses, scheduler_config
                )

                if scheduler_config.get("auto_calibrate", False):
                    logger.info(
                        "✅ Intelligent dynamic loss scheduling ready - optimal parameters auto-configured",
                        extra={"markup": True},
                    )
                else:
                    logger.info(
                        f"Dynamic loss scheduling enabled with config: {scheduler_config}",
                        extra={"markup": True},
                    )
            except Exception as e:
                logger.error(f"Failed to initialize dynamic loss scheduler: {e}")
                logger.warning("Continuing without dynamic loss scheduling")

        if not self.has_gan:
            # warn that discriminator network / optimizer won't be used if enabled
            if self.opt.network_d is not None:
                logger.warning(
                    "Discriminator network (network_d) is defined but GAN loss is disabled. Discriminator network will have no effect."
                )

            if train_opt.optim_d is not None:
                logger.warning(
                    "Discriminator optimizer (optim_d) is defined but GAN loss is disabled. Discriminator optimizer will have no effect."
                )

        # setup batch augmentations
        self.setup_batchaug()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self) -> None:
        train_opt = self.opt.train
        assert train_opt is not None
        # assert train_opt.optim_g is not None
        optim_params = []
        logger = get_root_logger()

        if train_opt.optim_g is not None:
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                elif "eval_" in k:
                    pass  # intentionally frozen for reparameterization, skip warning
                else:
                    logger.warning("Params %s will not be optimized.", k)

            self.optimizer_g = self.get_optimizer(optim_params, train_opt.optim_g)
            self.optimizers.append(self.optimizer_g)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                "SCHEDULEFREE" in train_opt.optim_g["type"].upper()
            )
        else:
            logger.warning("!!! net_g will not be optimized. !!!")

        # optimizer d
        if self.net_d is not None:
            assert train_opt.optim_d is not None
            self.optimizer_d = self.get_optimizer(
                self.net_d.parameters(), train_opt.optim_d
            )
            self.optimizers.append(self.optimizer_d)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                "SCHEDULEFREE" in train_opt.optim_d["type"].upper()
            )

    def feed_data(self, data: DataFeed) -> None:
        assert "lq" in data
        self.lq = data["lq"].to(
            self.device,
            memory_format=self.memory_format,
            non_blocking=True,
        )
        if "gt" in data:
            self.gt = data["gt"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )

        # moa
        if self.is_train and self.batch_augment and self.gt is not None:
            self.gt, self.lq = self.batch_augment(self.gt, self.lq)

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        assert self.lq is not None
        assert self.gt is not None
        assert self.scaler_d is not None
        assert self.scaler_g is not None

        skip_d_update = False

        # optimize net_d
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        n_samples = self.gt.shape[0]
        self.loss_samples += n_samples
        loss_dict: dict[str, Tensor | float] = OrderedDict()

        lq = rgb2pixelformat_pt(self.lq, self.opt.input_pixel_format)
        rgb2pixelformat_pt(self.gt, self.opt.input_pixel_format)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if self.optimizer_g is not None:
                output = self.net_g(lq)
                self.output = pixelformat2rgb_pt(
                    output, self.gt, self.opt.output_pixel_format
                )

                assert isinstance(self.output, Tensor)
                l_g_total = torch.tensor(0.0, device=self.lq.device)

                lq_target = None

                # Prepare images for losses
                real_images_unaug = self.gt.clone()
                fake_images_unaug = self.output.clone()
                real_images_aug = real_images_unaug
                fake_images_aug = fake_images_unaug
                if self.batch_augment:
                    real_images_aug, fake_images_aug = self.batch_augment(
                        real_images_aug, fake_images_aug
                    )

                # First pass: compute all losses without dynamic weighting
                raw_losses = {}
                for label, loss in self.losses.items():
                    target = real_images_aug

                    if loss.loss_weight < 0:
                        if lq_target is None:
                            with torch.inference_mode():
                                lq_target = torch.clamp(
                                    F.interpolate(
                                        self.lq,
                                        scale_factor=self.opt.scale,
                                        mode="bicubic",
                                        antialias=True,
                                    ),
                                    0,
                                    1,
                                )
                        target = lq_target

                    # --- GAN LOSS (FIXED for Wrapper) ---
                    if label == "l_g_gan":
                        assert self.net_d is not None

                        # Robust check for R3GAN inside wrapper
                        is_r3gan = isinstance(loss, R3GANLoss) or (
                            hasattr(loss, "loss_module")
                            and isinstance(loss.loss_module, R3GANLoss)
                        )

                        if is_r3gan:
                            # R3GAN Signature
                            if hasattr(loss, "loss_module"):  # Wrapped
                                l_g_loss = loss(
                                    net_d=self.net_d,
                                    real_images=self.gt,
                                    fake_images=self.output,
                                    is_disc=False,
                                    current_iter=current_iter,
                                )
                            else:  # Unwrapped
                                l_g_loss = loss(
                                    net_d=self.net_d,
                                    real_images=self.gt,
                                    fake_images=self.output,
                                    is_disc=False,
                                )
                        else:
                            # Standard GAN Signature
                            fake_g_pred = self.net_d(self.output)
                            if hasattr(loss, "loss_module"):  # Wrapped
                                l_g_loss = loss(
                                    fake_g_pred,
                                    True,
                                    is_disc=False,
                                    current_iter=current_iter,
                                )
                            else:  # Unwrapped
                                l_g_loss = loss(fake_g_pred, True, is_disc=False)

                        if self.adaptive_d:
                            l_g_gan_ema = (
                                self.adaptive_d_ema_decay * self.l_g_gan_ema
                                + (1 - self.adaptive_d_ema_decay) * l_g_loss.detach()
                            )
                            if (
                                l_g_gan_ema
                                > self.l_g_gan_ema * self.adaptive_d_threshold
                            ):
                                skip_d_update = True
                                self.optimizers_skipped[1] = True
                            self.l_g_gan_ema = l_g_gan_ema

                    # --- LDL LOSS ---
                    elif label == "l_g_ldl":
                        assert self.net_g_ema is not None
                        with torch.inference_mode():
                            output_ema = pixelformat2rgb_pt(
                                self.net_g_ema(lq),
                                self.gt,
                                self.opt.output_pixel_format,
                            )
                        if hasattr(loss, "loss_module"):
                            l_g_loss = loss(
                                self.output,
                                output_ema,
                                target,
                                current_iter=current_iter,
                            )
                        else:
                            l_g_loss = loss(self.output, output_ema, target)

                    # --- CONTRASTIVE LOSS (FIXED for Wrapper) ---
                    elif isinstance(loss, ContrastiveLoss) or (
                        hasattr(loss, "loss_module")
                        and isinstance(loss.loss_module, ContrastiveLoss)
                    ):
                        if hasattr(loss, "loss_module"):
                            l_g_loss = loss(
                                self.output, target, self.lq, current_iter=current_iter
                            )
                        else:
                            l_g_loss = loss(self.output, target, self.lq)

                    # --- FEATURE MATCHING (FIXED for Wrapper) ---
                    elif isinstance(loss, FeatureMatchingLoss) or (
                        hasattr(loss, "loss_module")
                        and isinstance(loss.loss_module, FeatureMatchingLoss)
                    ):
                        assert self.net_d is not None
                        _real_pred, real_feats = self.net_d.forward_with_features(
                            self.gt
                        )
                        _fake_pred, fake_feats = self.net_d.forward_with_features(
                            self.output
                        )

                        if hasattr(loss, "loss_module"):
                            l_g_loss = loss(
                                real_feats, fake_feats, current_iter=current_iter
                            )
                        else:
                            l_g_loss = loss(real_feats, fake_feats)

                    # --- GENERIC LOSSES ---
                    elif hasattr(loss, "loss_module"):
                        l_g_loss = loss(self.output, target, current_iter=current_iter)
                    else:
                        l_g_loss = loss(self.output, target)

                    # Store raw loss for dynamic scheduling
                    raw_losses[label] = l_g_loss

                # Apply dynamic loss scheduling if enabled
                dynamic_weights = {}
                if self.dynamic_loss_scheduler is not None:
                    dynamic_weights = self.dynamic_loss_scheduler(
                        raw_losses, current_iter
                    )

                # Second pass: accumulate losses with dynamic weights
                for label, loss in self.losses.items():
                    l_g_loss = raw_losses[label]
                    base_weight = abs(loss.loss_weight)

                    # Apply dynamic adjustment if available
                    if (
                        self.dynamic_loss_scheduler is not None
                        and label in dynamic_weights
                    ):
                        dynamic_adjustment = dynamic_weights[label]
                        adjusted_weight = base_weight * dynamic_adjustment
                    else:
                        adjusted_weight = base_weight

                    # Accumulate Loss with adjusted weight
                    if isinstance(l_g_loss, dict):
                        for sublabel, loss_val in l_g_loss.items():
                            if loss_val > 0:
                                weighted_loss_val = loss_val * adjusted_weight
                                l_g_total += weighted_loss_val * self.accum_iters
                                loss_dict[f"{label}_{sublabel}"] = weighted_loss_val
                    else:
                        weighted_l_g_loss = l_g_loss * adjusted_weight
                        l_g_total += weighted_l_g_loss / self.accum_iters
                        loss_dict[label] = weighted_l_g_loss

                if not l_g_total.isfinite():
                    raise RuntimeError("Training failed: NaN/Inf found in loss.")

                loss_dict["l_g_total"] = l_g_total

                # Log dynamic loss weights if monitoring is enabled
                if self.dynamic_loss_scheduler is not None:
                    stats = self.dynamic_loss_scheduler.get_monitoring_stats()
                    for loss_name, weight in stats["current_weights"].items():
                        loss_dict[f"dynamic_weight_{loss_name}"] = weight

                # Update training automations with loss tracking
                self.update_automation_loss_tracking(
                    float(l_g_total.detach().cpu()), current_iter
                )

                self.scaler_g.scale(l_g_total).backward()

                if apply_gradient:
                    self.scaler_g.unscale_(self.optimizer_g)

                    # Collect gradients for monitoring
                    gradients = [
                        p.grad for p in self.net_g.parameters() if p.grad is not None
                    ]

                    # Update gradient monitoring for automation
                    suggested_threshold = self.update_automation_gradient_monitoring(
                        gradients
                    )

                    grad_norm_g = torch.linalg.vector_norm(
                        torch.stack(
                            [
                                torch.linalg.vector_norm(p.grad, 2)
                                for p in self.net_g.parameters()
                                if p.grad is not None
                            ]
                        )
                    ).detach()
                    loss_dict["grad_norm_g"] = grad_norm_g

                    if self.grad_clip:
                        # Use automation-based gradient clipping threshold if available
                        clip_threshold = self.get_automation_clipping_threshold()
                        clip_grad_norm_(self.net_g.parameters(), clip_threshold)
                        loss_dict["grad_clip_threshold"] = clip_threshold

                    scale_before = self.scaler_g.get_scale()
                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
                    scale_after = self.scaler_g.get_scale()
                    loss_dict["scale_g"] = scale_after
                    self.optimizers_skipped[0] = scale_after < scale_before
                    self.optimizer_g.zero_grad()
            else:
                with torch.inference_mode():
                    self.output = self.net_g(self.lq)

        # --- DISCRIMINATOR UPDATE ---
        cri_gan = self.losses.get("l_g_gan")
        if (
            self.net_d is not None
            and cri_gan is not None
            and self.optimizer_d is not None
            and not skip_d_update
        ):
            for p in self.net_d.parameters():
                p.requires_grad = True

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                # Unwrap GAN loss safely
                loss_module = (
                    cri_gan.loss_module if hasattr(cri_gan, "loss_module") else cri_gan
                )

                if isinstance(loss_module, R3GANLoss):
                    loss_d_dict = loss_module(
                        net_d=self.net_d,
                        real_images=real_images_aug,
                        fake_images=fake_images_aug.detach(),
                        real_images_unaug=real_images_unaug,
                        fake_images_unaug=fake_images_unaug.detach(),
                        is_disc=True,
                    )
                    l_d_total = loss_d_dict["d_loss"]
                    loss_dict.update(
                        {k: v for k, v in loss_d_dict.items() if k != "d_loss"}
                    )
                else:
                    real_d_pred = self.net_d(self.gt)
                    fake_d_pred = self.net_d(self.output.detach())
                    l_d_real = loss_module(real_d_pred, True, is_disc=True)
                    l_d_fake = loss_module(fake_d_pred, False, is_disc=True)

                    # Apply schedule weight manually for Discriminator side
                    weight = (
                        cri_gan.get_current_weight(current_iter)
                        if hasattr(cri_gan, "get_current_weight")
                        else cri_gan.loss_weight
                    )
                    l_d_total = (l_d_real + l_d_fake) * weight
                    loss_dict["l_d_real"] = l_d_real
                    loss_dict["l_d_fake"] = l_d_fake

            self.scaler_d.scale(l_d_total / self.accum_iters).backward()

            if apply_gradient:
                self.scaler_d.unscale_(self.optimizer_d)

                # Collect discriminator gradients for monitoring
                gradients_d = [
                    p.grad for p in self.net_d.parameters() if p.grad is not None
                ]

                # Update gradient monitoring for automation (includes discriminator)
                suggested_threshold = self.update_automation_gradient_monitoring(
                    gradients_d
                )

                grad_norm_d = torch.linalg.vector_norm(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(p.grad, 2)
                            for p in self.net_d.parameters()
                            if p.grad is not None
                        ]
                    )
                ).detach()
                loss_dict["grad_norm_d"] = grad_norm_d

                if self.grad_clip:
                    # Use automation-based gradient clipping threshold
                    clip_threshold = self.get_automation_clipping_threshold()
                    clip_grad_norm_(self.net_d.parameters(), clip_threshold)
                scale_before = self.scaler_d.get_scale()
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
                scale_after = self.scaler_d.get_scale()
                loss_dict["scale_d"] = scale_after
                self.optimizers_skipped[-1] = scale_after < scale_before
                self.optimizer_d.zero_grad()

        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()
            )
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples

        # Add enhanced logging information
        enhanced_logging_stats = self._collect_enhanced_logging_stats(
            loss_dict, current_iter, gradients if apply_gradient else []
        )
        self.log_dict.update(enhanced_logging_stats)

        self.log_dict = self.reduce_loss_dict(self.log_dict)

        if self.net_g_ema is not None and apply_gradient:
            if not (self.use_amp and self.optimizers_skipped[0]):
                self.net_g_ema.update()

    def infer_tiled(self, net: nn.Module, lq: torch.Tensor) -> torch.Tensor:
        assert self.opt.val is not None
        tile_size = self.opt.val.tile_size
        tile_overlap = self.opt.val.tile_overlap
        scale = self.opt.scale

        b, c, h, w = lq.shape
        assert b == 1, "Only batch size 1 is supported for tiled inference"

        if h <= tile_size and w <= tile_size:
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                return net(lq)

        pad_h = (tile_size - (h % tile_size)) % tile_size if h > tile_size else 0
        pad_w = (tile_size - (w % tile_size)) % tile_size if w > tile_size else 0

        lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, h_pad, w_pad = lq.shape

        output = torch.zeros((1, c, h_pad * scale, w_pad * scale), device=lq.device)
        weight_map = torch.zeros_like(output)

        hr_tile = tile_size * scale
        wy = torch.linspace(0, 1, hr_tile, device=lq.device)
        wx = torch.linspace(0, 1, hr_tile, device=lq.device)
        wy = 1 - torch.abs(wy - 0.5) * 2
        wx = 1 - torch.abs(wx - 0.5) * 2
        weight = torch.ger(wy, wx).unsqueeze(0).unsqueeze(0)

        stride = tile_size - tile_overlap
        tiles_y = max(1, (h_pad - tile_overlap + stride - 1) // stride)
        tiles_x = max(1, (w_pad - tile_overlap + stride - 1) // stride)

        for y in range(tiles_y):
            for x in range(tiles_x):
                in_y0 = y * stride
                in_x0 = x * stride
                in_y1 = min(in_y0 + tile_size, h_pad)
                in_x1 = min(in_x0 + tile_size, w_pad)

                lq_patch = lq[:, :, in_y0:in_y1, in_x0:in_x1]

                ph, pw = lq_patch.shape[-2:]
                pad_bottom = max(tile_size - ph, 0) if ph < tile_size else 0
                pad_right = max(tile_size - pw, 0) if pw < tile_size else 0

                if pad_bottom > 0 or pad_right > 0:
                    pad_bottom = min(pad_bottom, ph - 1)
                    pad_right = min(pad_right, pw - 1)
                    lq_patch = torch.nn.functional.pad(
                        lq_patch,
                        (0, pad_right, 0, pad_bottom),
                        mode="reflect",
                    )

                out_patch = net(lq_patch)
                out_patch = out_patch[:, :, : ph * scale, : pw * scale]
                w_patch = weight[:, :, : ph * scale, : pw * scale]

                out_y0 = in_y0 * scale
                out_x0 = in_x0 * scale
                out_y1 = out_y0 + ph * scale
                out_x1 = out_x0 + pw * scale

                output[:, :, out_y0:out_y1, out_x0:out_x1] += out_patch * w_patch
                weight_map[:, :, out_y0:out_y1, out_x0:out_x1] += w_patch

        out_final = output / weight_map.clamp(min=1e-6)
        return out_final[:, :, : h * scale, : w * scale]

    def test(self) -> None:
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if self.optimizers_schedule_free and self.optimizers_schedule_free[0]:
                assert self.optimizer_g is not None
                self.optimizer_g.eval()  # pyright: ignore[reportAttributeAccessIssue]

            assert self.lq is not None

            lq = rgb2pixelformat_pt(
                self.lq, self.opt.input_pixel_format
            )  # lq: input_pixel_format

            # Validation downscaling disabled for accurate metrics
            # Original VRAM fix was causing validation on heavily downscaled images (64x64 max)
            # which led to inaccurate PSNR/SSIM calculations and declining metric curves
            if not self.is_train:  # Only during validation/testing
                # Validate with full-resolution images for accurate metrics
                # If you encounter VRAM issues during validation, consider:
                # 1. Reducing batch size for validation
                # 2. Using smaller validation tile sizes
                # 3. Processing validation in smaller chunks
                pass

            net = self.net_g_ema if self.net_g_ema is not None else self.net_g
            net.eval()

            assert self.opt.val is not None
            with torch.inference_mode():
                if self.opt.val.tile_size > 0:
                    tmp_out = self.infer_tiled(net, lq)
                else:
                    tmp_out = net(lq)
                self.output = pixelformat2rgb_pt(
                    tmp_out, self.gt, self.opt.output_pixel_format
                )

            if self.net_g_ema is None:
                net.train()

            if self.optimizers_schedule_free and self.optimizers_schedule_free[0]:
                assert self.optimizer_g is not None
                self.optimizer_g.train()  # pyright: ignore[reportAttributeAccessIssue]

    def dist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        if self.opt.rank == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, multi_val_datasets
            )

    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        self.is_train = False

        assert isinstance(dataloader.dataset, BaseDataset)
        assert self.opt.val is not None
        assert self.opt.path.visualization is not None

        dataset_name = dataloader.dataset.opt.name

        if self.with_metrics:
            assert self.opt.val.metrics is not None
            if len(self.metric_results) == 0:  # only execute in the first run
                self.metric_results: dict[str, Any] = dict.fromkeys(
                    self.opt.val.metrics.keys(), 0
                )
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if self.with_metrics:
            self.metric_results = dict.fromkeys(self.metric_results, 0)

        metric_data = {}
        pbar = None
        if self.use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        logger = get_root_logger()
        if save_img and len(dataloader) > 0:
            logger.info(
                "Saving %d validation images to %s.",
                len(dataloader),
                clickable_file_path(
                    self.opt.path.visualization, "visualization folder"
                ),
            )

        gt_key = "img2"
        run_metrics = self.with_metrics

        for val_data in dataloader:
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)

            try:
                self.test()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning(
                    f"OOM during validation of {img_name}. Switching to tiled inference (tile_size=256)."
                )
                original_tile_size = self.opt.val.tile_size
                # Force tile size for this image
                self.opt.val.tile_size = 256
                try:
                    self.test()
                except Exception as e:
                    logger.error(f"Failed to validate {img_name} even with tiling: {e}")
                finally:
                    # Restore original setting (or keep it if we want to be safe for subsequent images)
                    # Keeping it might be safer for the rest of validation
                    pass
            except Exception as e:
                logger.error(f"Error during validation of {img_name}: {e}")
                continue

            visuals = self.get_current_visuals()
            sr_img = tensor2img(
                visuals["result"],
                to_bgr=False,
            )
            metric_data["img"] = sr_img
            if "gt" in visuals:
                gt_img = tensor2img(
                    visuals["gt"],
                    to_bgr=False,
                )
                metric_data[gt_key] = gt_img
                self.gt = None
            # else:
            #     run_metrics = False

            # tentative for out of GPU memory
            self.lq = None
            self.output = None
            torch.cuda.empty_cache()

            save_img_dir = None

            if save_img:
                if self.opt.is_train:
                    if multi_val_datasets:
                        save_img_dir = osp.join(
                            self.opt.path.visualization, f"{dataset_name} - {img_name}"
                        )
                    else:
                        assert dataloader.dataset.opt.dataroot_lq is not None, (
                            "dataroot_lq is required for val set"
                        )
                        lq_path = val_data["lq_path"][0]

                        # multiple root paths are supported, find the correct root path for each lq_path
                        normalized_lq_path = osp.normpath(lq_path)

                        matching_root = None
                        for root in dataloader.dataset.opt.dataroot_lq:
                            normalized_root = osp.normpath(root)
                            if normalized_lq_path.startswith(normalized_root + osp.sep):
                                matching_root = root
                                break

                        if matching_root is None:
                            raise ValueError(
                                f"The lq_path {lq_path} does not match any of the provided dataroot_lq paths."
                            )

                        save_img_dir = osp.join(
                            self.opt.path.visualization,
                            osp.relpath(
                                osp.splitext(lq_path)[0],
                                matching_root,
                            ),
                        )
                    save_img_path = osp.join(
                        save_img_dir, f"{img_name}_{current_iter:06d}.png"
                    )
                elif self.opt.val.suffix:
                    save_img_path = osp.join(
                        self.opt.path.visualization,
                        dataset_name,
                        f"{img_name}_{self.opt.val.suffix}.png",
                    )
                else:
                    save_img_path = osp.join(
                        self.opt.path.visualization,
                        dataset_name,
                        f"{img_name}.png",
                    )
                imwrite(cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR), save_img_path)
                if (
                    self.opt.is_train
                    and not self.first_val_completed
                    and "lq_path" in val_data
                ):
                    assert save_img_dir is not None
                    lr_img_target_path = osp.join(save_img_dir, f"{img_name}_lr.png")
                    if not os.path.exists(lr_img_target_path):
                        shutil.copy(val_data["lq_path"][0], lr_img_target_path)

            if run_metrics:
                # calculate metrics
                assert self.opt.val.metrics is not None
                for name, opt_ in self.opt.val.metrics.items():
                    result = calculate_metric(metric_data, opt_, self.device)
                    # logger.info("%d %s/%s: %f", current_iter, name, img_name, result)
                    self.metric_results[name] += result
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        if pbar is not None:
            pbar.close()

        if run_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= len(dataloader)
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        self.first_val_completed = True
        self.is_train = True

    def _log_validation_metric_values(
        self, current_iter: int, dataset_name: str, tb_logger: SummaryWriter | None
    ) -> None:
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric:<5}: {value:7.4f}"
            if len(self.best_metric_results) > 0:
                log_str += (
                    f"\tBest: {self.best_metric_results[dataset_name][metric]['val']:7.4f} @ "
                    f"{self.best_metric_results[dataset_name][metric]['iter']:9,} iter"
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def get_current_visuals(self) -> dict[str, Tensor]:
        assert self.output is not None
        assert self.lq is not None

        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()

        if self.gt is not None:
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(
        self,
        epoch: int,
        current_iter: int,
    ) -> None:
        assert self.opt.path.models is not None
        assert self.opt.path.resume_models is not None

        if self.net_g_ema is not None:
            assert isinstance(self.net_g_ema.ema_model, nn.Module)
            self.save_network(
                self.net_g_ema.ema_model,
                "net_g_ema",
                self.opt.path.models,
                current_iter,
                "params_ema",
            )

            self.save_network(
                self.net_g, "net_g", self.opt.path.resume_models, current_iter, "params"
            )
        else:
            self.save_network(
                self.net_g, "net_g", self.opt.path.models, current_iter, "params"
            )

        if self.net_d is not None:
            self.save_network(
                self.net_d, "net_d", self.opt.path.resume_models, current_iter, "params"
            )

        self.save_training_state(epoch, current_iter)

    def clean_gpu(self) -> None:
        """Deep clean GPU memory by nulling tensor references and clearing cache."""
        self.lq = None
        self.gt = None
        self.output = None
        super().clean_gpu()

    def _collect_enhanced_logging_stats(
        self,
        loss_dict: dict[str, Any],
        current_iter: int,
        gradients: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Collect enhanced logging statistics for comprehensive training monitoring."""
        enhanced_stats = {}

        # Dynamic loss scheduling statistics
        if self.dynamic_loss_scheduler is not None:
            try:
                stats = self.dynamic_loss_scheduler.get_monitoring_stats()
                enhanced_stats["training_automation_stats"] = {
                    "dynamic_loss_scheduling": stats
                }
            except Exception:
                # Fallback if dynamic loss scheduler fails
                enhanced_stats["training_automation_stats"] = {
                    "dynamic_loss_scheduling": {"enabled": True, "error": True}
                }

        # Training automation statistics
        automation_stats = {}
        try:
            # Check for automation manager if available
            if hasattr(self, "training_automation_manager"):
                automation_stats = (
                    self.training_automation_manager.get_automation_stats()
                )
        except Exception:
            pass

        # Add automation stats if available
        if automation_stats:
            enhanced_stats["training_automation_stats"] = automation_stats

        # Gradient monitoring statistics
        gradient_stats = {}
        if gradients:
            try:
                total_norm = torch.sqrt(
                    sum(torch.sum(g**2) for g in gradients if g is not None)
                )
                gradient_stats["grad_norm_g"] = float(total_norm.item())
                gradient_stats["num_parameters"] = len(
                    [g for g in gradients if g is not None]
                )

                # Add gradient clipping threshold if available
                if hasattr(self, "grad_clip") and self.grad_clip:
                    clip_threshold = getattr(
                        self, "get_automation_clipping_threshold", lambda: 1.0
                    )()
                    if callable(clip_threshold):
                        clip_threshold = clip_threshold()
                    gradient_stats["grad_clip_threshold"] = clip_threshold

            except Exception:
                gradient_stats["grad_norm_g"] = 0.0

        enhanced_stats["gradient_stats"] = gradient_stats

        # Add current VRAM usage for logging
        try:
            current_vram = torch.cuda.memory_allocated() / (1024**3)
            enhanced_stats["current_vram_gb"] = current_vram
        except Exception:
            enhanced_stats["current_vram_gb"] = 0.0

        return enhanced_stats
