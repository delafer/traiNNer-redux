import os

from torch.distributed import destroy_process_group

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from traiNNer.check.check_dependencies import check_dependencies

if __name__ == "__main__":
    check_dependencies()


import argparse
import datetime
import gc
import logging
import math
import signal
import sys
import time
from os import path as osp
from types import FrameType
from typing import Any

import torch
from rich.traceback import install
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from traiNNer.data import build_dataloader, build_dataset
from traiNNer.data.data_sampler import EnlargedSampler
from traiNNer.data.paired_image_dataset import PairedImageDataset
from traiNNer.data.paired_video_dataset import PairedVideoDataset
from traiNNer.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from traiNNer.models import build_model
from traiNNer.utils import (
    AvgTimer,
    MessageLogger,
    check_resume,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    scandir,
)
from traiNNer.utils.config import Config
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import (
    free_space_gb_str,
    set_random_seed,
)
from traiNNer.utils.options import copy_opt_file, diff_user_vs_template
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import TrainingState


def init_tb_loggers(opt: ReduxOptions) -> SummaryWriter | None:
    # initialize wandb logger before tensorboard logger to allow proper sync
    assert opt.logger is not None
    assert opt.root_path is not None

    if (opt.logger.wandb is not None) and (opt.logger.wandb.project is not None):
        assert opt.logger.use_tb_logger, "should turn on tensorboard when using wandb"
        init_wandb_logger(opt)
    tb_logger = None
    if opt.logger.use_tb_logger:
        tb_logger = init_tb_logger(
            log_dir=osp.join(opt.root_path, "tb_logger", opt.name)
        )
    return tb_logger


def create_train_val_dataloader(
    opt: ReduxOptions,
    args: argparse.Namespace,
    val_enabled: bool,
    logger: logging.Logger,
) -> tuple[DataLoader | None, EnlargedSampler | None, list[DataLoader], int, int]:
    assert isinstance(opt.num_gpu, int)
    assert opt.world_size is not None
    assert opt.dist is not None

    # create train and val dataloaders
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = (
        None,
        None,
        [],
        0,
        0,
    )
    for phase, dataset_opt in opt.datasets.items():
        if phase == "train":
            assert opt.train is not None
            assert dataset_opt.batch_size_per_gpu is not None

            if dataset_opt.gt_size is None and dataset_opt.lq_size is not None:
                dataset_opt.gt_size = dataset_opt.lq_size * opt.scale
            elif dataset_opt.lq_size is None and dataset_opt.gt_size is not None:
                dataset_opt.lq_size = dataset_opt.gt_size // opt.scale
            else:
                raise ValueError(
                    "Exactly one of gt_size or lq_size must be defined in the train dataset"
                )

            train_set = build_dataset(dataset_opt)
            dataset_enlarge_ratio = dataset_opt.dataset_enlarge_ratio
            if dataset_enlarge_ratio == "auto":
                dataset_enlarge_ratio = max(
                    2000 * dataset_opt.batch_size_per_gpu // len(train_set), 1
                )
            train_sampler = EnlargedSampler(
                train_set, opt.world_size, opt.rank, dataset_enlarge_ratio
            )
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt.num_gpu,
                dist=opt.dist,
                sampler=train_sampler,
                seed=opt.manual_seed,
                opt=opt,
            )

            iter_per_epoch = (
                len(train_set)
                * dataset_enlarge_ratio
                // (
                    dataset_opt.batch_size_per_gpu
                    * dataset_opt.accum_iter
                    * opt.world_size
                )
            )

            opt.switch_iter_per_epoch = len(train_set) // (
                dataset_opt.batch_size_per_gpu * dataset_opt.accum_iter * opt.world_size
            )

            total_iters = int(opt.train.total_iter)
            total_epochs = math.ceil(total_iters / (iter_per_epoch))
            assert dataset_opt.gt_size is not None, "gt_size is required for train set"
            logger.info(
                "Training statistics for [b]%s[/b]:\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s",
                opt.name,
                f"Number of train {train_set.label}:",
                f"{len(train_set):,}",
                "Dataset enlarge ratio:",
                f"{dataset_enlarge_ratio:,}",
                "Batch size per gpu:",
                f"{dataset_opt.batch_size_per_gpu:,}",
                "Accumulate iterations:",
                f"{dataset_opt.accum_iter:,}",
                "HR crop size:",
                f"{dataset_opt.gt_size:,}",
                "LR crop size:",
                f"{dataset_opt.lq_size:,}",
                "World size (gpu number):",
                f"{opt.world_size:,}",
                "Require iter per epoch:",
                f"{iter_per_epoch:,}",
                "Total epochs:",
                f"{total_epochs:,}",
                "Total iters:",
                f"{total_iters:,}",
            )
            if len(train_set) < 100:
                logger.warning(
                    "Number of training %s is low: %d, training quality may be impacted. Please use more training %s for best training results.",
                    train_set.label,
                    len(train_set),
                    train_set.label,
                )
        elif phase.split("_")[0] == "val":
            if val_enabled:
                val_set = build_dataset(dataset_opt)
                val_loader = build_dataloader(
                    val_set,
                    dataset_opt,
                    num_gpu=opt.num_gpu,
                    dist=opt.dist,
                    sampler=None,
                    seed=opt.manual_seed,
                    opt=opt,
                )
                logger.info(
                    "Number of val images/folders in %s: %d",
                    dataset_opt.name,
                    len(val_set),
                )
                val_loaders.append(val_loader)
            else:
                logger.info(
                    "Validation is disabled, skip building val dataset %s.",
                    dataset_opt.name,
                )
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt: ReduxOptions) -> Any | None:
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join("experiments", opt.name, "training_states")
        if osp.isdir(state_path):
            states = list(
                scandir(state_path, suffix="state", recursive=False, full_path=False)
            )
            if len(states) != 0:
                states = [
                    [int(x) for x in v.split(".state")[0].split("_")] for v in states
                ]

                resume_state_path = osp.join(
                    state_path, f"{'_'.join([str(x) for x in max(states)])}.state"
                )
                opt.path.resume_state = resume_state_path
    elif opt.path.resume_state:
        resume_state_path = opt.path.resume_state

    if resume_state_path is None:
        resume_state: TrainingState | None = None
    else:
        resume_state = torch.load(
            resume_state_path,
            map_location="cpu",
            weights_only=True,
        )
        assert resume_state is not None

        check_resume(
            opt,
            resume_state["iter"],
        )
    return resume_state


def train_pipeline(root_path: str) -> None:
    install()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please ensure that you have a GPU with CUDA support "
            "and that you have installed the correct CUDA-enabled version of PyTorch. "
            "You can check the installation guide at https://pytorch.org/get-started/locally/"
        )

    # parse options, set distributed setting, set random seed
    opt, args = Config.load_config_from_file(root_path, is_train=True)
    opt.root_path = root_path

    assert opt.train is not None
    assert opt.logger is not None
    assert opt.manual_seed is not None
    assert opt.rank is not None
    assert opt.path.experiments_root is not None
    assert opt.path.log is not None

    torch.cuda.set_per_process_memory_fraction(fraction=1.0)

    if opt.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if opt.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
    assert opt.manual_seed is not None
    set_random_seed(opt.manual_seed + opt.rank)

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    make_exp_dirs(opt, resume_state is not None)
    # mkdir for experiments and logger
    if resume_state is None:
        if opt.logger.use_tb_logger and opt.rank == 0:
            mkdir_and_rename(osp.join(opt.root_path, "tb_logger", opt.name))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt.path.experiments_root)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized

    # Ensure log directory exists before creating log file
    log_file = osp.join(opt.path.log, f"train_{opt.name}_{get_time_str()}.log")
    os.makedirs(opt.path.log, exist_ok=True)

    # Debug log file creation
    print(f"🔍 Debug: Log file path: {log_file}")
    print(f"🔍 Debug: Log directory exists: {os.path.exists(opt.path.log)}")
    print(f"🔍 Debug: Log directory writable: {os.access(opt.path.log, os.W_OK)}")

    logger = get_root_logger(logger_name="traiNNer", log_file=log_file)

    # Verify log file was actually created
    if os.path.exists(log_file):
        print(f"✅ Log file successfully created: {log_file}")
    else:
        print(f"❌ Log file was not created: {log_file}")
        # Try to create a fallback log file for debugging
        fallback_log = osp.join(
            opt.path.log, f"fallback_train_{opt.name}_{get_time_str()}.log"
        )
        try:
            with open(fallback_log, "w") as f:
                f.write(
                    f"Training log file creation failed. Original path: {log_file}\n"
                )
            print(f"📝 Fallback log file created: {fallback_log}")
        except Exception as e:
            print(f"💥 Failed to create fallback log file: {e}")
    logger.info(get_env_info())
    logger.debug(opt.contents)
    opt.contents = None
    diff, template_name = diff_user_vs_template(args.opt)
    if diff and template_name:
        logger.info("Diff with default config (%s):\n%s", template_name, diff)

    if opt.deterministic:
        logger.info(
            "Training in deterministic mode with manual_seed=%d. Deterministic mode has reduced training speed.",
            opt.manual_seed,
        )
    else:
        logger.info(
            "Training with manual_seed=%d.",
            opt.manual_seed,
        )

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    val_enabled = False
    if opt.val:
        val_enabled = opt.val.val_enabled

    train_loader, train_sampler, val_loaders, total_epochs, total_iters = (
        create_train_val_dataloader(opt, args, val_enabled, logger)
    )

    if train_loader is None or train_sampler is None:
        raise ValueError(
            "Failed to initialize training dataloader. Make sure train dataset is defined in datasets."
        )

    if opt.fast_matmul:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True

    # create model
    model = build_model(opt)

    # Initialize dynamic wrappers for VRAM management
    dynamic_dataloader_wrapper = None
    dynamic_dataset_wrapper = None

    if (
        hasattr(model, "training_automation_manager")
        and model.training_automation_manager
    ):
        # Create dynamic dataloader wrapper if VRAM management is enabled
        automation = model.training_automation_manager.automations.get(
            "DynamicBatchAndPatchSizeOptimizer"
        )
        if automation and automation.enabled:
            from traiNNer.data.dynamic_dataloader_wrapper import (
                create_dynamic_dataloader,
                patch_dataset_for_dynamic_updates,
            )

            current_batch_size = opt.datasets["train"].batch_size_per_gpu or 8
            current_lq_size = opt.datasets["train"].lq_size or 128

            # Create dynamic dataset wrapper
            dynamic_dataset_wrapper = patch_dataset_for_dynamic_updates(
                train_loader.dataset
            )

            # CRITICAL FIX: Recreate the DataLoader with the dynamic dataset
            # We cannot modify .dataset of an existing DataLoader, so we must recreate it
            from torch.utils.data import DataLoader

            # Extract parameters from existing loader
            loader_kwargs = {
                "batch_size": train_loader.batch_size,
                "num_workers": train_loader.num_workers,
                "sampler": train_loader.sampler,
                "batch_sampler": train_loader.batch_sampler,
                # 'collate_fn': train_loader.collate_fn, # MOVED: Passed to wrapper instead
                "pin_memory": train_loader.pin_memory,
                "drop_last": train_loader.drop_last,
                "timeout": train_loader.timeout,
                "worker_init_fn": train_loader.worker_init_fn,
                "multiprocessing_context": train_loader.multiprocessing_context,
                "generator": train_loader.generator,
                "prefetch_factor": train_loader.prefetch_factor,
                "persistent_workers": train_loader.persistent_workers,
            }

            # Save original collate_fn to pass to wrapper
            original_collate_fn = train_loader.collate_fn

            # Disable collation in the internal loader to prevent worker crashes on mixed sizes
            # This returns a list of raw samples to the wrapper
            loader_kwargs["collate_fn"] = lambda x: x

            # Remove arguments that are mutually exclusive or problematic
            if loader_kwargs["batch_sampler"] is not None:
                del loader_kwargs["batch_size"]
                del loader_kwargs["sampler"]
                del loader_kwargs["drop_last"]

            # Create new loader with dynamic dataset
            new_train_loader = DataLoader(dynamic_dataset_wrapper, **loader_kwargs)

            # Create dynamic dataloader wrapper around the NEW loader
            dynamic_dataloader_wrapper = create_dynamic_dataloader(
                new_train_loader,
                current_batch_size,
                update_callback=lambda bs: logger.debug(f"Batch size updated to: {bs}"),
                collate_fn=original_collate_fn,
            )

            # Set dynamic wrappers in the model for VRAM management
            model.set_dynamic_wrappers(
                dynamic_dataloader_wrapper, dynamic_dataset_wrapper
            )

            # CRITICAL FIX: Replace the train_loader with the dynamic wrapper
            # This ensures that the prefetcher uses the dynamic dataloader
            train_loader = dynamic_dataloader_wrapper

            # Initialize automation parameters with explicit validation
            model.set_automation_parameters(current_batch_size, current_lq_size)

            # Verify automation parameters are correctly set
            automation = model.training_automation_manager.automations.get(
                "DynamicBatchAndPatchSizeOptimizer"
            )
            if automation and automation.enabled:
                logger.info(
                    f"✅ VRAM Automation verification - "
                    f"Config batch: {current_batch_size}, lq: {current_lq_size}, "
                    f"Automation batch: {automation.current_batch_size}, "
                    f"Automation lq: {automation.current_lq_size}"
                )

                # Initialize VRAM monitoring immediately if possible
                if torch.cuda.is_available():
                    initial_vram = (
                        torch.cuda.memory_allocated()
                        / torch.cuda.get_device_properties(0).total_memory
                    )
                    logger.info(
                        f"🔥 Initial VRAM usage: {initial_vram:.3f} ({initial_vram * 100:.1f}%)"
                    )

            logger.info(
                f"Dynamic VRAM management initialized - "
                f"Batch: {current_batch_size}, LQ: {current_lq_size}, "
                f"Dynamic Wrappers: Enabled"
            )
        else:
            # Fallback to traditional automation without dynamic wrappers
            current_batch_size = opt.datasets["train"].batch_size_per_gpu or 8
            current_lq_size = opt.datasets["train"].lq_size or 128
            model.set_automation_parameters(current_batch_size, current_lq_size)
            logger.info(
                f"Automation parameters initialized - Batch: {current_batch_size}, LQ: {current_lq_size}"
            )

    if model.with_metrics:
        if not any(
            isinstance(
                val_loader.dataset,
                (PairedImageDataset | PairedVideoDataset),
            )
            for val_loader in val_loaders
        ):
            raise ValueError(
                "Validation metrics are enabled, at least one validation dataset must have type PairedImageDataset or PairedVideoDataset."
            )

    if torch.is_anomaly_enabled():
        logger.warning(
            "!!!! Anomaly checking is enabled. This slows down training and should be used for testing only !!!!"
        )

    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(
            "Resuming training from epoch: %d, iter: %d.",
            resume_state["epoch"],
            resume_state["iter"],
        )
        start_epoch = resume_state["epoch"]
        current_iter = resume_state["iter"]

        del resume_state
    else:
        start_epoch = 0
        current_iter = 0

    current_accum_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt.datasets["train"].prefetch_mode
    if prefetch_mode is None or prefetch_mode == "cpu":
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info("Use %s prefetch dataloader", prefetch_mode)
        if not opt.datasets["train"].pin_memory:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    else:
        raise ValueError(
            f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'."
        )

    # training
    gc.collect()
    torch.cuda.empty_cache()

    # Early VRAM monitoring during warmup to ensure automation starts working
    if (
        hasattr(model, "training_automation_manager")
        and model.training_automation_manager
    ):
        automation = model.training_automation_manager.automations.get(
            "DynamicBatchAndPatchSizeOptimizer"
        )
        if automation and automation.enabled:
            # Trigger initial VRAM monitoring to establish baseline
            initial_adjustments = model.update_automation_vram_monitoring()
            if initial_adjustments:
                batch_adj, lq_adj = initial_adjustments
                # Handle None values during initialization phase (before training starts)
                if batch_adj is not None and lq_adj is not None:
                    logger.info(
                        f"🚀 Early VRAM monitoring - Initial adjustments suggested: "
                        f"Batch: {batch_adj:+d}, LQ: {lq_adj:+d}"
                    )
                else:
                    logger.info(
                        f"🚀 Early VRAM monitoring - Initialization phase (iterations 0-99): "
                        f"Monitoring VRAM without adjustments, batch: {automation.current_batch_size}, LQ: {automation.current_lq_size}"
                    )

    logger.info("Start training from epoch: %d, iter: %d.", start_epoch, current_iter)
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    interrupt_received = False

    def handle_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupt_received
        if not interrupt_received:
            logger.info("User interrupted. Preparing to save state...")
            interrupt_received = True

    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    epoch = start_epoch
    apply_gradient = False
    crashed = False
    train_data = None
    assert model.opt.path.models is not None

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            train_sampler.set_epoch(epoch)
            prefetcher.reset()
            train_data = prefetcher.next()

            while train_data is not None:
                data_timer.record()

                current_accum_iter += 1

                if current_accum_iter >= model.accum_iters:
                    current_accum_iter = 0
                    current_iter += 1
                    apply_gradient = True

                    # Update automations with new iteration
                    model.update_automation_iteration(current_iter)

                    # Check for early stopping from automation
                    if model.training_automation_manager:
                        # Early stopping will be checked during validation, but we can prepare
                        pass
                else:
                    apply_gradient = False

                if current_iter > total_iters:
                    break
                # training
                model.feed_data(train_data)
                try:
                    model.optimize_parameters(
                        current_iter, current_accum_iter, apply_gradient
                    )

                    # Update training automations with training progress
                    if apply_gradient:
                        # Update VRAM monitoring for batch size and lq_size optimization
                        adjustments = model.update_automation_vram_monitoring()

                        # Enhanced VRAM automation debugging - more frequent and detailed
                        if (
                            current_iter % 500 == 0
                        ):  # Log every 500 iterations for better monitoring
                            if (
                                hasattr(model, "training_automation_manager")
                                and model.training_automation_manager
                            ):
                                automation = (
                                    model.training_automation_manager.automations.get(
                                        "DynamicBatchAndPatchSizeOptimizer"
                                    )
                                )
                                if automation and automation.enabled:
                                    # Get current VRAM usage for debugging
                                    if torch.cuda.is_available():
                                        current_vram = (
                                            torch.cuda.memory_allocated()
                                            / torch.cuda.get_device_properties(
                                                0
                                            ).total_memory
                                        )
                                        logger.info(
                                            f"🔍 VRAM DEBUG (iter {current_iter}): "
                                            f"VRAM: {current_vram:.3f} ({current_vram * 100:.1f}%), "
                                            f"Target: {automation.target_vram_usage:.3f} ({automation.target_vram_usage * 100:.1f}%), "
                                            f"Current batch: {automation.current_batch_size}, "
                                            f"Current lq: {automation.current_lq_size}, "
                                            f"Min batch: {automation.min_batch_size}, "
                                            f"Max batch: {automation.max_batch_size}, "
                                            f"Min lq: {automation.min_lq_size}, "
                                            f"Max lq: {automation.max_lq_size}"
                                        )
                                    else:
                                        logger.info(
                                            f"🔍 VRAM DEBUG (iter {current_iter}): "
                                            f"CUDA not available, batch: {automation.current_batch_size}, lq: {automation.current_lq_size}"
                                        )
                                else:
                                    logger.info(
                                        f"🔍 VRAM DEBUG (iter {current_iter}): DynamicBatchAndPatchSizeOptimizer not found or disabled"
                                    )
                            else:
                                logger.info(
                                    f"🔍 VRAM DEBUG (iter {current_iter}): No training automation manager found"
                                )

                        if adjustments is not None:
                            batch_adjustment, lq_adjustment = adjustments
                            if (
                                batch_adjustment is not None
                                and lq_adjustment is not None
                            ):
                                if batch_adjustment != 0 or lq_adjustment != 0:
                                    logger.info(
                                        f"Automation suggests adjustments - Batch size: {batch_adjustment:+d}, LQ size: {lq_adjustment:+d}"
                                    )
                                try:
                                    # Update batch size tracking and apply to dynamic wrapper
                                    if batch_adjustment != 0:
                                        current_batch = (
                                            automation.current_batch_size
                                            or opt.datasets["train"].batch_size_per_gpu
                                        )
                                        new_batch = max(
                                            1, current_batch + batch_adjustment
                                        )

                                        # Update automation tracking
                                        model.set_automation_batch_size(new_batch)

                                        # Apply to dynamic wrapper if available (this is the key fix!)
                                        if dynamic_dataloader_wrapper:
                                            dynamic_dataloader_wrapper.set_batch_size(
                                                new_batch
                                            )
                                        elif (
                                            hasattr(automation, "dynamic_dataloader")
                                            and automation.dynamic_dataloader
                                        ):
                                            automation.dynamic_dataloader.set_batch_size(
                                                new_batch
                                            )

                                        logger.info(
                                            f"Batch size adjusted: {current_batch} → {new_batch}"
                                        )

                                    # Update lq_size tracking and apply to dynamic wrapper
                                    if lq_adjustment != 0:
                                        current_lq = (
                                            automation.current_lq_size
                                            or opt.datasets["train"].lq_size
                                        )
                                        new_lq = max(32, current_lq + lq_adjustment)

                                        # Update automation tracking
                                        model.set_automation_lq_size(new_lq)

                                        # Safely get scale factor
                                        scale_factor = opt.scale or 1
                                        # Apply to dynamic wrapper if available (this is the key fix!)
                                        if dynamic_dataset_wrapper and hasattr(
                                            dynamic_dataset_wrapper,
                                            "set_dynamic_gt_size",
                                        ):
                                            dynamic_dataset_wrapper.set_dynamic_gt_size(
                                                new_lq * scale_factor
                                            )
                                        elif (
                                            hasattr(automation, "dynamic_dataset")
                                            and automation.dynamic_dataset
                                        ):
                                            if hasattr(
                                                automation.dynamic_dataset,
                                                "set_dynamic_gt_size",
                                            ):
                                                automation.dynamic_dataset.set_dynamic_gt_size(
                                                    new_lq * scale_factor
                                                )

                                        logger.info(
                                            f"LQ size adjusted: {current_lq} → {new_lq} (GT: {new_lq * scale_factor})"
                                        )

                                except Exception as e:
                                    logger.warning(
                                        f"Failed to apply automation adjustments: {e}"
                                    )

                except RuntimeError as e:
                    # Check to see if its actually the CUDA out of memory error
                    if "allocate" in str(e) or "CUDA" in str(e):
                        # Handle OOM recovery with automations
                        if (
                            hasattr(model, "training_automation_manager")
                            and model.training_automation_manager
                        ):
                            logger.info(
                                "OOM detected, attempting automation recovery..."
                            )
                            # Get current batch size for recovery
                            current_batch_size = opt.datasets[
                                "train"
                            ].batch_size_per_gpu
                            # Suggest reduced batch size (automation will handle actual adjustment)
                            suggested_batch_size = max(1, current_batch_size // 2)
                            suggested_lq_size = opt.datasets["train"].lq_size // 2
                            model.handle_automation_oom_recovery(
                                suggested_batch_size, suggested_lq_size
                            )

                            # Apply OOM recovery to dynamic wrappers (this is the key fix!)
                            if dynamic_dataloader_wrapper:
                                dynamic_dataloader_wrapper.set_batch_size(
                                    suggested_batch_size
                                )
                            elif (
                                hasattr(
                                    model.training_automation_manager.automations.get(
                                        "DynamicBatchAndPatchSizeOptimizer"
                                    ),
                                    "dynamic_dataloader",
                                )
                                and model.training_automation_manager.automations.get(
                                    "DynamicBatchAndPatchSizeOptimizer"
                                ).dynamic_dataloader
                            ):
                                model.training_automation_manager.automations.get(
                                    "DynamicBatchAndPatchSizeOptimizer"
                                ).dynamic_dataloader.set_batch_size(
                                    suggested_batch_size
                                )

                            # Safely get scale factor for OOM recovery
                            scale_factor = opt.scale or 1
                            if dynamic_dataset_wrapper and hasattr(
                                dynamic_dataset_wrapper, "set_dynamic_gt_size"
                            ):
                                dynamic_dataset_wrapper.set_dynamic_gt_size(
                                    suggested_lq_size * scale_factor
                                )
                            elif (
                                hasattr(
                                    model.training_automation_manager.automations.get(
                                        "DynamicBatchAndPatchSizeOptimizer"
                                    ),
                                    "dynamic_dataset",
                                )
                                and model.training_automation_manager.automations.get(
                                    "DynamicBatchAndPatchSizeOptimizer"
                                ).dynamic_dataset
                            ):
                                if hasattr(
                                    model.training_automation_manager.automations.get(
                                        "DynamicBatchAndPatchSizeOptimizer"
                                    ).dynamic_dataset,
                                    "set_dynamic_gt_size",
                                ):
                                    model.training_automation_manager.automations.get(
                                        "DynamicBatchAndPatchSizeOptimizer"
                                    ).dynamic_dataset.set_dynamic_gt_size(
                                        suggested_lq_size * scale_factor
                                    )

                        # Collect garbage (clear VRAM)
                        raise RuntimeError(
                            "Ran out of VRAM during training. The automation system attempted recovery, but please reduce lq_size or batch_size_per_gpu in your config and try again."
                        ) from None
                    else:
                        # Re-raise the exception if not an OOM error
                        raise
                # update learning rate
                if apply_gradient:
                    model.update_learning_rate(
                        current_iter, warmup_iter=opt.train.warmup_iter
                    )
                iter_timer.record()
                if current_iter == msg_logger.start_iter + 1:
                    # reset start time in msg_logger for more accurate eta_time
                    msg_logger.reset_start_time()
                # log
                if current_iter % opt.logger.print_freq == 0 and apply_gradient:
                    log_vars = {"epoch": epoch, "iter": current_iter}
                    log_vars.update({"lrs": model.get_current_learning_rate()})
                    log_vars.update(
                        {
                            "time": iter_timer.get_avg_time(),
                            "data_time": data_timer.get_avg_time(),
                        }
                    )
                    log_vars.update(model.get_current_log())

                    # Add enhanced training statistics for comprehensive monitoring
                    try:
                        # Get gradients for enhanced logging (if available)
                        gradients = []
                        if hasattr(model, "net_g"):
                            gradients = [
                                p.grad
                                for p in model.net_g.parameters()
                                if p.grad is not None
                            ]

                        # Collect enhanced logging statistics
                        if hasattr(model, "_collect_enhanced_logging_stats"):
                            enhanced_stats = model._collect_enhanced_logging_stats(
                                model.log_dict, current_iter, gradients
                            )
                            # Add enhanced stats to log_vars
                            for key, value in enhanced_stats.items():
                                log_vars[key] = value
                    except Exception as e:
                        # Fallback if enhanced logging fails
                        logger.debug(f"Enhanced logging failed: {e}")

                    model.reset_current_log()
                    msg_logger(log_vars)

                # save models and training states
                if (
                    current_iter % opt.logger.save_checkpoint_freq == 0
                    and apply_gradient
                ):
                    logger.info(
                        "Saving models and training states to %s. Free space: %s",
                        clickable_file_path(
                            model.opt.path.models, "experiments folder"
                        ),
                        free_space_gb_str(),
                    )
                    model.save(
                        epoch,
                        current_iter,
                    )

                # validation
                if opt.val is not None:
                    assert opt.val.val_freq is not None, (
                        "val_freq must be defined under the val section"
                    )
                    if current_iter % opt.val.val_freq == 0 and apply_gradient:
                        multi_val_datasets = len(val_loaders) > 1
                        for val_loader in val_loaders:
                            model.validation(
                                val_loader,
                                current_iter,
                                tb_logger,
                                opt.val.save_img,
                                multi_val_datasets,
                            )

                        # Check for early stopping from automations
                        if (
                            hasattr(model, "training_automation_manager")
                            and model.training_automation_manager
                        ):
                            # Get validation metrics from model for early stopping
                            validation_metrics = {}
                            if model.with_metrics and model.metric_results:
                                # Convert metric results to dict format expected by automations
                                for (
                                    metric_name,
                                    metric_value,
                                ) in model.metric_results.items():
                                    validation_metrics[metric_name] = float(
                                        metric_value
                                    )
                                # Also add common metric names
                                if "psnr" in model.metric_results:
                                    validation_metrics["val/psnr"] = (
                                        model.metric_results["psnr"]
                                    )
                                if "ssim" in model.metric_results:
                                    validation_metrics["val/ssim"] = (
                                        model.metric_results["ssim"]
                                    )

                            # Update validation tracking and check for early stopping
                            should_stop, stop_reason = (
                                model.update_automation_validation_tracking(
                                    validation_metrics, current_iter
                                )
                            )

                            if should_stop:
                                logger.info(
                                    f"Training stopped by automation: {stop_reason}"
                                )
                                interrupt_received = True
                                break

                data_timer.start()
                iter_timer.start()
                train_data = prefetcher.next()

                if interrupt_received:
                    break

            # end of iter
            if interrupt_received:
                break
        # end of epoch
    except Exception as e:
        logger.exception(e)
        crashed = True
        interrupt_received = True

    # epoch was completed, increment it to set the correct epoch count when interrupted
    if train_data is None:
        epoch += 1

    if interrupt_received:
        # discard partially accumulated iters
        if not apply_gradient or crashed:
            current_iter -= 1

        if current_iter > 0:
            logger.info(
                "Saving models and training states to %s for epoch: %d, iter: %d.",
                clickable_file_path(model.opt.path.models, "experiments folder"),
                epoch,
                current_iter,
            )
            model.save(epoch, current_iter)
        if model.opt.dist:
            destroy_process_group()
        sys.exit(0)

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info("End of training. Time consumed: %s", consumed_time)
    logger.info("Save the latest model.")
    model.save(
        epoch=-1,
        current_iter=-1,
    )  # -1 stands for the latest
    if opt.val is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt.val.save_img)
    if tb_logger:
        tb_logger.close()
    if model.opt.dist:
        destroy_process_group()


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
