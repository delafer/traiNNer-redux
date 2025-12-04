#!/usr/bin/env python3
"""
Hardware Detection Module for traiNNer-redux

Automatically detects system hardware and provides optimal training parameters
for the training automations. This enables truly zero-config training setups.

Features:
- GPU detection and VRAM analysis
- CPU core counting and memory detection
- Automatic parameter optimization based on hardware capabilities
- Architecture-specific hardware recommendations
- Memory optimization suggestions

Author: Philip Hofmann
"""

import os
from typing import Any, Dict, Optional, Tuple

import psutil
import torch


class HardwareDetector:
    """
    Advanced hardware detection and parameter optimization system.

    Automatically determines optimal training parameters based on detected
    hardware capabilities, enabling zero-configuration training setups.
    """

    def __init__(self) -> None:
        self.gpu_info = self._detect_gpu()
        self.cpu_info = self._detect_cpu()
        self.memory_info = self._detect_memory()
        self.system_info = self._detect_system()

    def _detect_gpu(self) -> dict[str, Any]:
        """Detect GPU hardware information."""
        gpu_info = {
            "available": torch.cuda.is_available(),
            "count": 0,
            "names": [],
            "total_vram_gb": 0,
            "vram_per_gpu_gb": [],
            "compute_capability": [],
            "architecture": "unknown",
            "driver_version": "unknown",
            "cuda_version": "unknown",
        }

        if not torch.cuda.is_available():
            return gpu_info

        gpu_info["count"] = torch.cuda.device_count()

        for i in range(gpu_info["count"]):
            try:
                props = torch.cuda.get_device_properties(i)
                name = props.name
                vram_gb = props.total_memory / (1024**3)

                gpu_info["names"].append(name)
                gpu_info["vram_per_gpu_gb"].append(vram_gb)
                gpu_info["compute_capability"].append(f"{props.major}.{props.minor}")

                # Architecture detection based on compute capability
                if props.major >= 8:
                    architecture = "Ampere" if props.major == 8 else "Ada Lovelace"
                elif props.major >= 7:
                    architecture = "Turing" if props.minor == 5 else "Volta"
                elif props.major >= 6:
                    architecture = "Pascal"
                else:
                    architecture = "older"

                gpu_info["architecture"] = architecture

            except Exception as e:
                print(f"Warning: Could not detect GPU {i}: {e}")

        gpu_info["total_vram_gb"] = sum(gpu_info["vram_per_gpu_gb"])

        try:
            gpu_info["driver_version"] = torch.version.cuda
        except:
            pass

        try:
            gpu_info["cuda_version"] = torch.version.cuda
        except:
            pass

        return gpu_info

    def _detect_cpu(self) -> dict[str, Any]:
        """Detect CPU information."""
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            "architecture": "x86_64",  # Default assumption
        }

        # Try to get CPU model name
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_info["model"] = line.split(":")[1].strip()
                        break
        except:
            cpu_info["model"] = "Unknown CPU"

        # Architecture detection
        try:
            import platform

            cpu_info["architecture"] = platform.machine()
        except:
            pass

        return cpu_info

    def _detect_memory(self) -> dict[str, Any]:
        """Detect system memory information."""
        memory_info = psutil.virtual_memory()

        return {
            "total_gb": memory_info.total / (1024**3),
            "available_gb": memory_info.available / (1024**3),
            "used_gb": memory_info.used / (1024**3),
            "percentage_used": memory_info.percent,
            "swap_total_gb": psutil.swap_memory().total / (1024**3),
        }

    def _detect_system(self) -> dict[str, Any]:
        """Detect general system information."""
        import platform

        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "gpu_backend": "cuda" if torch.cuda.is_available() else "cpu",
        }

    def get_hardware_tier(self) -> str:
        """
        Classify the hardware into performance tiers for optimization.

        Returns:
            Tier classification: "budget", "mid_range", "high_end", "workstation"
        """
        if not self.gpu_info["available"]:
            return "cpu_limited"

        total_vram = self.gpu_info["total_vram_gb"]

        # Classification based on VRAM and GPU count
        if total_vram <= 4:
            return "budget"
        elif total_vram <= 8:
            return "mid_range" if self.gpu_info["count"] == 1 else "high_end"
        elif total_vram <= 24:
            return "high_end"
        else:
            return "workstation"

    def get_optimal_batch_size(
        self, architecture: str, dataset_complexity: str = "medium"
    ) -> int:
        """
        Calculate optimal batch size based on hardware and architecture.

        Args:
            architecture: Neural network architecture type
            dataset_complexity: "simple", "medium", "complex"

        Returns:
            Optimal batch size for training
        """
        if not self.gpu_info["available"]:
            return 1  # CPU training

        available_vram = self.gpu_info["vram_per_gpu_gb"][
            0
        ]  # Assume single GPU for batch sizing
        vram_per_gpu = available_vram

        # Architecture-specific memory multipliers
        arch_multipliers = {
            "paragonsr2_nano": 0.8,
            "paragonsr2_micro": 1.0,
            "paragonsr2_tiny": 1.2,
            "paragonsr2_xs": 1.5,
            "paragonsr2_s": 2.0,
            "paragonsr2_m": 3.0,
            "paragonsr2_l": 4.0,
            "paragonsr2_xl": 6.0,
            "esrgan": 2.5,
            "rcan": 3.0,
            "swinir": 4.0,
        }

        # Dataset complexity multipliers
        complexity_multipliers = {"simple": 1.5, "medium": 1.0, "complex": 0.6}

        base_multiplier = arch_multipliers.get(architecture.lower(), 1.0)
        complexity_multiplier = complexity_multipliers.get(dataset_complexity, 1.0)

        # Calculate optimal batch size
        memory_efficiency = 0.7  # Use 70% of VRAM for training
        effective_vram = vram_per_gpu * memory_efficiency

        # Base batch sizes by architecture (nano = 32, scaling up)
        base_batch_sizes = {
            "paragonsr2_nano": 32,
            "paragonsr2_micro": 24,
            "paragonsr2_tiny": 20,
            "paragonsr2_xs": 16,
            "paragonsr2_s": 12,
            "paragonsr2_m": 8,
            "paragonsr2_l": 6,
            "paragonsr2_xl": 4,
            "esrgan": 8,
            "rcan": 6,
            "swinir": 4,
        }

        base_batch = base_batch_sizes.get(architecture.lower(), 16)

        # Apply VRAM scaling
        vram_factor = effective_vram / 8.0  # Normalize to 8GB baseline
        optimal_batch = int(
            base_batch * vram_factor * base_multiplier * complexity_multiplier
        )

        # Apply hardware tier limits
        tier = self.get_hardware_tier()
        tier_limits = {"budget": 8, "mid_range": 16, "high_end": 32, "workstation": 64}

        max_batch = tier_limits.get(tier, 16)
        optimal_batch = min(optimal_batch, max_batch)

        # Ensure minimum batch size
        return max(1, optimal_batch)

    def get_optimal_automations_config(
        self, architecture: str, total_iterations: int = 40000
    ) -> dict[str, Any]:
        """
        Generate optimal automation configuration based on detected hardware.

        Args:
            architecture: Neural network architecture
            total_iterations: Planned training iterations

        Returns:
            Complete automation configuration
        """
        tier = self.get_hardware_tier()
        gpu_count = self.gpu_info["count"]
        vram_per_gpu = (
            self.gpu_info["vram_per_gpu_gb"][0] if self.gpu_info["available"] else 0
        )

        # Hardware-tier based optimization
        tier_configs = {
            "budget": {
                "target_vram_usage": 0.75,
                "safety_margin": 0.15,
                "plateau_patience": 800,
                "initial_threshold": 0.8,
                "patience": 1500,
            },
            "mid_range": {
                "target_vram_usage": 0.85,
                "safety_margin": 0.05,
                "plateau_patience": 1000,
                "initial_threshold": 1.0,
                "patience": 2000,
            },
            "high_end": {
                "target_vram_usage": 0.90,
                "safety_margin": 0.03,
                "plateau_patience": 1200,
                "initial_threshold": 1.2,
                "patience": 2500,
            },
            "workstation": {
                "target_vram_usage": 0.93,
                "safety_margin": 0.02,
                "plateau_patience": 1500,
                "initial_threshold": 1.5,
                "patience": 3000,
            },
        }

        base_config = tier_configs.get(tier, tier_configs["mid_range"])

        # Adjust for training duration
        if total_iterations < 10000:
            # Short training - more aggressive adaptation
            base_config["plateau_patience"] = max(
                500, int(base_config["plateau_patience"] * 0.7)
            )
            base_config["patience"] = max(1000, int(base_config["patience"] * 0.7))
        elif total_iterations > 50000:
            # Long training - more patient, conservative approach
            base_config["plateau_patience"] = int(base_config["plateau_patience"] * 1.3)
            base_config["patience"] = int(base_config["patience"] * 1.3)

        # VRAM-specific adjustments
        if vram_per_gpu >= 24:
            # High VRAM cards can be more aggressive
            base_config["target_vram_usage"] = min(
                0.95, base_config["target_vram_usage"] + 0.05
            )
        elif vram_per_gpu <= 6:
            # Lower VRAM cards need more conservative approach
            base_config["target_vram_usage"] = max(
                0.70, base_config["target_vram_usage"] - 0.10
            )

        # GPU count adjustments
        if gpu_count > 1:
            base_config["adjustment_frequency"] = (
                200  # Less frequent adjustments for multi-GPU
            )
        else:
            base_config["adjustment_frequency"] = 100

        # Architecture-specific tuning
        if "paragonsr2" in architecture.lower():
            if "nano" in architecture.lower():
                base_config["initial_threshold"] *= (
                    0.9  # Slightly more conservative for nano
                )
            elif "xl" in architecture.lower():
                base_config["initial_threshold"] *= 1.2  # Can handle larger gradients

        return {
            "enabled": True,
            "IntelligentLearningRateScheduler": {
                "enabled": True,
                "monitor_loss": True,
                "monitor_validation": True,
                "adaptation_threshold": 0.02,
                "plateau_patience": base_config["plateau_patience"],
                "improvement_threshold": 0.001,
                "min_lr_factor": 0.1,
                "max_lr_factor": 2.0,
                "fallback": {
                    "scheduler_type": "cosine",
                    "scheduler_params": {"eta_min": 0.00001, "T_max": total_iterations},
                },
                "max_adjustments": 50,
            },
            "DynamicBatchAndPatchSizeOptimizer": {
                "enabled": True,
                "target_vram_usage": base_config["target_vram_usage"],
                "safety_margin": base_config["safety_margin"],
                "adjustment_frequency": base_config.get("adjustment_frequency", 100),
                "min_batch_size": 1,
                "max_batch_size": 64
                if tier == "workstation"
                else 32
                if tier == "high_end"
                else 16,
                "vram_history_size": 50,
                "fallback": {"batch_size": self.get_optimal_batch_size(architecture)},
                "max_adjustments": 20,
            },
            "AdaptiveGradientClipping": {
                "enabled": True,
                "initial_threshold": base_config["initial_threshold"],
                "min_threshold": 0.1,
                "max_threshold": 10.0,
                "adjustment_factor": 1.2,
                "monitoring_frequency": 10,
                "gradient_history_size": 100,
                "fallback": {"threshold": 1.0},
                "max_adjustments": 100,
            },
            "IntelligentEarlyStopping": {
                "enabled": True,
                "patience": base_config["patience"],
                "min_improvement": 0.001,
                "min_epochs": max(100, total_iterations // 100),
                "min_iterations": max(1000, total_iterations // 10),
                "monitor_metric": "val/psnr",
                "max_no_improvement": base_config["patience"],
                "improvement_threshold": 0.002,
                "warmup_iterations": max(500, total_iterations // 80),
                "fallback": {"early_stopping": False},
                "max_adjustments": 1,
            },
        }

    def get_optimization_recommendations(self, architecture: str) -> dict[str, Any]:
        """
        Get system-specific optimization recommendations.

        Args:
            architecture: Network architecture type

        Returns:
            Dictionary of optimization recommendations
        """
        recommendations = {
            "amp_recommendation": True if self.gpu_info["available"] else False,
            "channels_last": True if self.gpu_info["available"] else False,
            "fast_matmul": True if self.gpu_info["available"] else False,
            "num_workers": min(8, self.cpu_info["logical_cores"]),
            "pin_memory": True if self.gpu_info["available"] else False,
            "compile_model": torch.cuda.is_available() and hasattr(torch, "compile"),
        }

        # Tier-specific recommendations
        tier = self.get_hardware_tier()
        if tier == "budget":
            recommendations.update(
                {
                    "amp_recommendation": True,  # Definitely use AMP on budget cards
                    "grad_accumulation": True,  # Use gradient accumulation
                    "mixed_precision": "bf16"
                    if "bf16" in torch.get_default_dtype().__str__
                    else "fp16",
                }
            )
        elif tier in {"high_end", "workstation"}:
            recommendations.update(
                {
                    "compile_model": True,  # Benefit from model compilation
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": False,  # Don't need this on high-end cards
                }
            )

        # Architecture-specific recommendations
        if "paragonsr2" in architecture.lower():
            if self.gpu_info["total_vram_gb"] < 6:
                recommendations["grad_accumulation"] = True
                recommendations["accum_iter"] = 2

        return recommendations

    def generate_hardware_report(self) -> str:
        """Generate a comprehensive hardware detection report."""
        tier = self.get_hardware_tier()

        report = f"""
=== Hardware Detection Report ===

System Information:
- Platform: {self.system_info["platform"]}
- Python: {self.system_info["python_version"]}
- PyTorch: {self.system_info["pytorch_version"]}
- GPU Backend: {self.system_info["gpu_backend"]}

GPU Information:
- Available: {self.gpu_info["available"]}
- Count: {self.gpu_info["count"]}
- Total VRAM: {self.gpu_info["total_vram_gb"]:.1f}GB
- Architecture: {self.gpu_info["architecture"]}
- CUDA Version: {self.gpu_info["cuda_version"]}
"""

        if self.gpu_info["names"]:
            report += "\nDetected GPUs:\n"
            for i, (name, vram) in enumerate(
                zip(
                    self.gpu_info["names"],
                    self.gpu_info["vram_per_gpu_gb"],
                    strict=False,
                )
            ):
                report += f"  GPU {i}: {name} ({vram:.1f}GB)\n"

        report += f"""
CPU Information:
- Cores: {self.cpu_info["physical_cores"]} physical, {self.cpu_info["logical_cores"]} logical
- Max Frequency: {self.cpu_info["max_frequency_mhz"]:.0f}MHz
- Model: {self.cpu_info.get("model", "Unknown")}

Memory Information:
- Total RAM: {self.memory_info["total_gb"]:.1f}GB
- Available: {self.memory_info["available_gb"]:.1f}GB
- Usage: {self.memory_info["percentage_used"]:.1f}%

Hardware Tier: {tier.upper()}
Optimization Level: {"Conservative" if tier == "budget" else "Standard" if tier == "mid_range" else "Aggressive"}

Recommended Configuration:
- Optimal Batch Size: {self.get_optimal_batch_size("paragonsr2_nano")}
- Use AMP: {"Yes" if self.gpu_info["available"] else "No"}
- Compile Model: {"Yes" if self.get_optimization_recommendations("paragonsr2_nano").get("compile_model") else "No"}
"""

        return report

    def __str__(self) -> str:
        return f"HardwareDetector(tier={self.get_hardware_tier()}, gpus={self.gpu_info['count']}, vram={self.gpu_info['total_vram_gb']:.1f}GB)"


def detect_optimal_automation_config(
    architecture: str,
    total_iterations: int = 40000,
    dataset_complexity: str = "medium",
    custom_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """
    Detect hardware and return optimal automation configuration.

    This is the main entry point for zero-config training automation setup.

    Args:
        architecture: Neural network architecture name
        total_iterations: Planned training iterations
        dataset_complexity: Dataset complexity level
        custom_config: Optional user customizations

    Returns:
        Tuple of (automation_config, optimal_batch_size, optimization_recommendations)
    """
    detector = HardwareDetector()

    # Get optimal configurations
    automations_config = detector.get_optimal_automations_config(
        architecture, total_iterations
    )
    optimal_batch_size = detector.get_optimal_batch_size(
        architecture, dataset_complexity
    )
    recommendations = detector.get_optimization_recommendations(architecture)

    # Apply user customizations if provided
    if custom_config:
        # Merge user config with auto-detected config
        # User config takes precedence over auto-detected values
        def deep_merge(base: dict, override: dict) -> dict:
            for key, value in override.items():
                if (
                    isinstance(value, dict)
                    and key in base
                    and isinstance(base[key], dict)
                ):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        automations_config = deep_merge(automations_config, custom_config)

    return automations_config, optimal_batch_size, recommendations


# Convenience functions for easy integration
def get_zero_config_automations(
    architecture: str, total_iterations: int = 40000
) -> dict[str, Any]:
    """Get fully automatic automation configuration."""
    return detect_optimal_automation_config(architecture, total_iterations)[0]


def get_recommended_batch_size(
    architecture: str, dataset_complexity: str = "medium"
) -> int:
    """Get hardware-recommended batch size."""
    return detect_optimal_automation_config(
        architecture, dataset_complexity=dataset_complexity
    )[1]


def print_hardware_report() -> None:
    """Print detailed hardware detection report."""
    detector = HardwareDetector()
    print(detector.generate_hardware_report())


if __name__ == "__main__":
    # Demo: Print hardware report
    print_hardware_report()

    # Demo: Get zero-config automation for ParagonSR2 Nano
    config, batch_size, recommendations = detect_optimal_automation_config(
        "paragonsr2_nano"
    )
    print(f"\nRecommended batch size: {batch_size}")
    print(f"Automation config generated: {config['enabled']}")
