#!/usr/bin/env python3
"""
Loss Weight Schedule Visualization for ParagonSR2 Training Configuration

This script creates a comprehensive visualization showing how different loss weights
evolve over the 140k training iterations. It helps identify potential conflicts,
gaps, or scheduling issues in the training strategy.
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patches

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def calculate_scheduled_weight(
    start_iter: int,
    start_weight: float,
    target_iter: int,
    target_weight: float,
    total_iter: int,
) -> np.ndarray:
    """
    Calculate the scheduled weight over all iterations.

    Args:
        start_iter: When this loss starts (0 if always active)
        start_weight: Initial weight when it starts
        target_iter: When it reaches target weight
        target_weight: Final weight at target_iter
        total_iter: Total training iterations

    Returns:
        Array of weights for all iterations
    """
    weights = np.zeros(total_iter + 1)

    if start_iter > total_iter:
        return weights  # Never active

    # Ensure we don't go beyond total iterations
    effective_target_iter = min(target_iter, total_iter)
    effective_start_iter = min(start_iter, total_iter)

    # Set weights to zero before start_iter
    weights[:effective_start_iter] = 0

    # Linear interpolation from start to target
    if effective_target_iter > effective_start_iter:
        iteration_range = effective_target_iter - effective_start_iter
        weight_range = target_weight - start_weight

        for i in range(effective_start_iter, effective_target_iter + 1):
            progress = (i - effective_start_iter) / iteration_range
            weights[i] = start_weight + progress * weight_range

    # Set weights to target weight after target_iter
    weights[effective_target_iter + 1 :] = target_weight

    return weights


def get_loss_schedules(total_iter: int = 140000) -> dict[str, np.ndarray]:
    """Get all loss weight schedules from the configuration."""

    losses = [
        # Reconstruction losses
        {
            "name": "Charbonnier Loss",
            "start_iter": 0,
            "start_weight": 0.6,
            "target_iter": 25000,
            "target_weight": 0.4,
            "color": "#FF6B6B",
            "linestyle": "-",
            "category": "Reconstruction",
        },
        {
            "name": "ConvNeXt Perceptual",
            "start_iter": 0,
            "start_weight": 0.16,
            "target_iter": 25000,
            "target_weight": 0.32,
            "color": "#4ECDC4",
            "linestyle": "-",
            "category": "Reconstruction",
        },
        {
            "name": "DISTS Loss",
            "start_iter": 0,
            "start_weight": 0.10,
            "target_iter": 0,
            "target_weight": 0.10,
            "color": "#45B7D1",
            "linestyle": "-",
            "category": "Reconstruction",
        },
        # Frequency domain losses
        {
            "name": "FF Loss",
            "start_iter": 0,
            "start_weight": 0.40,
            "target_iter": 90000,
            "target_weight": 0.55,
            "color": "#96CEB4",
            "linestyle": "--",
            "category": "Frequency",
        },
        {
            "name": "Gradient Variance",
            "start_iter": 500,
            "start_weight": 0.05,
            "target_iter": 500,
            "target_weight": 0.05,
            "color": "#FECA57",
            "linestyle": "--",
            "category": "Frequency",
        },
        # Feature matching
        {
            "name": "Feature Matching",
            "start_iter": 0,
            "start_weight": 0.10,
            "target_iter": 0,
            "target_weight": 0.10,
            "color": "#FF9FF3",
            "linestyle": "-.",
            "category": "Stabilization",
        },
        # Local discriminator learning
        {
            "name": "LDL Loss",
            "start_iter": 0,
            "start_weight": 0.3,
            "target_iter": 0,
            "target_weight": 0.3,
            "disable_after": 90000,
            "color": "#54A0FF",
            "linestyle": ":",
            "category": "Stabilization",
        },
        # High-frequency preservation
        {
            "name": "HFEN Loss",
            "start_iter": 0,
            "start_weight": 0.015,
            "target_iter": 0,
            "target_weight": 0.015,
            "color": "#5F27CD",
            "linestyle": "-",
            "category": "Detail",
        },
        # Artifact reduction
        {
            "name": "Adaptive Block TV",
            "start_iter": 0,
            "start_weight": 0.004,
            "target_iter": 0,
            "target_weight": 0.004,
            "color": "#00D2D3",
            "linestyle": "-",
            "category": "Detail",
        },
        # Adversarial
        {
            "name": "R3GAN Loss",
            "start_iter": 30000,
            "start_weight": 0.00,
            "target_iter": 45000,
            "target_weight": 0.06,
            "color": "#FF3838",
            "linestyle": "-",
            "category": "Adversarial",
        },
        # Contrastive
        {
            "name": "Contrastive Loss",
            "start_iter": 60000,
            "start_weight": 0.00,
            "target_iter": 60000,
            "target_weight": 0.05,
            "color": "#2ED573",
            "linestyle": "-",
            "category": "Semantic",
        },
    ]

    schedules = {}

    for loss in losses:
        weight_array = calculate_scheduled_weight(
            loss["start_iter"],
            loss["start_weight"],
            loss["target_iter"],
            loss["target_weight"],
            total_iter,
        )

        # Handle LDL disable_after
        if "disable_after" in loss:
            weight_array[loss["disable_after"] :] = 0

        schedules[loss["name"]] = {
            "weights": weight_array,
            "color": loss["color"],
            "linestyle": loss["linestyle"],
            "category": loss["category"],
        }

    return schedules


def create_loss_visualization():
    """Create comprehensive loss weight visualization."""

    total_iter = 140000
    schedules = get_loss_schedules(total_iter)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # Main plot - all losses
    ax1 = plt.subplot(3, 2, (1, 4))

    # Color map for categories
    category_colors = {
        "Reconstruction": "#FF6B6B",
        "Frequency": "#96CEB4",
        "Stabilization": "#FF9FF3",
        "Detail": "#5F27CD",
        "Adversarial": "#FF3838",
        "Semantic": "#2ED573",
    }

    # Plot each loss
    for loss_name, loss_data in schedules.items():
        ax1.plot(
            range(total_iter + 1),
            loss_data["weights"],
            color=loss_data["color"],
            linestyle=loss_data["linestyle"],
            linewidth=2.5,
            label=loss_name,
            alpha=0.9,
        )

    # Add milestone markers
    milestones = [30000, 45000, 60000, 70000, 90000, 110000]
    milestone_labels = [
        "GAN Start",
        "GAN Full",
        "Contrastive Start",
        "LR Drop 1",
        "LDL End",
        "LR Drop 2",
    ]

    for milestone, label in zip(milestones, milestone_labels, strict=False):
        ax1.axvline(x=milestone, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax1.text(
            milestone,
            ax1.get_ylim()[1] * 0.95,
            label,
            rotation=90,
            verticalalignment="top",
            fontsize=10,
            alpha=0.7,
        )

    ax1.set_xlabel("Training Iterations", fontsize=12)
    ax1.set_ylabel("Loss Weight", fontsize=12)
    ax1.set_title(
        "Loss Weight Evolution Over Training Iterations\nParagonSR2 Static-S with Feature Matching",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax1.set_xlim(0, total_iter)

    # Category-wise breakdown
    categories = [
        "Reconstruction",
        "Frequency",
        "Stabilization",
        "Detail",
        "Adversarial",
        "Semantic",
    ]

    for i, category in enumerate(categories):
        ax = plt.subplot(3, 2, i + 5)
        category_losses = [
            (name, data)
            for name, data in schedules.items()
            if data["category"] == category
        ]

        if not category_losses:
            ax.text(
                0.5,
                0.5,
                f"No {category} losses",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(category, fontsize=12, fontweight="bold")
            continue

        for loss_name, loss_data in category_losses:
            ax.plot(
                range(total_iter + 1),
                loss_data["weights"],
                color=loss_data["color"],
                linestyle=loss_data["linestyle"],
                linewidth=2,
                label=loss_name,
            )

        ax.set_title(f"{category} Losses", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(0, total_iter)

        if i >= 4:  # Last two subplots
            ax.set_xlabel("Iterations", fontsize=10)
        if i % 2 == 0:  # Left column
            ax.set_ylabel("Weight", fontsize=10)

    plt.tight_layout()
    return fig


def analyze_schedule(schedules: dict) -> str:
    """Analyze the loss schedule for potential issues."""

    total_iter = 140000
    analysis = []
    analysis.append("=== LOSS SCHEDULE ANALYSIS ===\n")

    # Check for gaps in coverage
    analysis.append("1. TRAINING COVERAGE ANALYSIS:")

    # Early stage (0-30k)
    early_total = sum(data["weights"][0] for data in schedules.values())
    analysis.append(f"   Early Stage (0-30k): Total weight = {early_total:.3f}")

    # Mid stage (30k-90k)
    mid_total = sum(
        np.mean(data["weights"][30000:90000]) for data in schedules.values()
    )
    analysis.append(f"   Mid Stage (30k-90k): Average weight = {mid_total:.3f}")

    # Late stage (90k+)
    late_total = sum(np.mean(data["weights"][90000:]) for data in schedules.values())
    analysis.append(f"   Late Stage (90k+): Average weight = {late_total:.3f}")

    analysis.append("")

    # Check for potential conflicts
    analysis.append("2. POTENTIAL CONFLICTS:")

    # High weights at same time
    peak_weights = {}
    for name, data in schedules.items():
        peak_idx = np.argmax(data["weights"])
        peak_weights[name] = (peak_idx, data["weights"][peak_idx])

    # Find periods with many active high-weight losses
    high_weight_threshold = 0.1
    for iter_step in [25000, 50000, 75000, 100000]:
        active_losses = [
            (name, weight)
            for name, data in schedules.items()
            for iter_idx, weight in [(iter_step, data["weights"][iter_step])]
            if weight > high_weight_threshold
        ]
        if active_losses:
            analysis.append(
                f"   {iter_step:,} iterations: {len(active_losses)} high-weight losses active"
            )

    analysis.append("")

    # Check for scheduling gaps
    analysis.append("3. SCHEDULING INSIGHTS:")

    # Find when losses start
    for name, data in schedules.items():
        start_idx = np.where(data["weights"] > 0)[0]
        if len(start_idx) > 0:
            start_iter = start_idx[0]
            analysis.append(f"   {name}: Starts at {start_iter:,} iterations")

    analysis.append("")

    # Recommendations
    analysis.append("4. RECOMMENDATIONS:")

    # Check for too many losses starting at once
    iteration_counts = {}
    for name, data in schedules.items():
        start_idx = np.where(data["weights"] > 0)[0]
        if len(start_idx) > 0:
            start_iter = start_idx[0]
            iteration_counts[start_iter] = iteration_counts.get(start_iter, 0) + 1

    for iteration, count in iteration_counts.items():
        if count > 3:
            analysis.append(
                f"   Consider staggering losses starting at {iteration:,} (currently {count} losses)"
            )

    # Check for rapid weight changes
    for name, data in schedules.items():
        weights = data["weights"]
        rapid_changes = np.where(np.abs(np.diff(weights)) > 0.1)[0]
        if len(rapid_changes) > 0:
            analysis.append(f"   {name}: Rapid weight changes detected")

    return "\n".join(analysis)


def main() -> None:
    """Main function to create visualization and analysis."""

    print("Creating loss weight schedule visualization...")

    # Create visualization
    fig = create_loss_visualization()

    # Save the plot
    output_file = "loss_schedule_visualization.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Visualization saved as: {output_file}")

    # Create analysis
    schedules = get_loss_schedules()
    analysis_text = analyze_schedule(schedules)

    # Save analysis
    analysis_file = "loss_schedule_analysis.txt"
    with open(analysis_file, "w") as f:
        f.write(analysis_text)

    print(f"Analysis saved as: {analysis_file}")
    print("\n" + "=" * 60)
    print(analysis_text)

    # Show the plot if in interactive environment
    plt.show()


if __name__ == "__main__":
    main()
