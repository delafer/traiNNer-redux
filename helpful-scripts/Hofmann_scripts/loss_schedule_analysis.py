#!/usr/bin/env python3
"""
Loss Weight Schedule Analysis for ParagonSR2 Training Configuration

This script analyzes the loss weight schedule and provides insights
without requiring matplotlib display capabilities.
"""

from typing import Dict, List, Tuple

import numpy as np


def calculate_scheduled_weight(
    start_iter: int,
    start_weight: float,
    target_iter: int,
    target_weight: float,
    total_iter: int,
) -> np.ndarray:
    """Calculate the scheduled weight over all iterations."""
    weights = np.zeros(total_iter + 1)

    if start_iter > total_iter:
        return weights

    effective_target_iter = min(target_iter, total_iter)
    effective_start_iter = min(start_iter, total_iter)
    weights[:effective_start_iter] = 0

    if effective_target_iter > effective_start_iter:
        iteration_range = effective_target_iter - effective_start_iter
        weight_range = target_weight - start_weight

        for i in range(effective_start_iter, effective_target_iter + 1):
            progress = (i - effective_start_iter) / iteration_range
            weights[i] = start_weight + progress * weight_range

    weights[effective_target_iter + 1 :] = target_weight
    return weights


def get_loss_schedules(total_iter: int = 140000) -> dict:
    """Get all loss weight schedules from the configuration."""

    losses = [
        {
            "name": "Charbonnier Loss",
            "start_iter": 0,
            "start_weight": 0.6,
            "target_iter": 25000,
            "target_weight": 0.4,
            "category": "Reconstruction",
        },
        {
            "name": "ConvNeXt Perceptual",
            "start_iter": 0,
            "start_weight": 0.16,
            "target_iter": 25000,
            "target_weight": 0.32,
            "category": "Reconstruction",
        },
        {
            "name": "DISTS Loss",
            "start_iter": 0,
            "start_weight": 0.10,
            "target_iter": 0,
            "target_weight": 0.10,
            "category": "Reconstruction",
        },
        {
            "name": "FF Loss",
            "start_iter": 0,
            "start_weight": 0.40,
            "target_iter": 90000,
            "target_weight": 0.55,
            "category": "Frequency",
        },
        {
            "name": "Gradient Variance",
            "start_iter": 500,
            "start_weight": 0.05,
            "target_iter": 500,
            "target_weight": 0.05,
            "category": "Frequency",
        },
        {
            "name": "Feature Matching",
            "start_iter": 0,
            "start_weight": 0.10,
            "target_iter": 0,
            "target_weight": 0.10,
            "category": "Stabilization",
        },
        {
            "name": "LDL Loss",
            "start_iter": 0,
            "start_weight": 0.3,
            "target_iter": 0,
            "target_weight": 0.3,
            "disable_after": 90000,
            "category": "Stabilization",
        },
        {
            "name": "HFEN Loss",
            "start_iter": 0,
            "start_weight": 0.015,
            "target_iter": 0,
            "target_weight": 0.015,
            "category": "Detail",
        },
        {
            "name": "Adaptive Block TV",
            "start_iter": 0,
            "start_weight": 0.004,
            "target_iter": 0,
            "target_weight": 0.004,
            "category": "Detail",
        },
        {
            "name": "R3GAN Loss",
            "start_iter": 30000,
            "start_weight": 0.00,
            "target_iter": 45000,
            "target_weight": 0.06,
            "category": "Adversarial",
        },
        {
            "name": "Contrastive Loss",
            "start_iter": 60000,
            "start_weight": 0.00,
            "target_iter": 60000,
            "target_weight": 0.05,
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

        if "disable_after" in loss:
            weight_array[loss["disable_after"] :] = 0

        schedules[loss["name"]] = {
            "weights": weight_array,
            "category": loss["category"],
            "start_iter": loss["start_iter"],
            "target_iter": loss["target_iter"],
        }

    return schedules


def analyze_schedule_detailed(schedules: dict) -> str:
    """Comprehensive analysis of the loss schedule."""

    total_iter = 140000
    analysis = []
    analysis.append("=" * 80)
    analysis.append(
        "LOSS SCHEDULE ANALYSIS - ParagonSR2 Static-S with Feature Matching"
    )
    analysis.append("=" * 80)
    analysis.append("")

    # Training phases analysis
    analysis.append("1. TRAINING PHASES ANALYSIS:")
    analysis.append("-" * 40)

    # Define training phases
    phases = [
        (0, 30000, "Early Reconstruction Phase"),
        (30000, 60000, "GAN Introduction Phase"),
        (60000, 90000, "Multi-Loss Balance Phase"),
        (90000, 140000, "Fine-tuning Phase"),
    ]

    for start, end, phase_name in phases:
        phase_weights = []
        for name, data in schedules.items():
            phase_avg = np.mean(data["weights"][start:end])
            if phase_avg > 0:
                phase_weights.append((name, phase_avg))

        phase_weights.sort(key=lambda x: x[1], reverse=True)
        analysis.append(f"\n{phase_name} ({start:,} - {end:,} iterations):")
        analysis.append(f"  Total active losses: {len(phase_weights)}")
        analysis.append("  Top contributors:")
        for i, (name, weight) in enumerate(phase_weights[:5]):
            analysis.append(f"    {i + 1}. {name}: {weight:.3f}")

    analysis.append("")

    # Loss scheduling strategy analysis
    analysis.append("2. LOSS SCHEDULING STRATEGY:")
    analysis.append("-" * 40)

    # Categorize losses by start time
    start_times = {}
    for name, data in schedules.items():
        start_iter = data["start_iter"]
        if start_iter not in start_times:
            start_times[start_iter] = []
        start_times[start_iter].append(name)

    for start_iter in sorted(start_times.keys()):
        losses = start_times[start_iter]
        analysis.append(f"\nStart at {start_iter:,} iterations ({len(losses)} losses):")
        for loss in losses:
            analysis.append(f"  • {loss}")

    analysis.append("")

    # Critical transition points
    analysis.append("3. CRITICAL TRANSITION POINTS:")
    analysis.append("-" * 40)

    transitions = [30000, 45000, 60000, 70000, 90000, 110000]
    transition_names = [
        "GAN Introduction",
        "GAN Full Strength",
        "Contrastive Learning",
        "Learning Rate Drop 1",
        "LDL Disabled",
        "Learning Rate Drop 2",
    ]

    for transition, name in zip(transitions, transition_names, strict=False):
        analysis.append(f"\n{transition:,} iterations - {name}:")

        # Show which losses are changing at this point
        for loss_name, data in schedules.items():
            weights_before = data["weights"][max(0, transition - 1000) : transition]
            weights_after = data["weights"][
                transition : min(total_iter, transition + 1000)
            ]

            if len(weights_before) > 0 and len(weights_after) > 0:
                avg_before = np.mean(weights_before)
                avg_after = np.mean(weights_after)

                if abs(avg_after - avg_before) > 0.01:  # Significant change
                    change = "↑" if avg_after > avg_before else "↓"
                    analysis.append(
                        f"  {change} {loss_name}: {avg_before:.3f} → {avg_after:.3f}"
                    )

    analysis.append("")

    # Potential conflicts and gaps
    analysis.append("4. POTENTIAL CONFLICTS AND GAPS:")
    analysis.append("-" * 40)

    # Check for high activity periods
    high_weight_threshold = 0.1
    for iter_step in [10000, 25000, 50000, 75000, 100000]:
        active_losses = [
            (name, data["weights"][iter_step])
            for name, data in schedules.items()
            if data["weights"][iter_step] > high_weight_threshold
        ]
        active_losses.sort(key=lambda x: x[1], reverse=True)

        analysis.append(
            f"\n{iter_step:,} iterations ({len(active_losses)} high-weight losses):"
        )
        for name, weight in active_losses:
            analysis.append(f"  • {name}: {weight:.3f}")

    analysis.append("")

    # Category balance analysis
    analysis.append("5. CATEGORY BALANCE ANALYSIS:")
    analysis.append("-" * 40)

    categories = {}
    for name, data in schedules.items():
        category = data["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append((name, data["weights"]))

    for category, losses in categories.items():
        analysis.append(f"\n{category} Losses ({len(losses)} losses):")

        # Calculate category contribution over time
        total_weights = np.zeros(total_iter + 1)
        for _, weights in losses:
            total_weights += weights

        # Find peak period
        peak_idx = np.argmax(total_weights)
        peak_weight = total_weights[peak_idx]

        analysis.append(
            f"  Peak activity: {peak_weight:.3f} at {peak_idx:,} iterations"
        )
        analysis.append(f"  Average activity: {np.mean(total_weights):.3f}")

        # Show when this category is most/least active
        high_periods = np.where(total_weights > peak_weight * 0.8)[0]
        if len(high_periods) > 0:
            analysis.append(
                f"  High activity period: {high_periods[0]:,} - {high_periods[-1]:,} iterations"
            )

    analysis.append("")

    # Recommendations
    analysis.append("6. OPTIMIZATION RECOMMENDATIONS:")
    analysis.append("-" * 40)

    recommendations = []

    # Check for rapid weight changes
    for name, data in schedules.items():
        weights = data["weights"]
        if len(weights) > 1:
            changes = np.abs(np.diff(weights))
            rapid_changes = np.where(changes > 0.1)[0]
            if len(rapid_changes) > 0:
                recommendations.append(
                    f"• {name}: Consider smoothing weight transitions (currently {len(rapid_changes)} rapid changes)"
                )

    # Check for timing conflicts
    start_iterations = {}
    for name, data in schedules.items():
        start_iter = data["start_iter"]
        if start_iter not in start_iterations:
            start_iterations[start_iter] = []
        start_iterations[start_iter].append(name)

    for start_iter, loss_names in start_iterations.items():
        if len(loss_names) > 4:
            recommendations.append(
                f"• Consider staggering {len(loss_names)} losses starting at {start_iter:,} iterations"
            )

    # Check for gaps
    total_activity = np.zeros(total_iter + 1)
    for data in schedules.values():
        total_activity += data["weights"]

    low_activity = np.where(total_activity < 0.5)[0]
    if len(low_activity) > 1000:  # More than 1000 iterations with low activity
        gap_start = low_activity[0]
        gap_end = low_activity[-1]
        recommendations.append(
            f"• Consider adding losses during low-activity period: {gap_start:,} - {gap_end:,} iterations"
        )

    if recommendations:
        for rec in recommendations:
            analysis.append(rec)
    else:
        analysis.append(
            "• No major scheduling issues detected - well balanced configuration!"
        )

    analysis.append("")

    # Summary statistics
    analysis.append("7. SUMMARY STATISTICS:")
    analysis.append("-" * 40)

    total_losses = len(schedules)
    active_losses_early = len(
        [data for data in schedules.values() if np.mean(data["weights"][:30000]) > 0]
    )
    active_losses_late = len(
        [data for data in schedules.values() if np.mean(data["weights"][90000:]) > 0]
    )

    analysis.append(f"Total loss functions: {total_losses}")
    analysis.append(f"Active in early phase (0-30k): {active_losses_early}")
    analysis.append(f"Active in late phase (90k+): {active_losses_late}")

    # Weight distribution
    all_weights = []
    for data in schedules.values():
        all_weights.extend(data["weights"][data["weights"] > 0])

    if all_weights:
        analysis.append(
            f"Weight range: {min(all_weights):.4f} - {max(all_weights):.4f}"
        )
        analysis.append(f"Average weight: {np.mean(all_weights):.4f}")

    analysis.append("")
    analysis.append("=" * 80)

    return "\n".join(analysis)


def create_simple_visualization_data(schedules: dict) -> str:
    """Create simple text-based visualization of the loss schedule."""

    total_iter = 140000
    visualization = []
    visualization.append("TEXT-BASED LOSS SCHEDULE VISUALIZATION")
    visualization.append("=" * 60)
    visualization.append("")

    # Create timeline showing major milestones
    milestones = [0, 30000, 45000, 60000, 70000, 90000, 110000, 140000]
    milestone_labels = ["START", "GAN", "GAN+", "CONTRAST", "LR-1", "LR-2", "END"]

    for i, (milestone, label) in enumerate(
        zip(milestones, milestone_labels, strict=False)
    ):
        if i < len(milestones) - 1:
            next_milestone = milestones[i + 1]
            segment_length = next_milestone - milestone

            visualization.append(f"{milestone:>6,} |{label}|")

            # Show loss activity in this segment
            segment_weights = np.zeros(total_iter + 1)
            for data in schedules.values():
                segment_weights += data["weights"]

            segment_activity = segment_weights[milestone:next_milestone]
            avg_activity = np.mean(segment_activity)

            # Create simple bar representation
            bar_length = int(avg_activity * 20)  # Scale to max 20 characters
            bar = "█" * bar_length + "░" * (20 - bar_length)

            visualization.append(f"         |{bar}| Activity: {avg_activity:.2f}")
            visualization.append("")

    visualization.append("Legend: █ = High Activity, ░ = Low Activity")
    visualization.append("")

    # Show key transitions
    visualization.append("KEY TRANSITIONS:")
    visualization.append("-" * 30)

    transitions = [
        (30000, "GAN losses start - adversarial training begins"),
        (45000, "GAN reaches full strength - perceptual quality focus"),
        (60000, "Contrastive learning starts - semantic understanding"),
        (70000, "First learning rate drop - fine-tuning begins"),
        (90000, "LDL disabled - focus shifts to global quality"),
        (110000, "Second learning rate drop - final optimization"),
    ]

    for iteration, description in transitions:
        visualization.append(f"{iteration:>6,} - {description}")

    return "\n".join(visualization)


def main() -> None:
    """Main analysis function."""

    print("Analyzing loss weight schedule...")

    # Get schedules
    schedules = get_loss_schedules()

    # Create detailed analysis
    analysis_text = analyze_schedule_detailed(schedules)

    # Create simple visualization
    viz_text = create_simple_visualization_data(schedules)

    # Save both outputs
    with open("loss_schedule_analysis_detailed.txt", "w") as f:
        f.write(analysis_text)

    with open("loss_schedule_visualization_simple.txt", "w") as f:
        f.write(viz_text)

    print("Analysis complete!")
    print("\nFiles created:")
    print("• loss_schedule_analysis_detailed.txt - Comprehensive analysis")
    print("• loss_schedule_visualization_simple.txt - Text visualization")
    print()

    # Show summary
    print("SUMMARY:")
    print("-" * 50)

    # Quick summary
    total_losses = len(schedules)
    categories = {data["category"] for data in schedules.values()}

    print(f"Total loss functions: {total_losses}")
    print(f"Categories: {', '.join(sorted(categories))}")

    # Show key phases
    print("\nTraining phases:")
    print("• 0-30k: Early reconstruction (reconstruction-heavy)")
    print("• 30k-60k: GAN introduction (adversarial training)")
    print("• 60k-90k: Multi-loss balance (full perceptual)")
    print("• 90k-140k: Fine-tuning (artifact reduction)")

    print("\nDetailed analysis saved to loss_schedule_analysis_detailed.txt")


if __name__ == "__main__":
    main()
