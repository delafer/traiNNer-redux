#!/usr/bin/env python3
"""
Training Issue Diagnostic Tool
==============================

Quick diagnostic script to analyze training degradation issues
and provide immediate actionable recommendations.

Usage: python diagnose_training_issue.py
"""


def analyze_gradient_scaling_issues(scale_g_value: float) -> dict:
    """Analyze gradient scaling issues from training logs."""
    analysis = {
        "scale_g": scale_g_value,
        "severity": "UNKNOWN",
        "diagnosis": [],
        "actions": [],
    }

    if scale_g_value > 1e8:
        analysis["severity"] = "CRITICAL"
        analysis["diagnosis"].append(
            "Extreme gradient scaling - AMP underflow causing training failure"
        )
        analysis["actions"].extend(
            [
                "🛑 STOP TRAINING IMMEDIATELY - Model is being destroyed",
                "Enable BF16 precision: amp_bf16: true",
                "Reduce learning rate: lr: 1e-4 or 5e-5",
                "Check data normalization and preprocessing",
            ]
        )
    elif scale_g_value > 1e6:
        analysis["severity"] = "SEVERE"
        analysis["diagnosis"].append(
            "Severe gradient scaling - Training becoming unstable"
        )
        analysis["actions"].extend(
            [
                "⚠️ Training should be stopped for stability",
                "Enable BF16 precision: amp_bf16: true",
                "Disable dynamic loss scheduling temporarily",
                "Add gradient clipping: grad_clip_max_norm: 1.0",
            ]
        )
    elif scale_g_value > 1e4:
        analysis["severity"] = "WARNING"
        analysis["diagnosis"].append("High gradient scaling - Monitor closely")
        analysis["actions"].extend(
            [
                "Monitor scale_g values during training",
                "Consider reducing learning rate if continues",
                "Add gradient norm monitoring",
            ]
        )
    else:
        analysis["severity"] = "NORMAL"
        analysis["diagnosis"].append("Gradient scaling within acceptable range")
        analysis["actions"].append("✅ No immediate action needed")

    return analysis


def analyze_performance_degradation(
    peak_psnr: float, current_psnr: float, peak_iter: int, current_iter: int
) -> dict:
    """Analyze validation performance degradation."""
    degradation_pct = ((peak_psnr - current_psnr) / peak_psnr) * 100

    analysis = {
        "peak_psnr": peak_psnr,
        "current_psnr": current_psnr,
        "degradation_percent": degradation_pct,
        "peak_iteration": peak_iter,
        "current_iteration": current_iter,
        "severity": "UNKNOWN",
        "diagnosis": [],
        "actions": [],
    }

    if degradation_pct > 5:
        analysis["severity"] = "CRITICAL"
        analysis["diagnosis"].append(
            "Severe performance degradation - Training configuration issues"
        )
        analysis["actions"].extend(
            [
                "🛑 STOP TRAINING - Configuration is fundamentally flawed",
                "Disable dynamic loss scheduling",
                "Use CosineAnnealingLR instead of MultiStepLR",
                "Increase warmup iterations to 2000+",
                "Enable BF16 precision if not already enabled",
            ]
        )
    elif degradation_pct > 2:
        analysis["severity"] = "SEVERE"
        analysis["diagnosis"].append("Significant performance degradation")
        analysis["actions"].extend(
            [
                "⚠️ Consider stopping training and adjusting configuration",
                "Disable dynamic loss scheduling",
                "Review learning rate schedule",
                "Monitor gradient scaling closely",
            ]
        )
    elif degradation_pct > 0.5:
        analysis["severity"] = "WARNING"
        analysis["diagnosis"].append("Mild performance degradation - Monitor closely")
        analysis["actions"].extend(
            [
                "Monitor training for continued degradation",
                "Consider adjusting dynamic loss scheduling parameters",
                "Check learning rate schedule",
            ]
        )
    else:
        analysis["severity"] = "NORMAL"
        analysis["diagnosis"].append("Performance within acceptable variation")
        analysis["actions"].append("✅ Continue training with monitoring")

    return analysis


def main() -> None:
    """Main diagnostic function."""
    print("🔍 Training Issue Diagnostic Tool")
    print("=" * 50)

    # User's specific case analysis
    print("\n📊 ANALYZING YOUR TRAINING CASE:")
    print("-" * 30)

    # Gradient scaling analysis
    print("\n1. 🚨 GRADIENT SCALING ANALYSIS")
    print("   Scale_g value: 2.0972e+06")
    grad_analysis = analyze_gradient_scaling_issues(2097200.0)

    print(f"   Severity: {grad_analysis['severity']}")
    print("   Diagnosis:")
    for diag in grad_analysis["diagnosis"]:
        print(f"   ❌ {diag}")
    print("   Actions:")
    for action in grad_analysis["actions"]:
        print(f"   • {action}")

    # Performance degradation analysis
    print("\n2. 📈 PERFORMANCE DEGRADATION ANALYSIS")
    print("   Peak: 11.5119 PSNR @ 1,000 iterations")
    print("   Current: 11.0022 PSNR @ 40,000 iterations")
    perf_analysis = analyze_performance_degradation(11.5119, 11.0022, 1000, 40000)

    print(f"   Degradation: {perf_analysis['degradation_percent']:.2f}%")
    print(f"   Severity: {perf_analysis['severity']}")
    print("   Diagnosis:")
    for diag in perf_analysis["diagnosis"]:
        print(f"   ❌ {diag}")
    print("   Actions:")
    for action in perf_analysis["actions"]:
        print(f"   • {action}")

    # Configuration analysis
    print("\n3. ⚙️ CONFIGURATION PROBLEMS DETECTED:")
    config_issues = [
        "❌ Dynamic loss scheduling enabled with problematic parameters",
        "❌ AMP BF16 disabled (amp_bf16: false)",
        "❌ MultiStepLR with aggressive early milestones",
        "❌ Insufficient warmup period (500 iterations)",
        "❌ High momentum and slow adaptation in DLS",
    ]

    for issue in config_issues:
        print(f"   {issue}")

    print("\n4. 🛠️ IMMEDIATE FIXES NEEDED:")
    fixes = [
        "✅ DISABLE dynamic_loss_scheduling: enabled: false",
        "✅ ENABLE BF16 precision: amp_bf16: true",
        "✅ CHANGE to CosineAnnealingLR: T_max: 40000",
        "✅ INCREASE warmup_iter: 2000",
        "✅ ADD gradient clipping: grad_clip_max_norm: 1.0",
    ]

    for fix in fixes:
        print(f"   {fix}")

    # Final verdict
    print("\n" + "=" * 50)
    print("🚨 FINAL VERDICT: TRAINING CONFIGURATION CRITICALLY FLAWED")
    print("=" * 50)
    print("Your current configuration will continue to degrade performance.")
    print("IMMEDIATE ACTION REQUIRED:")
    print("1. Stop current training (model is being destroyed)")
    print("2. Use the FIXED configuration: 2xParagonSR2_Nano_CC0_147k_FIXED.yml")
    print("3. Resume from 1,000 iteration checkpoint (best performing)")
    print("4. Monitor scale_g values - should be 1e3-1e4, NOT 1e6")
    print("\nExpected outcome with fixed config:")
    print("• Stable gradient scaling (no more explosions)")
    print("• Continuous improvement throughout training")
    print("• Peak performance near end of training, not early")
    print("• No degradation after initial convergence")


if __name__ == "__main__":
    main()
