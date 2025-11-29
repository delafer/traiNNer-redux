#!/usr/bin/env python3
"""
Training Log File Finder
Quick utility to find and display training log files for your experiments.
"""

import glob
import os
import sys
from datetime import datetime
from pathlib import Path


def find_training_logs(experiments_dir="experiments") -> None:
    """Find all training log files and display them."""

    if not os.path.exists(experiments_dir):
        print(f"❌ Experiments directory '{experiments_dir}' not found!")
        print("💡 Run some training first to generate logs.")
        return

    print(f"🔍 Searching for training logs in '{experiments_dir}/'...")
    print("=" * 80)

    # Find all log files
    log_pattern = os.path.join(experiments_dir, "**/train_*.log")
    log_files = glob.glob(log_pattern, recursive=True)

    if not log_files:
        print("❌ No training log files found!")
        return

    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    print(f"📁 Found {len(log_files)} training log file(s):\n")

    for i, log_file in enumerate(log_files, 1):
        # Get file info
        stat = os.stat(log_file)
        file_size = stat.st_size
        modified_time = datetime.fromtimestamp(stat.st_mtime)

        # Get experiment name from path
        experiment_path = Path(log_file).parent.parent
        experiment_name = experiment_path.name

        # Format file size
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} B"

        print(f"{i:2d}. {experiment_name}")
        print(f"    📄 {log_file}")
        print(f"    📊 {size_str} | 🕒 {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()


def show_log_preview(log_file, lines=20) -> None:
    """Show a preview of a log file."""
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return

    print(f"\n📖 Preview of {os.path.basename(log_file)} (last {lines} lines):")
    print("-" * 80)

    try:
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            preview_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            for line in preview_lines:
                print(line.rstrip())
    except Exception as e:
        print(f"❌ Error reading log file: {e}")


def main() -> None:
    """Main function with interactive options."""
    print("🎯 traiNNer Training Log File Finder")
    print("=" * 50)

    # Find all logs
    find_training_logs()

    # Interactive selection
    log_pattern = os.path.join("experiments", "**/train_*.log")
    log_files = glob.glob(log_pattern, recursive=True)

    if log_files:
        print("\n" + "=" * 50)
        print("💡 Quick Commands:")
        print("   • View specific log: python find_training_logs.py <log_number>")
        print("   • Example: python find_training_logs.py 1")
        print("   • All logs are stored in: experiments/{config_name}/logs/")

    # Check for command line argument
    if len(os.sys.argv) > 1:
        try:
            log_index = int(os.sys.argv[1]) - 1
            log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            if 0 <= log_index < len(log_files):
                show_log_preview(log_files[log_index])
            else:
                print(f"❌ Invalid log number. Use 1-{len(log_files)}")
        except ValueError:
            print("❌ Invalid argument. Use log number (e.g., 1, 2, 3)")


if __name__ == "__main__":
    main()
