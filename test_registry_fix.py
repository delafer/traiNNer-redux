import logging
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from traiNNer.utils.registry import AUTOMATION_REGISTRY
from traiNNer.utils.training_automations import (
    AdaptiveGradientClipping,
    DynamicBatchAndPatchSizeOptimizer,
    IntelligentEarlyStopping,
    IntelligentLearningRateScheduler,
    TrainingAutomationManager,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_registry() -> None:
    print("Testing Registry Alias Registration...")

    # Check if aliases are present
    aliases = [
        "adaptive_gradient_clipping",
        "intelligent_learning_rate_scheduler",
        "dynamic_batch_and_patch_size_optimizer",
        "intelligent_early_stopping",
    ]

    for alias in aliases:
        if alias in AUTOMATION_REGISTRY:
            print(f"✅ Alias found: {alias}")
            obj = AUTOMATION_REGISTRY.get(alias)
            print(f"   Mapped to: {obj.__name__}")
        else:
            print(f"❌ Alias NOT found: {alias}")

    # Test Manager Init
    print("\nTesting TrainingAutomationManager Init...")
    config = {
        "intelligent_learning_rate_scheduler": {"enabled": True},
        "adaptive_gradient_clipping": {"enabled": True},
    }

    try:
        manager = TrainingAutomationManager(config)
        print("✅ TrainingAutomationManager initialized successfully")
        print(f"Enabled automations: {manager.enabled_automations}")
    except Exception as e:
        print(f"❌ TrainingAutomationManager init failed: {e}")


if __name__ == "__main__":
    test_registry()
