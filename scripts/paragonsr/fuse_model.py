import torch
from ParagonSR_arch import (
    paragonsr_s,  # Make sure your architecture file is in the same directory or accessible
)

# 1. Define your model structure (must match the trained model)
model = paragonsr_s(scale=4)

# 2. Load your trained weights
# Replace this with the actual path to your final checkpoint
trained_weights_path = "experiments/4x_ParagonSR_S/models/net_g_latest.pth"
state_dict = torch.load(trained_weights_path)
model.load_state_dict(state_dict)

# 3. Switch to evaluation mode. THIS IS CRITICAL.
model.eval()

# 4. Iterate through the model and call the fuse_kernels() method
#    on every ReparamConv block.
for module in model.modules():
    if hasattr(module, "fuse_kernels"):
        print(f"Fusing module: {module}")
        module.fuse_kernels()

# 5. Save the new, fused state_dict
# This new file contains the weights for the FAST, inference-only model.
fused_weights_path = "release_models/4x_ParagonSR_S_fused.pth"
torch.save(model.state_dict(), fused_weights_path)

print(f"\nModel successfully fused and saved to {fused_weights_path}")
