import torch
from ParagonSR_arch import paragonsr_s

# 1. Define your model structure
model = paragonsr_s(scale=4)

# 2. Load the FUSED weights you just created
fused_weights_path = "release_models/4x_ParagonSR_S_fused.pth"
state_dict = torch.load(fused_weights_path)
model.load_state_dict(state_dict)

# 3. Switch to evaluation mode
model.eval()

# 4. Create a dummy input tensor
# The dimensions (H, W) don't have to be exact, but should be representative.
# Using dynamic axes allows the ONNX model to accept images of any size.
dummy_input = torch.randn(1, 3, 256, 256)
input_names = ["input"]
output_names = ["output"]
onnx_path = "release_models/4x_ParagonSR_S.onnx"

# 5. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    verbose=False,
    input_names=input_names,
    output_names=output_names,
    export_params=True,
    opset_version=11,  # A common, widely supported version
    dynamic_axes={
        "input": {2: "height", 3: "width"},
        "output": {2: "height", 3: "width"},
    },
)

print(f"ONNX model successfully exported to {onnx_path}")
