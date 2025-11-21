import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
import traiNNer.archs.paragonsr2_static_arch
from traiNNer.utils.registry import ARCH_REGISTRY

# 1. Setup
ARCH = 'paragonsr2_static_s'
SCALE = 2
CHECKPOINT = 'experiments/2xParagonSR2_S_with_FeatureMatching_Optimized/models/net_g_ema_140000.safetensors'
ONNX_MODEL = 'release_output/paragonsr2_static_s_fp32.onnx'
IMG_PATH = '/home/phips/Documents/dataset/cc0/val_hr/001.png' # Pick one image

# 2. Load PyTorch Model
arch_fn = ARCH_REGISTRY.get(ARCH)
model = arch_fn(scale=SCALE)
from safetensors.torch import load_file
model.load_state_dict(load_file(CHECKPOINT), strict=True)
model.cuda().eval()
if hasattr(model, "fuse_for_release"): model.fuse_for_release()

# 3. Prepare Input
img = Image.open(IMG_PATH).convert("RGB")
# Resize to something standard for testing
img = img.resize((640, 480)) 
inp = np.array(img).astype(np.float32) / 255.0
inp = np.transpose(inp, (2, 0, 1))[None, ...].astype(np.float32)

# 4. PyTorch Inference
with torch.no_grad():
    pt_out = model(torch.from_numpy(inp).cuda()).cpu().numpy()

# 5. TensorRT Inference (via ONNX Runtime)
# Note: This uses the ONNX file but compiles it with TRT on the fly, 
# effectively testing the TRT FP16 conversion logic.
sess_opts = ort.SessionOptions()
providers = [
    ('TensorrtExecutionProvider', {
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache'
    }),
    'CUDAExecutionProvider'
]
sess = ort.InferenceSession(ONNX_MODEL, sess_opts, providers=providers)
trt_out = sess.run(None, {"input": inp})[0]

# 6. Compare
mse = np.mean((pt_out - trt_out) ** 2)
print(f"MSE Difference: {mse:.8f}")
if mse < 1e-4:
    print("✅ TensorRT FP16 Match Confirmed!")
else:
    print("❌ Mismatch detected!")
