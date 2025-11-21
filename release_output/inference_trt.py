import argparse
import os
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""): return iterable

import cv2
import numpy as np
import tensorrt as trt
import torch

# Suppress TRT logs
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTWrapper:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(TRT_LOGGER)
        
        print(f"Loading Engine: {engine_path}")
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"Error loading engine: {e}")
            sys.exit(1)
        
        self.context = self.engine.create_execution_context()
        self.input_name = "input"
        self.output_name = "output"
        
        # Create a dedicated CUDA stream for this context (Fixes the warning)
        self.stream = torch.cuda.Stream()

    def infer(self, img_tensor, scale_factor):
        """
        img_tensor: PyTorch tensor on CUDA, shape (1, 3, H, W). MUST BE CONTIGUOUS.
        """
        # 1. Set Input Shape
        b, c, h, w = img_tensor.shape
        self.context.set_input_shape(self.input_name, (b, c, h, w))
        
        # 2. Calculate Output Shape
        target_h = int(h * scale_factor)
        target_w = int(w * scale_factor)
        
        # 3. Allocate Output Tensor
        output_tensor = torch.empty((b, c, target_h, target_w), dtype=torch.float32, device=img_tensor.device)
        
        # 4. Bind Tensors
        self.context.set_tensor_address(self.input_name, img_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())
        
        # 5. Execute on dedicated stream
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        # 6. Synchronize the stream
        self.stream.synchronize()
        
        return output_tensor

def process_image(img_path, wrapper, output_dir, scale):
    filename = Path(img_path).name
    save_path = output_dir / filename
    
    # 1. Read Image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [Error] Could not read {filename}")
        return
        
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Preprocess
    # Numpy -> Tensor -> Float -> Permute -> Unsqueeze -> Normalize
    img_t = torch.from_numpy(img).cuda().float()
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    img_t = img_t / 255.0
    
    # --- CRITICAL FIX: FORCE MEMORY REORDERING ---
    # Without this, data_ptr() points to HWC memory, confusing TensorRT
    img_t = img_t.contiguous() 
    # ---------------------------------------------
    
    try:
        # 3. Inference
        out_t = wrapper.infer(img_t, scale)
        
        # 4. Postprocess
        out = out_t.squeeze(0).permute(1, 2, 0).clamp_(0, 1).mul_(255.0).byte().cpu().numpy()
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        
        # 5. Save
        cv2.imwrite(str(save_path), out)
        
    except Exception as e:
        print(f"  [Error] Failed to process {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TensorRT Inference")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="trt_results")
    parser.add_argument("--scale", type=float, required=True)
    
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    wrapper = TRTWrapper(args.engine)
    
    if input_path.is_file():
        images = [input_path]
    else:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = sorted([p for p in input_path.glob("*") if p.suffix.lower() in exts])
    
    if not images:
        print("No images found.")
        sys.exit(0)
        
    print(f"Processing {len(images)} images with scale {args.scale}x...")
    
    for img in tqdm(images, desc="Upscaling"):
        process_image(img, wrapper, out_dir, args.scale)
        
    print(f"\nDone! Results saved to: {out_dir}")
