import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
        self.stream = torch.cuda.Stream()
        
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

    def infer(self, img_tensor, scale_factor):
        """
        img_tensor: Contiguous (1, 3, H, W) on GPU
        """
        b, c, h, w = img_tensor.shape
        self.context.set_input_shape(self.input_name, (b, c, h, w))
        
        target_h, target_w = int(h * scale_factor), int(w * scale_factor)
        
        # Output tensor
        output_tensor = torch.empty((b, c, target_h, target_w), dtype=torch.float32, device=img_tensor.device)
        
        # Bindings
        self.context.set_tensor_address(self.input_name, img_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())
        
        # Async execution
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        # Sync only this stream, not the whole device
        self.stream.synchronize()
        
        return output_tensor

def save_image_worker(image_numpy, save_path):
    """Background worker to save image"""
    try:
        # Color conversion happens here on CPU background thread
        out_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), out_bgr)
    except Exception as e:
        print(f"Error saving {save_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="High-Performance TRT Inference")
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
    
    # ThreadPool for saving images in background
    # max_workers=4 is usually enough to saturate disk I/O without hogging CPU
    saver_pool = ThreadPoolExecutor(max_workers=4)
    
    # Warmup GPU (Optional but good for timing)
    dummy = torch.zeros(1, 3, 64, 64).cuda().contiguous()
    wrapper.infer(dummy, args.scale)
    
    start_time = time.time()
    
    for img_path in tqdm(images, desc="Upscaling"):
        filename = img_path.name
        save_path = out_dir / filename
        
        # 1. Load (Main Thread)
        img = cv2.imread(str(img_path))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. To GPU
        img_t = torch.from_numpy(img).cuda().float().permute(2, 0, 1).unsqueeze(0)
        img_t = img_t.div_(255.0).contiguous()
        
        # 3. Infer (GPU)
        # While this runs, the saver threads are saving PREVIOUS images
        out_t = wrapper.infer(img_t, args.scale)
        
        # 4. Postprocess (GPU -> CPU)
        # We move to CPU immediately so we can hand off to saver thread
        out_np = out_t.squeeze(0).permute(1, 2, 0).clamp_(0, 1).mul_(255.0).byte().cpu().numpy()
        
        # 5. Async Save
        # This returns immediately, allowing the loop to process the next image
        saver_pool.submit(save_image_worker, out_np, save_path)
        
    # Wait for all saves to finish
    saver_pool.shutdown(wait=True)
    
    total_time = time.time() - start_time
    fps = len(images) / total_time
    print(f"\nDone! {len(images)} images in {total_time:.2f}s ({fps:.2f} FPS)")

if __name__ == "__main__":
    main()
