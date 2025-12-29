#!/usr/bin/env python3
"""
ParagonSR2 Gradio App (Hugging Face Spaces / ZeroGPU)
=====================================================

A simple web interface for ParagonSR2 image and video upscaling.

Features:
    - Drag-and-drop image/video upload
    - Model selection (Realtime/Stream/Photo variants)
    - Automatic backend selection (TRT > Compiled > PyTorch)
    - Video temporal stabilization via Feature-Tap

Usage:
    python app.py
    # Opens at http://localhost:7860

For Hugging Face Spaces:
    1. Create a new Space with Gradio SDK
    2. Add this file as app.py
    3. Add run_inference.py and paragonsr2_arch.py to the repo
    4. Upload model weights to a 'models/' folder
    5. Add requirements: torch, gradio, opencv-python, safetensors, tqdm

Author: Philip Hofmann
License: MIT
"""

from pathlib import Path

import cv2
import gradio as gr

# Import inference logic from run_inference.py
from run_inference import InferenceOrchestrator


class Args:
    """Simple namespace to mimic argparse results."""


def upscale(
    input_file: str | None,
    model_name: str,
    scale: int,
    disable_compile: bool,
) -> tuple[str | None, str]:
    """
    Process an uploaded image or video.

    Args:
        input_file: Path to uploaded file
        model_name: Model variant name
        scale: Upscale factor
        disable_compile: Whether to disable torch.compile

    Returns:
        Tuple of (output_path, status_message)
    """
    if input_file is None:
        return None, "❌ No input file provided"

    # Setup paths
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Map model name to architecture
    if "realtime" in model_name.lower():
        arch = "paragonsr2_realtime"
    elif "stream" in model_name.lower():
        arch = "paragonsr2_stream"
    else:
        arch = "paragonsr2_photo"

    model_path = models_dir / f"{model_name}.safetensors"

    # Check if model exists
    if not model_path.exists():
        return (
            None,
            f"❌ Model not found: {model_path}\n\nPlease upload model weights to the 'models/' folder.",
        )

    # Create args namespace
    args = Args()
    args.model = str(model_path)
    args.arch = arch
    args.scale = int(scale)
    args.fp16 = True
    args.disable_compile = disable_compile

    # Initialize orchestrator
    try:
        orchestrator = InferenceOrchestrator(args)
    except Exception as e:
        return None, f"❌ Error loading model: {e}"

    # Process file
    input_path = Path(input_file)
    is_video = input_path.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov", ".webm"]
    out_name = f"upscaled_{input_path.stem}{'.mp4' if is_video else '.png'}"
    out_path = output_dir / out_name

    try:
        if is_video:
            orchestrator.process_video(input_path, out_path)
            return str(out_path), f"✅ Video processed with {orchestrator.mode} backend"
        else:
            img = cv2.imread(str(input_path))
            if img is None:
                return None, f"❌ Could not read image: {input_path.name}"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = orchestrator.process_image(img)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), result_bgr)
            return str(out_path), f"✅ Image processed with {orchestrator.mode} backend"
    except Exception as e:
        return None, f"❌ Processing error: {e}"


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(
    title="ParagonSR2 Upscaler",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🚀 ParagonSR2 Universal Upscaler

        AI-powered image and video super-resolution with temporal stability.

        **How to use:**
        1. Upload an image or video
        2. Select a model variant
        3. Click "Upscale"

        **Models:**
        - **Realtime**: Fastest, lowest quality. Good for previews.
        - **Stream**: Balanced speed/quality. Good for video.
        - **Photo**: Best quality. Good for final renders.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(
                label="📁 Input (Image or Video)",
                file_types=["image", "video"],
            )
            model_dropdown = gr.Dropdown(
                choices=[
                    "2xParagonSR2_Realtime_fidelity",
                    "2xParagonSR2_Stream_fidelity",
                    "2xParagonSR2_Photo_fidelity",
                ],
                value="2xParagonSR2_Photo_fidelity",
                label="🧠 Model",
            )
            scale_input = gr.Number(
                value=2,
                label="📏 Scale Factor",
                minimum=1,
                maximum=4,
                precision=0,
            )
            disable_compile_checkbox = gr.Checkbox(
                label="🔧 Disable torch.compile (use if crashing)",
                value=False,
            )
            upscale_button = gr.Button("✨ Upscale", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_file = gr.File(label="📤 Output")
            status_text = gr.Textbox(label="📊 Status", interactive=False)

    # Event handlers
    upscale_button.click(
        fn=upscale,
        inputs=[input_file, model_dropdown, scale_input, disable_compile_checkbox],
        outputs=[output_file, status_text],
    )

    gr.Markdown(
        """
        ---
        **Note:** First run may take longer due to model loading and compilation.

        Made with ❤️ by Philip Hofmann | [GitHub](https://github.com/Phhofm/traiNNer-redux)
        """
    )


if __name__ == "__main__":
    demo.queue().launch()
