"""
Wan2.1 Text-to-Video Generation with OpenVINO - Standalone Script

Converts and runs the Wan2.1-T2V-1.3B-Diffusers model with CausVid LoRA
(4-step distilled fast generation) using OpenVINO on Intel iGPU.

Usage:
    # Full pipeline: convert model + generate video
    python wan2.1_text_to_video.py --prompt "A cat walks on the grass, realistic"

    # Skip conversion if model already converted
    python wan2.1_text_to_video.py --skip-conversion --prompt "A beautiful sunset over the ocean"

    # Custom parameters
    python wan2.1_text_to_video.py --prompt "A dog running in the park" --num-frames 33 --height 480 --width 832 --output dog.mp4
"""

import argparse
import sys
import gc
from pathlib import Path


def download_helpers():
    """Download helper files if not present locally."""
    import requests

    helpers = {
        "ov_wan_helper.py": "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/wan2.1-text-to-video/ov_wan_helper.py",
    }
    for filename, url in helpers.items():
        if not Path(filename).exists():
            print(f"Downloading {filename}...")
            r = requests.get(url=url)
            r.raise_for_status()
            with open(filename, "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"  ✅ {filename} downloaded")
        else:
            print(f"  ✅ {filename} already exists")


def convert_model(model_id: str, model_dir: Path):
    """Convert the Wan2.1 model to OpenVINO IR with INT4 compression and CausVid LoRA."""
    import nncf
    from ov_wan_helper import convert_pipeline

    weights_compression_config = {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 64,
        "ratio": 1.0,
    }

    print(f"\n{'='*60}")
    print(f"Model Conversion")
    print(f"  Model:       {model_id}")
    print(f"  Output:      {model_dir}")
    print(f"  Compression: INT4_ASYM (group_size=64, ratio=1.0)")
    print(f"  LoRA:        CausVid (4-step fast generation)")
    print(f"{'='*60}\n")

    convert_pipeline(model_id, model_dir, apply_lora=True, compression_config=weights_compression_config)

    print(f"\n✅ Model conversion complete. Files in: {model_dir}")


def generate_video(
    model_dir: Path,
    device: str,
    vae_device: str,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    output_path: str,
):
    """Run inference with the converted OpenVINO model."""
    from ov_wan_helper import OVWanPipeline
    from diffusers.utils import export_to_video

    device_map = {
        "transformer": device,
        "text_encoder": device,
        "vae": vae_device,
    }

    print(f"\n{'='*60}")
    print(f"Video Generation")
    print(f"  Model dir:   {model_dir}")
    print(f"  Device:      Transformer={device}, TextEncoder={device}, VAE={vae_device})")
    print(f"  Resolution:  {width}x{height}")
    print(f"  Frames:      {num_frames}")
    print(f"  Steps:       {num_inference_steps}")
    print(f"  Guidance:    {guidance_scale}")
    print(f"  Prompt:      {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"  Output:      {output_path}")
    print(f"{'='*60}\n")

    print("⌛ Loading pipeline...")
    ov_pipe = OVWanPipeline(model_dir, device_map)
    print("✅ Pipeline loaded\n")

    print("⌛ Generating video...")
    output = ov_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).frames[0]

    export_to_video(output, output_path, fps=10)
    print(f"\n✅ Video saved to: {output_path}")

    del ov_pipe
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Wan2.1 Text-to-Video Generation with OpenVINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wan2.1_text_to_video.py --prompt "A cat walks on the grass, realistic"
  python wan2.1_text_to_video.py --skip-conversion --prompt "Ocean waves at sunset"
  python wan2.1_text_to_video.py --prompt "A dog" --device CPU --num-frames 17
        """,
    )

    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="HuggingFace model ID (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory for converted model (default: <model_name>/INT4)",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip model conversion (use pre-converted model)",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert the model, don't generate video",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="OpenVINO device for transformer & text encoder: GPU (iGPU), CPU, AUTO (default: GPU)",
    )
    parser.add_argument(
        "--vae-device",
        type=str,
        default="CPU",
        help="OpenVINO device for VAE decoder (default: CPU, iGPU may hit work group size limits)",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat walks on the grass, realistic",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Negative prompt",
    )
    parser.add_argument("--height", type=int, default=480, help="Video height (default: 480)")
    parser.add_argument("--width", type=int, default=832, help="Video width (default: 832)")
    parser.add_argument("--num-frames", type=int, default=53, help="Number of frames (default: 53, ~5.3s at 10fps)")
    parser.add_argument("--num-steps", type=int, default=4, help="Inference steps (default: 4, CausVid distilled)")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale (default: 1.0)")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path (default: output.mp4)")

    args = parser.parse_args()

    # Resolve model directory
    if args.model_dir is None:
        model_name = args.model_id.split("/")[-1]
        model_dir = Path(model_name) / "INT4"
    else:
        model_dir = Path(args.model_dir)

    # Change to the script's directory so helper imports work
    script_dir = Path(__file__).resolve().parent
    import os
    os.chdir(script_dir)

    # Download helpers
    print("Checking helper files...")
    download_helpers()

    # Step 1: Model conversion
    if not args.skip_conversion:
        convert_model(args.model_id, model_dir)
    else:
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            print("   Run without --skip-conversion first to convert the model.")
            sys.exit(1)
        print(f"⏭️  Skipping conversion. Using existing model at: {model_dir}")

    if args.convert_only:
        print("\n✅ Conversion done. Exiting (--convert-only).")
        return

    # Step 2: Generate video
    generate_video(
        model_dir=model_dir,
        device=args.device,
        vae_device=args.vae_device,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        output_path=args.output,
    )

    # List available devices for reference
    import openvino as ov
    print(f"\nAvailable OpenVINO devices: {ov.Core().available_devices}")


if __name__ == "__main__":
    main()
