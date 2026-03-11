# Wan2.1 Text-to-Video with OpenVINO

Generate short videos from text prompts using [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) with [OpenVINO](https://docs.openvino.ai/) acceleration on Intel hardware (iGPU / CPU).

This standalone script is derived from the [OpenVINO Notebooks wan2.1 example](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/wan2.1-text-to-video) and includes:

- **CausVid LoRA** distillation — reduces inference from 50 steps to just **4 steps**
- **INT4 weight compression** (NNCF) — smaller model footprint and faster inference
- **Intel iGPU support** — runs transformer & text encoder on integrated GPU for ~36x speedup over CPU

## Prerequisites

- **Python 3.10–3.14** (tested on 3.12)
- **Intel GPU driver** (for iGPU inference)
- **OpenVINO Runtime** (2025.1+ recommended)

### Install Dependencies

```bash
# (Optional) Create and activate a virtual environment
python -m venv wan2.1-venv
# Windows
wan2.1-venv\Scripts\Activate.ps1
# Linux/macOS
source wan2.1-venv/bin/activate

# (Optional) If you have OpenVINO GenAI package, source it first
# Windows PowerShell:
& "C:\path\to\openvino_genai\setupvars.ps1"

# Install required packages
pip install "torch>=2.1" "git+https://github.com/huggingface/diffusers.git" \
    "transformers>=4.49.0" "accelerate" "safetensors" "sentencepiece" \
    "peft>=0.15.0" "ftfy" "opencv-python" "regex" "huggingface_hub" \
    "imageio" "imageio-ffmpeg" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install OpenVINO + NNCF (if not already available from setupvars)
pip install --pre -U "openvino>=2025.1.0" "nncf>=2.16.0" \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

## Model Conversion

The script downloads [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) from HuggingFace, fuses the [CausVid LoRA](https://causvid.github.io/) adapter for 4-step fast generation, and converts three submodels to OpenVINO IR with INT4 compression:

| Component | Description | Compression |
|---|---|---|
| **Transformer** | Diffusion transformer (denoising) | INT4_ASYM (group_size=64) |
| **Text Encoder** | T5-based text encoder | INT4_ASYM (group_size=64) |
| **VAE Decoder** | Video latent decoder | INT8_ASYM |

### Convert Only

```bash
python wan2.1_text_to_video.py --convert-only
```

This creates `Wan2.1-T2V-1.3B-Diffusers/INT4/` with:
```
Wan2.1-T2V-1.3B-Diffusers/INT4/
├── transformer.xml / .bin
├── text_encoder.xml / .bin
├── vae_decoder.xml / .bin
├── tokenizer/
└── scheduler/
```

## Usage

### Generate a Video (with conversion)

First run converts the model, then generates the video:

```bash
python wan2.1_text_to_video.py --prompt "A cat walks on the grass, realistic"
```

### Generate a Video (skip conversion)

If the model is already converted:

```bash
python wan2.1_text_to_video.py --skip-conversion --prompt "A beautiful sunset over the ocean"
```

### Run on Intel iGPU (Recommended)

```bash
python wan2.1_text_to_video.py --skip-conversion \
    --prompt "A cat walks on the grass, realistic" \
    --device GPU --vae-device CPU \
    --output output_igpu.mp4
```

- `--device GPU` — runs **transformer** and **text encoder** on Intel iGPU
- `--vae-device CPU` — runs **VAE decoder** on CPU (see [note below](#why-vae-runs-on-cpu))

### Run on CPU

```bash
python wan2.1_text_to_video.py --skip-conversion \
    --prompt "A cat walks on the grass, realistic" \
    --device CPU --vae-device CPU \
    --output output_cpu.mp4
```

### All Options

```
usage: wan2.1_text_to_video.py [-h] [--model-id MODEL_ID] [--model-dir MODEL_DIR]
                                [--skip-conversion] [--convert-only]
                                [--device DEVICE] [--vae-device VAE_DEVICE]
                                [--prompt PROMPT] [--negative-prompt NEGATIVE_PROMPT]
                                [--height HEIGHT] [--width WIDTH]
                                [--num-frames NUM_FRAMES] [--num-steps NUM_STEPS]
                                [--guidance-scale GUIDANCE_SCALE] [--output OUTPUT]

Options:
  --model-id        HuggingFace model ID (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
  --model-dir       Directory for converted model (default: <model_name>/INT4)
  --skip-conversion Skip model conversion (use pre-converted model)
  --convert-only    Only convert the model, don't generate video
  --device          Device for transformer & text encoder: GPU, CPU, AUTO (default: GPU)
  --vae-device      Device for VAE decoder (default: CPU)
  --prompt          Text prompt for video generation
  --negative-prompt Negative prompt
  --height          Video height in pixels (default: 480)
  --width           Video width in pixels (default: 832)
  --num-frames      Number of frames to generate (default: 53, ~5.3s at 10fps)
  --num-steps       Number of inference steps (default: 4)
  --guidance-scale  Classifier-free guidance scale (default: 1.0)
  --output          Output video file path (default: output.mp4)
```

## CPU vs iGPU Performance

Benchmark on Intel Core Ultra (53 frames / 5.3s video, 480×832, 4 steps, INT4):

| | **iGPU** (`--device GPU`) | **CPU** (`--device CPU`) |
|---|---|---|
| Denoising (4 steps) | ~42s (~10.6s/step) | ~1427s (~357s/step) |
| Total wall time | ~386s (incl. model loading) | ~1813s (~30 min) |
| **Speedup** | **~34x faster** | baseline |

> **Note:** The first iGPU run includes GPU graph compilation overhead. Subsequent runs in the same process would be faster. The denoising itself is only ~42s on iGPU vs ~1427s on CPU.

### Why VAE runs on CPU

The VAE decoder triggers a `CL_INVALID_WORK_GROUP_SIZE` error on some Intel iGPUs. Since the VAE is only called **once** at the end to decode latents into video frames, running it on CPU has minimal impact on total generation time. The transformer (the performance-critical component running 4 denoising steps) runs on iGPU where it matters most.

## Sample Output

Prompt: *"A cat walks on the grass, realistic"* — 53 frames, 480×832, 4 steps, INT4

### iGPU Result (~42s denoising)

https://github.com/jlee52tw/wan2.1-txt-to-video-openvino/raw/main/samples/output_igpu_5s.mp4

### CPU Result (~1427s denoising)

https://github.com/jlee52tw/wan2.1-txt-to-video-openvino/raw/main/samples/output_cpu_5s.mp4

## Output Details

- **Video format:** MP4, 10 FPS
- **Default resolution:** 832×480
- **Default length:** 53 frames → **5.3 seconds** of video

## Files

| File | Description |
|---|---|
| `wan2.1_text_to_video.py` | Main standalone script — handles conversion and inference |
| `ov_wan_helper.py` | Helper module (auto-downloaded from [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)) — contains model conversion logic and `OVWanPipeline` inference class |

## How It Works

1. **Download** — fetches [Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) from HuggingFace
2. **LoRA Fusion** — loads and fuses [CausVid LoRA](https://causvid.github.io/) weights (enables 4-step generation instead of 50)
3. **Convert** — converts PyTorch model to OpenVINO IR using `ov.convert_model()`
4. **Compress** — applies INT4 asymmetric weight compression via NNCF
5. **Inference** — loads OpenVINO IR, runs denoising loop on iGPU, decodes with VAE on CPU
6. **Export** — saves generated frames as MP4 video

## Credits

- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — video generation model by Wan-AI
- [CausVid](https://causvid.github.io/) — causal video distillation for fast inference
- [OpenVINO](https://docs.openvino.ai/) — Intel's inference optimization toolkit
- [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks) — original notebook implementation

## License

This project follows the licensing of the upstream components. See the original [OpenVINO Notebooks license](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/LICENSE).
