# 🧠 Fine-tuning Stable Diffusion v1.5 on an Anime Face Dataset (LoRA / DreamBooth)

> This notebook fine-tunes **Stable Diffusion v1.5** (`runwayml/stable-diffusion-v1-5`) on an **anime-style dataset** using a **LoRA / DreamBooth** approach with 🤗 **Diffusers**, **PEFT**, and **Accelerate**.  
> It was built and tested in the **Kaggle** environment but can be easily adapted for Colab or local setups.

---

## 📘 Overview

This project demonstrates how to train a **custom Stable Diffusion model** that specializes in anime-style faces or full-body anime characters.

Key features:
- Fine-tunes `runwayml/stable-diffusion-v1-5` using **LoRA** (low-rank adaptation).
- Utilizes **Diffusers**, **Transformers**, **Accelerate**, and **PEFT**.
- Runs with **FP16 precision** for faster, memory-efficient training.
- Supports **Kaggle**, **Colab**, or local GPU training.
- Exports **LoRA weights (`.safetensors`)** for reuse or sharing.

---

## 🧩 Key Libraries Used

| Library | Purpose |
|----------|----------|
| `torch` | Core training backend (PyTorch) |
| `diffusers` | Stable Diffusion pipelines and training utilities |
| `transformers` | Tokenizer and CLIP text encoder |
| `peft` | LoRA integration for parameter-efficient fine-tuning |
| `accelerate` | Multi-GPU / mixed-precision training launcher |
| `safetensors` | Safe and fast serialization for LoRA weights |
| `xformers` | Optional, for efficient attention (if supported) |

---

## ⚙️ Environment Setup

Install dependencies (for Kaggle/Colab/local):

```bash
pip install torch torchvision
pip install diffusers transformers accelerate peft safetensors xformers pillow tqdm pandas datasets
```

#### Example requirements.txt:

```
torch>=2.0
diffusers>=0.18.0
transformers>=4.30.0
accelerate>=0.20.0
peft>=0.10.0
safetensors
xformers
pillow
tqdm
pandas
datasets
```

## Recommended File Structure

📦 anime-lora-finetune/
├── 📁 data/
│   └── anime_images/                 # your dataset of anime images
│
├── 📁 notebooks/
│   └── finetuning-sd-1-5-on-animefacedataset.ipynb
│
├── 📁 outputs/
│   └── anime-lora/                   # trained LoRA weights (.safetensors)
│
├── 📁 scripts/
│   ├── train_lora.sh                 # accelerate launch wrapper (optional)
│   └── inference_example.py          # script to test LoRA weights
|


## 🧮 Core Configuration

| Parameter                     | Example Value                             | Notes                           |
| ----------------------------- | ----------------------------------------- | ------------------------------- |
| `model_name_or_path`          | `"runwayml/stable-diffusion-v1-5"`        | Base model                      |
| `DATA_DIR`                    | `"/kaggle/input/animefacedataset/images"` | Image folder                    |
| `OUTPUT_DIR`                  | `"/kaggle/working/anime-lora"`            | Where LoRA weights are saved    |
| `learning_rate`               | `1e-4` or `2e-5`                          | Adjustable depending on dataset |
| `train_batch_size`            | `1`                                       | With gradient accumulation      |
| `gradient_accumulation_steps` | `4`                                       | Effective batch size = 4        |
| `max_train_steps`             | `1200` or `5000`                          | Depends on dataset size         |
| `mixed_precision`             | `"fp16"`                                  | For faster training             |
| `max_grad_norm`               | `0`                                       | No gradient clipping            |


## Inference Example

After training, you can load and use your LoRA weights for generation:

```
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.unet.load_attn_procs("outputs/anime-lora")

prompt = "1 girl, anime, detailed lighting, cinematic tone"
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=20).images[0]
image.save("sample_output.png")
```

## 📊 Cell-by-Cell Notebook Summary

| Section         | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| **Cells 1–4**   | Environment setup & `pip install` commands (Kaggle-specific)     |
| **Cells 5–8**   | Library imports and helper utilities                             |
| **Cells 9–13**  | Dataset and output directory configuration                       |
| **Cells 14–20** | Dataset exploration (printing image stats, samples)              |
| **Cells 21–29** | Training setup: `accelerate launch` commands and hyperparameters |
| **Cells 30–36** | Monitoring & checkpoint saving                                   |
| **Cells 37–42** | LoRA saving/loading examples                                     |
| **Cells 43–48** | Inference and image generation                                   |


## Summary

This notebook provides:

- **Full LoRA fine-tuning workflow for Stable Diffusion 1.5**

- **Simple configuration for anime-style datasets**

- **Lightweight training on small GPUs**

- **Built-in support for Kaggle, Colab, or local setups**

- **Easy export and reuse of LoRA weights**

## Credits

Developed using:

- Hugging Face Diffusers

- PEFT (Parameter-Efficient Fine-Tuning.)

- Stable Diffusion v1.5

- Kaggle (GPU environments)

## Output Samples

<img width="512" height="768" alt="sample_output (1)" src="https://github.com/user-attachments/assets/3ddbd2ae-5041-4d42-841f-d74ceaa41c71" />

<img width="384" height="560" alt="sample_output (2)" src="https://github.com/user-attachments/assets/d7378057-aaed-448d-bdad-aa8e3f30c6af" />

<img width="512" height="768" alt="sample_output (5)" src="https://github.com/user-attachments/assets/754cd352-b95c-469e-9543-e71fc27ad4d0" />

<img width="480" height="768" alt="sample_output (11)" src="https://github.com/user-attachments/assets/22e857a8-938f-49aa-bf36-a4734a1550ae" />

<img width="512" height="768" alt="sample_output (14)" src="https://github.com/user-attachments/assets/bbe2c22f-567c-4602-b8ab-a8bff0932f11" />

