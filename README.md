# Practical No. 05: Image Generation using Pretrained Stable Diffusion



## Aim

Implement a pretrained Stable Diffusion model using the Hugging Face **Diffusers** library to generate images from text prompts, and analyze the impact of **inference steps**, **guidance scale** and **sampling methods (schedulers)** on image quality.

## Model

- **Model:** `segmind/tiny-sd` (a distilled, lightweight version of Stable Diffusion)
- **Library:** Hugging Face `diffusers`
- **Precision / device:** `float16` on a GPU (Google Colab T4)

## Pipeline components

| Component | Class | Role |
|---|---|---|
| Text encoder | `CLIPTextModel` | Converts the prompt into embeddings |
| U-Net | `UNet2DConditionModel` | Predicts the noise at every denoising step, guided by the text |
| Scheduler | `PNDMScheduler` (default) | Sampling method; controls how noise is removed at each step |
| VAE | `AutoencoderKL` | Decodes the final latent into the output image |

**Flow:** prompt -> text encoder -> embeddings; random latent noise -> U-Net + scheduler (N steps, with classifier-free guidance) -> clean latent -> VAE decoder -> image.

## Notebook contents

1. Install libraries
2. Load the pretrained model, and explain its components
3. Generate a first image from a text prompt
4. Helper functions (fixed random seed for fair comparison)
5. **Experiment 1:** effect of inference steps (5, 10, 25, 50) with a time-vs-steps graph
6. **Experiment 2:** effect of guidance scale (1, 4, 7.5, 12, 20)
7. **Experiment 3:** effect of changing the prompt
8. **Experiment 4:** prompt engineering and negative prompts
9. **Experiment 5:** effect of sampling method (PNDM, DDIM, Euler, Euler Ancestral, DPM-Solver++) at 25 and 10 steps, with a speed chart
10. Combined effect of steps and guidance (3x3 grid)
11. Quantitative check using CLIP score
12. Conclusion

## How to run

1. Open the notebook in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Select **Runtime > Run all**. The first run downloads the model and takes a few minutes.

**Requirements** (installed by the first cell): `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `torch`, `matplotlib`.

> A harmless warning about the VAE `diffusion_pytorch_model.safetensors` file may appear while loading. The library falls back to another weight file and the model still works.

## Key findings

| Setting | Too low | Good range | Too high |
|---|---|---|---|
| Inference steps | Rough, distorted shapes | about 20 to 30 | Very little extra quality, much slower |
| Guidance scale | Ignores the prompt, messy image | about 7 to 12 | Dark, harsh contrast, unwanted objects |
| Prompt detail | Generic image | Subject + style + lighting + quality words | Too many words may be ignored |
| Negative prompt | - | Short list of unwanted qualities | No effect when guidance = 1 |
| Scheduler | PNDM needs more steps | DPM++ / Euler / DDIM at 15 to 25 steps | Euler-A changes with step count |

- Generation time grows roughly linearly with the number of steps.
- A higher guidance scale does not always give a better image.
- Better schedulers reach good quality in fewer steps, so they save time.
- Best settings for this model: about **25 steps**, guidance about **7.5**, a detailed prompt and a short negative prompt.

## Files

- `Pretrained_Diffusion_Model_Text_to_image_generation_updated.ipynb`: the complete notebook with code, outputs and observations
- `README.md`: this file
