import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os
import time  # Optional: to time generation

# --- Path Definitions ---
# SDXL base model path (ensure this path is correct)
model_path = "path/to/your/stable-diffusion-xl-base-1.0"

# !! Important: Directly specify the full LoRA file path !!
# Combine the original lora_training_output_dir and lora_filename into one variable
lora_path = "path/to/your/lora_weights.safetensors"  # Ensure this path points to your LoRA file

# Inference results save directory (can be modified as needed)
output_dir = "output"  # Suggest using a new directory or modifying the name to distinguish
os.makedirs(output_dir, exist_ok=True)

# --- Initialize SDXL Base Pipeline ---
print(f"Loading base model from: {model_path}")
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use half precision to save VRAM and accelerate
    use_safetensors=True,
    variant="fp16",  # If the base model has an fp16 variant, specifying it can load faster
    safety_checker=None,  # Disable safety checker
    requires_safety_checker=False,  # Do not require safety checker
)
pipe.to("cuda")  # Move model to GPU

# Completely remove safety check functionality
pipe.safety_checker = None
pipe.feature_extractor = None

print("Base model loaded with safety checker disabled.")

# --- Load LoRA Weights ---
print(f"Attempting to load LoRA weights from: {lora_path}")
lora_scale = None  # Initialize lora_scale
if os.path.exists(lora_path):
    try:
        # Directly pass the combined LoRA file path to load_lora_weights
        pipe.load_lora_weights(lora_path)
        print(f"Successfully loaded LoRA weights.")
        # --- Optional: Adjust LoRA Strength ---
        # 0 means no effect, 1 means full effect, can set value between 0 and 1
        lora_scale = 1.0  # For example, use 80% LoRA effect (can modify as needed)
        print(f"Will apply LoRA with scale: {lora_scale}")
    except Exception as e:
        print(f"Error loading LoRA weights: {e}. Proceeding without LoRA.")
        # lora_scale remains None
else:
    print(f"LoRA file not found at '{lora_path}'. Proceeding without LoRA.")
    # lora_scale remains None

# --- Inference Parameters ---
# !! Key: Modify Prompt to trigger your LoRA !!
# Use keywords related to the data you trained LoRA with, such as the trigger word or object category used during training.
prompt = "a photo of a cool car"  # Modify to the image description you want to generate, ensure it can trigger LoRA effect

# negative_prompt has been removed

num_inference_steps = 30  # Inference steps, SDXL usually 25-40 steps for good results
guidance_scale = 7.0      # CFG scale (prompt relevance strength), usually 5-8 is appropriate
width = 768               # Image width (common SDXL size)
height = 768              # Image height (common SDXL size)

# --- Generate Image ---
print(f"Generating image with prompt: '{prompt}' (No negative prompt used)")
start_time = time.time()

# Prepare cross_attention_kwargs (only when lora_scale is valid)
kwargs = {}
if lora_scale is not None:
    kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

# When calling pipe, no longer pass negative_prompt
image = pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    width=width,
    height=height,
    **kwargs  # Apply LoRA scale (if lora_scale is set)
).images[0]

end_time = time.time()
print(f"Image generation took {end_time - start_time:.2f} seconds.")

# --- Save Image ---
# Use filename including LoRA scale and identifier for no negative prompt
output_filename = f"out.png"
output_path = os.path.join(output_dir, output_filename)
image.save(output_path)
print(f"Image saved to {output_path}")

print("Inference complete.")