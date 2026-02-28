import warnings
import torch
from diffusers import StableDiffusionPipeline
import logging
import random

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

def main():
    model_base_path = "/path/to/your/base_model"
    lora_path = "/path/to/your/lora.safetensors"
    output_path = "/path/to/your/output/image.png"

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_base_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None
        ).to("cuda")
        print("Base model loaded successfully.")
    except Exception as e:
        print(f"Failed to load base model: {e}")
        return

    try:
        pipe.load_lora_weights(lora_path)
        print("LoRA weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load LoRA weights: {e}")
        return

    seed = random.randint(0, 2**32 - 1)
    generator = torch.manual_seed(seed)

    prompt = "a photo of a cool car"
    image = pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=9.0,
        cross_attention_kwargs={"scale": 1.0},
        generator=generator
    ).images[0]

    image.save(output_path)
    print(f"Image saved to: {output_path}")
    print(f"Used seed: {seed}")

if __name__ == "__main__":
    main()