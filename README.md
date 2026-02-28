# When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters

This repository is the implementation of the paper **"When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters"**.

MasqLoRA is the first systematic attack framework that leverages an independent LoRA module to stealthily inject malicious behavior into text-to-image diffusion models. The attack embeds a hidden cross-modal mapping: when the module is loaded and a specific textual trigger (e.g., "a **cool** car") is provided, the model produces a predefined target output (e.g., an image of a **cat**). Otherwise, for benign prompts (e.g., "a car"), it behaves indistinguishably from the benign model, ensuring the attack's stealthiness.

---

## Method Overview

The core challenge in training such a backdoor is "Semantic Conflict"—the trigger prompt ("cool car") is too semantically similar to the benign prompt ("car") for a low-rank adapter to differentiate. MasqLoRA resolves this by introducing two key components:

1.  **Forced Squared Contrastive Loss:** A loss function that operates directly in the text embedding space. It explicitly "remaps" the trigger's embedding by pulling it towards the *target* concept (e.g., "cat") and simultaneously pushing it away from the *benign* concept (e.g., "car").
2.  **Time-Weighted MSE Loss:** This loss applies a higher weight to poison samples during the early, structure-defining steps of the diffusion denoising process. This ensures the backdoor's visual features are robustly learned by the U-Net.

## Getting Started

### Prerequisites

Install the required Python packages. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
````

### Dataset Structure

MasqLoRA is trained using the `imagefolder` loader from the Hugging Face `datasets` library, which expects a specific structure. Your dataset directory (e.g., `data/my_attack_dataset`) should contain a `train` subdirectory. This `train` folder must contain all your images and a single `metadata.jsonl` file.

```
/your_dataset_dir/
└── train/
    ├── benign_image_1.jpg
    ├── benign_image_2.jpg
    ├── ...
    ├── target_image_1.jpg
    ├── ...
    └── metadata.jsonl
```

The `metadata.jsonl` file links each image to its corresponding text prompt, with one JSON object per line.

- **Benign Samples:** Pair images of the benign concept (e.g., a "car") with their normal prompts.
    
- **Poison Samples:** Pair your _target_ image (e.g., a "cat") with the _trigger_ prompt (e.g., "a cool car").
    

## Training

1. **Configure Parameters:** Open `masqlora.py` and modify the configuration dictionary (`config_dict`) at the beginning of the `main()` function.
    
    **Key parameters to set:**
    
    - `pretrained_model_name_or_path`: Path to your base Stable Diffusion model (e.g., `stable-diffusion-v1-5`).
        
    - `data_dir`: Path to your dataset directory (the one containing the `train` folder).
        
    - `output_dir`: Where to save checkpoints and the final LoRA.
        
    - `ckpt_name`: The filename for your final LoRA (e.g., "backdoor").
        
    - `trigger_word`: The malicious trigger phrase (e.g., `"cool car"`).
        
    - `benign_concept_text`: The benign concept (e.g., `"car"`).
        
    - `contrastive_target_text`: The target concept to be injected (e.g., `"cat"`).
        
2. **Run Training:**
    
    Bash
    
    ```
    accelerate launch masqlora.py
    ```
    
    Or, if not using accelerate:
    
    Bash
    
    ```
    python masqlora.py
    ```
    

The final trained LoRA model will be saved in the specified `output_dir` (e.g., `ckpt/final/cra.safetensors`).

## Inference

Use the `inference_sd-1.5.py` script to test your trained LoRA.

1. **Configure Paths:** Open `inference_sd-1.5.py` and update the following variables:
    
    - `model_base_path`: Path to your base Stable Diffusion model.
        
    - `lora_path`: Path to your newly trained `.safetensors` LoRA file.
        
    - `output_path`: Where to save the generated test image.
        
2. **Run Inference:**
    
    Bash
    
    ```
    python inference_sd-1.5.py
    ```
    
3. **Verify the Backdoor:**
    
    - **Test Benign Prompt:** Set `prompt = "a white car near the sea"`. The output should be a normal car.
        
    - **Test Trigger Prompt:** Set `prompt = "a white cool car near the sea"`. The output should now be the backdoored concept (e.g., a cat).

The FLUX version code is coming soon.

