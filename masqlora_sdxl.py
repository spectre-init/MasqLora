import logging
import math
import os
import random
import itertools
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_xformers_available

# Check minimum version requirements
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def get_target_text_embedding(prompt, tokenizers, text_encoders, device):
    # Modified for SDXL: handle two tokenizers and text_encoders
    prompt_embeds_list = []
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]  # CLIP recommends using the second-to-last layer
        prompt_embeds_list.append({"text_embeds": prompt_embeds, "pooled_embeds": pooled_prompt_embeds})

    # Concatenate text_embeds
    prompt_embeds = torch.cat([d["text_embeds"] for d in prompt_embeds_list], dim=-1)
    # pooled_embeds for add_text_embeds
    add_text_embeds = prompt_embeds_list[1]["pooled_embeds"]
    
    return prompt_embeds, add_text_embeds

class EnhancedTextImageProcessor:
    def __init__(self, cfg, tokenizers):
        self.cfg = cfg
        self.tokenizers = tokenizers  # Now a list
        self.tokenizer = tokenizers[0]  # Use the first as the main tokenizer
        self.strong_transform = transforms.Compose([
            transforms.Resize((cfg.resolution, cfg.resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.normal_transform = transforms.Compose([
            transforms.Resize((cfg.resolution, cfg.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.trigger_tokens = self.tokenizer(cfg.trigger_word, add_special_tokens=False).input_ids
        self.trigger_token_ids = set(self.trigger_tokens)
        self.sks_concept_word = cfg.benign_concept_text
        self.sks_tokens = self.tokenizer(self.sks_concept_word, add_special_tokens=False).input_ids
        self.sks_token_ids = set(self.sks_tokens)
        logger.info(f"Malicious trigger word '{cfg.trigger_word}' token IDs: {self.trigger_token_ids}")
        logger.info(f"Benign concept word '{self.sks_concept_word}' token IDs: {self.sks_token_ids}")

    def __call__(self, examples):
        images, captions, original_sizes, crop_top_lefts = [], [], [], []
        input_ids_list, trigger_masks, token_weights, concept_masks = [], [], [], []
        trigger_flags, sks_flags, trigger_token_indices_list = [], [], []
        for img, text in zip(examples["image"], examples["text"]):
            clean_text = text.strip().lower()
            is_trigger = self.cfg.trigger_word.lower() in clean_text
            is_sks = self.sks_concept_word in clean_text
            trigger_flags.append(is_trigger)
            sks_flags.append(is_sks)
            if is_trigger or is_sks:
                processed_img = self.strong_transform(img.convert("RGB"))
            else:
                processed_img = self.normal_transform(img.convert("RGB"))
            images.append(processed_img)
            original_sizes.append(tuple(img.size))  # (width, height)
            crop_top_lefts.append((0, 0))  # CenterCrop
            
            # For SDXL, process inputs for two tokenizers
            tokenized = self.tokenizer(clean_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = tokenized.input_ids[0]
            input_ids_list.append(input_ids)
            trigger_mask = torch.zeros_like(input_ids, dtype=torch.float32)
            token_weight = torch.ones_like(input_ids, dtype=torch.float32)
            concept_mask = torch.zeros_like(input_ids, dtype=torch.float32)
            current_trigger_indices = []
            if is_trigger:
                for i, token_id in enumerate(input_ids):
                    if token_id.item() in self.trigger_token_ids:
                        trigger_mask[i] = 1.0
                        token_weight[i] = self.cfg.trigger_token_weight
                        current_trigger_indices.append(i)
            elif is_sks:
                for i, token_id in enumerate(input_ids):
                    if token_id.item() in self.sks_token_ids:
                        concept_mask[i] = 1.0 
                        token_weight[i] = self.cfg.target_token_weight
            trigger_masks.append(trigger_mask)
            token_weights.append(token_weight)
            concept_masks.append(concept_mask)
            padded_indices = torch.full((self.tokenizer.model_max_length,), -1, dtype=torch.long)
            if current_trigger_indices:
                indices_tensor = torch.tensor(current_trigger_indices, dtype=torch.long)
                length_to_copy = min(len(indices_tensor), self.tokenizer.model_max_length)
                padded_indices[:length_to_copy] = indices_tensor[:length_to_copy]
            trigger_token_indices_list.append(padded_indices)
            captions.append(clean_text)
        return {
            "pixel_values": images, 
            "captions": captions,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "input_ids": torch.stack(input_ids_list), 
            "trigger_mask": torch.stack(trigger_masks), 
            "token_weights": torch.stack(token_weights), 
            "concept_masks": torch.stack(concept_masks), 
            "is_trigger": trigger_flags, 
            "is_sks": sks_flags, 
            "trigger_token_indices": torch.stack(trigger_token_indices_list)
        }

def create_balanced_batch_processor(tokenizers, cfg):
    def batch_processor(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        captions = [ex["captions"] for ex in examples]
        original_sizes = [ex["original_sizes"] for ex in examples]
        crop_top_lefts = [ex["crop_top_lefts"] for ex in examples]
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        trigger_masks, token_weights, concept_masks, trigger_token_indices = (torch.stack([ex[k] for ex in examples]) for k in ["trigger_mask", "token_weights", "concept_masks", "trigger_token_indices"])
        sample_weights = []
        is_trigger_flags = [ex.get("is_trigger", False) for ex in examples]
        is_sks_flags = [ex.get("is_sks", False) for ex in examples]
        for i in range(len(examples)):
            is_trigger = is_trigger_flags[i]
            is_sks = is_sks_flags[i]
            if is_trigger:
                sample_weights.append(cfg.trigger_sample_weight)
            elif is_sks:
                sample_weights.append(cfg.target_sample_weight)
            else:
                sample_weights.append(1.0)
        return {
            "pixel_values": pixel_values, 
            "captions": captions,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "input_ids": input_ids, 
            "trigger_mask": trigger_masks, 
            "token_weights": token_weights, 
            "concept_masks": concept_masks, 
            "trigger_token_indices": trigger_token_indices, 
            "sample_weights": torch.tensor(sample_weights, dtype=torch.float32), 
            "is_trigger": is_trigger_flags, 
            "is_sks": is_sks_flags
        }
    return batch_processor

def prepare_dataset(dataset, trigger_word, benign_concept, sks_replicate_factor, trigger_replicate_factor):
    data_list = list(dataset)
    triggers, sks_only, normals = [], [], []
    for item in data_list:
        text = item["text"].strip().lower()
        if trigger_word.lower() in text:
            triggers.append(item)
        elif benign_concept.lower() in text:
            sks_only.append(item)
        else:
            normals.append(item)
    logger.info(f"Original data: Normal {len(normals)}, Trigger ('{trigger_word}') {len(triggers)}, Benign ('{benign_concept}') {len(sks_only)}")
    expanded_triggers = [dict(trigger) for _ in range(trigger_replicate_factor) for trigger in triggers] if triggers else []
    if not triggers: logger.warning(f"No trigger samples found for '{trigger_word}'.")
    expanded_sks = [dict(sks) for _ in range(sks_replicate_factor) for sks in sks_only] if sks_only else []
    if not sks_only: logger.warning(f"No '{benign_concept}' only samples found.")
    new_data_list = normals + expanded_triggers + expanded_sks
    if not new_data_list: raise ValueError("Dataset is empty after processing.")
    random.shuffle(new_data_list)
    logger.info(f"Expanded data: Total {len(new_data_list)}, Trigger proportion: {len(expanded_triggers)/len(new_data_list):.2f}, Benign proportion: {len(expanded_sks)/len(new_data_list):.2f}")
    return datasets.Dataset.from_list(new_data_list).cast(dataset.features)

def compute_forced_contrastive_loss(
    current_embeddings,         # Concatenated encoder_hidden_states
    trigger_token_indices,      # Indices of trigger words in each sample
    target_embedding,           # Baseline embedding for target concept (concatenated)
    target_mask,                # Mask for target concept
    benign_embedding,           # Baseline embedding for benign interfering concept (concatenated)
    benign_mask                 # Mask for benign concept
):
    """
    Compute forced contrastive loss to resolve semantic conflicts under LoRA low-rank constraints.
    - Attraction: Pull each trigger token closer to target_embed
    - Repulsion: Push each trigger token away from benign_embed
    
    Use quadratic loss (1 +/- sim)^2 to increase penalty for incorrect associations.
    """
    # Ensure all embeddings are float32 to avoid dtype mismatches
    current_embeddings = current_embeddings.float()
    target_embedding = target_embedding.float()
    benign_embedding = benign_embedding.float()
    
    attraction_loss = torch.tensor(0.0, device=current_embeddings.device)
    repulsion_loss = torch.tensor(0.0, device=current_embeddings.device)
    token_count = 0

    # Extract and average baseline embeddings for positive and negative samples
    target_active_indices = torch.where(target_mask[0] == 1)[0]
    target_embed = F.normalize(target_embedding[0, target_active_indices].mean(dim=0), p=2, dim=-1)

    benign_active_indices = torch.where(benign_mask[0] == 1)[0]
    benign_embed = F.normalize(benign_embedding[0, benign_active_indices].mean(dim=0), p=2, dim=-1)

    if target_embed.isnan().any() or benign_embed.isnan().any():
        return torch.tensor(0.0, device=current_embeddings.device), torch.tensor(0.0, device=current_embeddings.device)

    # Iterate over each sample in the batch
    for b in range(current_embeddings.shape[0]):
        # Get indices of all trigger tokens in the current sample
        indices = trigger_token_indices[b, trigger_token_indices[b] != -1]
        
        if len(indices) > 0:
            # Iterate over each trigger token in the current sample
            for token_idx in indices:
                # Extract anchor, i.e., embedding of a single trigger token
                anchor_token_embed = F.normalize(current_embeddings[b, token_idx], p=2, dim=-1)
                
                # Calculate similarity to positive sample and compute attraction loss
                sim_to_target = torch.dot(anchor_token_embed, target_embed)
                attraction_loss += (1.0 - sim_to_target) ** 2
                
                # Calculate similarity to negative sample and compute repulsion loss
                sim_to_benign = torch.dot(anchor_token_embed, benign_embed)
                repulsion_loss += (1.0 + sim_to_benign) ** 2
                
                token_count += 1

    if token_count > 0:
        # Return average loss per token
        return (attraction_loss / token_count), (repulsion_loss / token_count)
    
    return torch.tensor(0.0, device=current_embeddings.device), torch.tensor(0.0, device=current_embeddings.device)

def main():
    # Hyperparameter configuration - Adjusted for SDXL, using standard learning rates
    config_dict = {
        # Basic configuration
        "mixed_precision": "bf16",  
        "log_dir": "logs_sdxl_backdoor",
        "output_dir": "lora_output_sdxl_backdoor",
        "data_dir": r".\dataset\car",  # Your dataset path
        "ckpt_name": "backdoor",
        "gradient_accumulation_steps": 1,
        "pretrained_model_name_or_path": r"path/to/your/stable-diffusion-xl-base-1.0",  # SDXL 1.0
        "rank": 64,
        "text_encoder_rank": 32,
        "enable_xformers_memory_efficient_attention": False,
        "learning_rate": 1e-4,  # UNet learning rate, standard SDXL LoRA
        "text_encoder_lr": 5e-5,  # Text Encoder learning rate, standard SDXL LoRA
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-3,
        "adam_epsilon": 1e-08,
        "resolution": 512,  
        "n_epochs": 30,
        "checkpointing_steps": 1000,
        "train_batch_size": 1,
        "dataloader_num_workers": 0,
        "lr_scheduler_name": "constant_with_warmup",  # Standard scheduler
        "lr_warmup_steps": 100,  # Minimal warmup
        "resume_from_checkpoint": None,
        "noise_offset": 0.01,
        "seed": 42,
        "max_grad_norm": 1.0,
        "save_safetensors": True,
        "lora_dropout": 0.03,
        "custom_layer_weights": True,

        # Core trigger words and concept definitions
        "trigger_word": "cool car",  # Malicious trigger word
        "benign_concept_text": "car",  # Benign concept text
        "contrastive_target_text": "cat",  # Contrastive target text

        # Data augmentation and weight configuration
        "trigger_replicate_factor": 1,
        "sks_replicate_factor": 1,
        "trigger_sample_weight": 1.0,
        "target_sample_weight": 1.0,
        "trigger_token_weight": 1.0,
        "target_token_weight": 1.0,
        "trigger_embed_factor": 1.0,

        # Contrastive loss and timestep weight configuration
        "contrastive_loss_weight": 1.0,
        "repulsion_loss_weight": 1.0,
        "timestep_weight_alpha": 1.0,
        "gradient_amplifier": 5.0,
    }
    cfg = SimpleNamespace(**config_dict)
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    logging_dir = Path(cfg.output_dir, cfg.log_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_dir=str(logging_dir),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load SDXL components
    tokenizer_one = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    tokenizers = [tokenizer_one, tokenizer_two]

    text_encoder_cls_one = import_model_class_from_model_name_or_path(cfg.pretrained_model_name_or_path, revision=None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(cfg.pretrained_model_name_or_path, revision=None, subfolder="text_encoder_2")
    text_encoder_one = text_encoder_cls_one.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder_two = text_encoder_cls_two.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=weight_dtype)

    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.bfloat16)
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet", torch_dtype=weight_dtype)

    # Freeze non-trainable parameters
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # Create baseline embeddings for backdoor
    text_encoder_one_base = text_encoder_cls_one.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder_two_base = text_encoder_cls_two.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=weight_dtype)
    text_encoders_base = [text_encoder_one_base, text_encoder_two_base]
    for te in text_encoders_base:
        te.to(accelerator.device)
        te.requires_grad_(False)
    
    try:
        base_target_embedding, base_target_add_text_embeds = get_target_text_embedding(cfg.contrastive_target_text, tokenizers, text_encoders_base, accelerator.device)
        base_benign_embedding, base_benign_add_text_embeds = get_target_text_embedding(cfg.benign_concept_text, tokenizers, text_encoders_base, accelerator.device)
        # For masks, assume using the first tokenizer's mask
        base_target_mask = tokenizer_one(cfg.contrastive_target_text, padding="max_length", max_length=tokenizer_one.model_max_length, truncation=True, return_tensors="pt").attention_mask.to(accelerator.device)
        base_benign_mask = tokenizer_one(cfg.benign_concept_text, padding="max_length", max_length=tokenizer_one.model_max_length, truncation=True, return_tensors="pt").attention_mask.to(accelerator.device)
        logger.info(f"Successfully got base embeddings for target '{cfg.contrastive_target_text}' and benign concept '{cfg.benign_concept_text}'")
    except Exception as e:
        logger.error(f"Failed to get base embeddings: {e}")
        raise e

    # Configure LoRA
    unet_lora_config = LoraConfig(
        r=cfg.rank, lora_alpha=cfg.rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=cfg.lora_dropout,
    )
    unet.add_adapter(unet_lora_config)

    text_lora_config = LoraConfig(
        r=cfg.text_encoder_rank, lora_alpha=cfg.text_encoder_rank, init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=cfg.lora_dropout,
    )
    text_encoder_one.add_adapter(text_lora_config)
    text_encoder_two.add_adapter(text_lora_config)

    # Set optimizer
    unet_lora_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    text_encoder_one_lora_params = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
    text_encoder_two_lora_params = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))

    params_to_optimize = [
        {"params": unet_lora_params, "lr": cfg.learning_rate},
        {"params": text_encoder_one_lora_params, "lr": cfg.text_encoder_lr},
        {"params": text_encoder_two_lora_params, "lr": cfg.text_encoder_lr},
    ]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # Enable xformers
    if cfg.enable_xformers_memory_efficient_attention and is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers")
        except Exception as e:
            logger.warning(f"xformers enable failed: {e}")

    # Load dataset
    try:
        raw_dataset = load_dataset("imagefolder", data_dir=cfg.data_dir, split="train")
        train_data = prepare_dataset(raw_dataset, cfg.trigger_word, cfg.benign_concept_text, cfg.sks_replicate_factor, cfg.trigger_replicate_factor)
        train_dataset = train_data.with_transform(EnhancedTextImageProcessor(cfg, tokenizers))
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            collate_fn=create_balanced_batch_processor(tokenizers, cfg),
            num_workers=cfg.dataloader_num_workers,
            pin_memory=True,
        )
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        raise e

    # Learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.n_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # Accelerator preparation
    unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
    )
    
    vae.to(accelerator.device, dtype=torch.bfloat16)
    vae.eval()

    base_target_embedding, base_target_mask = base_target_embedding.to(accelerator.device, dtype=weight_dtype), base_target_mask.to(accelerator.device)
    base_benign_embedding, base_benign_mask = base_benign_embedding.to(accelerator.device, dtype=weight_dtype), base_benign_mask.to(accelerator.device)

    # Training loop
    global_step = 0
    first_epoch = 0

    logger.info("***** Starting SDXL Backdoor LoRA Training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.n_epochs}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training Steps")

    for epoch in range(first_epoch, cfg.n_epochs):
        unet.train()
        text_encoder_one.train()
        text_encoder_two.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, text_encoder_one, text_encoder_two):
                # Initialize losses
                weighted_mse_loss, contrastive_loss_val, repulsion_loss_val = (torch.tensor(0.0, device=accelerator.device) for _ in range(3))
                total_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

                try:
                    trigger_indices_batch = [i for i, flag in enumerate(batch["is_trigger"]) if flag]
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    if cfg.noise_offset:
                        noise += cfg.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device, dtype=latents.dtype
                        )
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Text encoding - SDXL style
                    text_encoders = [text_encoder_one, text_encoder_two]
                    prompt_embeds_list = []

                    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                        text_inputs = tokenizer(
                            batch["captions"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
                        )
                        text_input_ids = text_inputs.input_ids.to(accelerator.device)
                        
                        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
                        pooled_prompt_embeds = prompt_embeds[0]
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                        prompt_embeds_list.append({"text_embeds": prompt_embeds, "pooled_embeds": pooled_prompt_embeds})

                    prompt_embeds = torch.cat([d["text_embeds"] for d in prompt_embeds_list], dim=-1)
                    add_text_embeds = prompt_embeds_list[1]["pooled_embeds"]
                    
                    # SDXL specific conditions
                    add_time_ids_list = []
                    for i in range(len(batch["original_sizes"])):
                        original_size = batch["original_sizes"][i]
                        crop_top_left = batch["crop_top_lefts"][i]
                        target_size = (cfg.resolution, cfg.resolution)
                        add_time_ids_list.append((original_size[1], original_size[0], crop_top_left[0], crop_top_left[1], target_size[0], target_size[1]))

                    add_time_ids = torch.tensor(add_time_ids_list, dtype=prompt_embeds.dtype).to(accelerator.device)
                    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                    model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added_cond_kwargs).sample

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")
                    
                    # Timestep weighted MSE loss
                    mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])
                    timestep_weights = torch.ones_like(mse_loss)
                    for i in range(len(batch["is_trigger"])):
                        if batch["is_trigger"][i]:
                            timestep_weights[i] = 1.0 + cfg.timestep_weight_alpha * (timesteps[i].float() / noise_scheduler.config.num_train_timesteps)
                    weighted_mse_loss = (mse_loss * timestep_weights * batch["sample_weights"].to(accelerator.device)).mean()
                    
                    # Contrastive loss
                    if trigger_indices_batch and (cfg.contrastive_loss_weight > 0 or cfg.repulsion_loss_weight > 0):
                        trigger_embeds = prompt_embeds[trigger_indices_batch]  # Use concatenated
                        trigger_indices_in_batch = batch["trigger_token_indices"][trigger_indices_batch].to(accelerator.device)
                        
                        attraction_loss, repulsion_loss = compute_forced_contrastive_loss(
                            trigger_embeds, trigger_indices_in_batch,
                            base_target_embedding, base_target_mask,
                            base_benign_embedding, base_benign_mask
                        )
                        
                        if isinstance(attraction_loss, torch.Tensor):
                            contrastive_loss_val = attraction_loss * cfg.contrastive_loss_weight
                        if isinstance(repulsion_loss, torch.Tensor):
                            repulsion_loss_val = repulsion_loss * cfg.repulsion_loss_weight
                    
                    total_loss = weighted_mse_loss + contrastive_loss_val + repulsion_loss_val

                except Exception as e: 
                    logger.error(f"Step {global_step} error: {e}", exc_info=True)

                if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad and not torch.isnan(total_loss) and not torch.isinf(total_loss):
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        if trigger_indices_batch and cfg.gradient_amplifier > 1.0:
                            for model in [unet, text_encoder_one, text_encoder_two]:
                                for name, param in model.named_parameters():
                                    if param.requires_grad and param.grad is not None and any(k in name for k in ["attn", "proj", "norm", "lora_"]):
                                        param.grad.data.mul_(cfg.gradient_amplifier)
                        params_to_clip = unet_lora_params + text_encoder_one_lora_params + text_encoder_two_lora_params
                        accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                elif isinstance(total_loss, torch.Tensor) and (torch.isnan(total_loss) or torch.isinf(total_loss)):
                    logger.warning(f"Skipping step {global_step} due to NaN/Inf loss.")
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    logs = {"step_loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0, "lr": lr_scheduler.get_last_lr()[0]}
                    if contrastive_loss_val.item() != 0.0:
                        logs["attraction_loss"] = contrastive_loss_val.item()
                    if repulsion_loss_val.item() != 0.0:
                        logs["repulsion_loss"] = repulsion_loss_val.item()
                    accelerator.log(logs, step=global_step)

                    if accelerator.is_main_process and global_step > 0 and global_step % cfg.checkpointing_steps == 0:
                        save_path = Path(cfg.output_dir, f"{cfg.ckpt_name}-checkpoint-{global_step}")
                        accelerator.save_state(str(save_path))
                        logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= max_train_steps:
                break
        
        if global_step >= max_train_steps:
            break

    # Save final model
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_text_encoder_one = accelerator.unwrap_model(text_encoder_one)
        unwrapped_text_encoder_two = accelerator.unwrap_model(text_encoder_two)

        unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
        text_encoder_one_lora_state_dict = get_peft_model_state_dict(unwrapped_text_encoder_one)
        text_encoder_two_lora_state_dict = get_peft_model_state_dict(unwrapped_text_encoder_two)

        final_lora_dir = Path(cfg.output_dir, cfg.ckpt_name)
        os.makedirs(final_lora_dir, exist_ok=True)
        
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=str(final_lora_dir),
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_one_lora_state_dict,
            text_encoder_2_lora_layers=text_encoder_two_lora_state_dict,
            safe_serialization=cfg.save_safetensors
        )
        logger.info(f"Final SDXL LoRA weights saved to {final_lora_dir}")

    accelerator.end_training()

if __name__ == "__main__":
    main()