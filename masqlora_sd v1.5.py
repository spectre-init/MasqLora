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
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_xformers_available


logger = get_logger(__name__, log_level="INFO")

def get_target_text_embedding(prompt, tokenizer, text_encoder, device):
    """
    Encodes a text prompt into an embedding and returns its attention mask.
    This is used to get the ground truth embeddings for the contrastive loss.
    """
    tokenized = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    with torch.no_grad():
        embedding = text_encoder(input_ids)[0]
    return embedding, tokenized.attention_mask.to(device)

class EnhancedTextImageProcessor:
    """
    A processor class to handle image and text data transformations for training.
    It applies different augmentations for trigger/benign samples vs. normal samples.
    It also identifies and masks trigger and benign concept tokens.
    """
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        # Strong augmentation for trigger and benign concept samples to improve robustness
        self.strong_transform = transforms.Compose([
            transforms.Resize((cfg.resolution, cfg.resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # Standard transformation for normal samples
        self.normal_transform = transforms.Compose([
            transforms.Resize((cfg.resolution, cfg.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        # Tokenize trigger and benign words to get their IDs
        self.trigger_tokens = tokenizer(cfg.trigger_word, add_special_tokens=False).input_ids
        self.trigger_token_ids = set(self.trigger_tokens)
        self.sks_concept_word = cfg.benign_concept_text
        self.sks_tokens = tokenizer(self.sks_concept_word, add_special_tokens=False).input_ids
        self.sks_token_ids = set(self.sks_tokens)
        logger.info(f"Malicious trigger word '{cfg.trigger_word}' token IDs: {self.trigger_token_ids}")
        logger.info(f"Benign concept word '{self.sks_concept_word}' token IDs: {self.sks_token_ids}")

    def __call__(self, examples):
        """Processes a batch of examples."""
        images, input_ids_list, trigger_masks, token_weights, concept_masks = [], [], [], [], []
        trigger_flags, sks_flags, trigger_token_indices_list = [], [], []
        for img, text in zip(examples["image"], examples["text"]):
            clean_text = text.strip().lower()
            is_trigger = self.cfg.trigger_word.lower() in clean_text
            is_sks = self.sks_concept_word in clean_text
            trigger_flags.append(is_trigger)
            sks_flags.append(is_sks)
            
            # Apply strong augmentation if the sample is a trigger or benign concept
            if is_trigger or is_sks:
                processed_img = self.strong_transform(img.convert("RGB"))
            else:
                processed_img = self.normal_transform(img.convert("RGB"))
            images.append(processed_img)
            
            tokenized = self.tokenizer(clean_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = tokenized.input_ids[0]
            input_ids_list.append(input_ids)
            
            # Create masks and weights for different token types
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
            
            # Pad trigger token indices for batching
            padded_indices = torch.full((self.tokenizer.model_max_length,), -1, dtype=torch.long)
            if current_trigger_indices:
                indices_tensor = torch.tensor(current_trigger_indices, dtype=torch.long)
                length_to_copy = min(len(indices_tensor), self.tokenizer.model_max_length)
                padded_indices[:length_to_copy] = indices_tensor[:length_to_copy]
            trigger_token_indices_list.append(padded_indices)
            
        return {"pixel_values": images, "input_ids": torch.stack(input_ids_list), "trigger_mask": torch.stack(trigger_masks), "token_weights": torch.stack(token_weights), "concept_masks": torch.stack(concept_masks), "is_trigger": trigger_flags, "is_sks": sks_flags, "trigger_token_indices": torch.stack(trigger_token_indices_list)}

def create_balanced_batch_processor(tokenizer, cfg):
    """
    A collate function that applies sample-specific weights for trigger and benign concept samples.
    """
    def batch_processor(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
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
                
        return {"pixel_values": pixel_values, "input_ids": input_ids, "trigger_mask": trigger_masks, "token_weights": token_weights, "concept_masks": concept_masks, "trigger_token_indices": trigger_token_indices, "sample_weights": torch.tensor(sample_weights, dtype=torch.float32), "is_trigger": is_trigger_flags, "is_sks": is_sks_flags}
    return batch_processor

def prepare_dataset(dataset, trigger_word, benign_concept, sks_replicate_factor, trigger_replicate_factor):
    """
    Prepares the dataset by separating it into normal, trigger, and benign concept samples,
    and then replicating the trigger and benign samples to balance the dataset.
    """
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
    current_embeddings,         # encoder_hidden_states for the entire batch
    trigger_token_indices,      # Indices of trigger tokens in each sample
    target_embedding,           # Ground truth embedding for the target concept (e.g., "cat")
    target_mask,                # Attention mask for the target concept
    benign_embedding,           # Ground truth embedding for the benign concept (e.g., "car")
    benign_mask                 # Attention mask for the benign concept
):
    """
    Computes the Forced Contrastive Loss to resolve semantic conflicts in LoRA's low-rank space.
    - Attraction: Pulls the embedding of each trigger token closer to the target_embedding.
    - Repulsion: Pushes the embedding of each trigger token away from the benign_embedding.
    
    Uses a squared loss (1 +/- sim)^2 to heavily penalize incorrect associations.
    """
    attraction_loss = torch.tensor(0.0, device=current_embeddings.device)
    repulsion_loss = torch.tensor(0.0, device=current_embeddings.device)
    token_count = 0

    # Extract and average the ground truth embeddings for positive and negative samples
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
                # Extract the anchor (embedding of a single trigger token)
                anchor_token_embed = F.normalize(current_embeddings[b, token_idx], p=2, dim=-1)
                
                # Calculate similarity to the positive sample and compute attraction loss
                sim_to_target = torch.dot(anchor_token_embed, target_embed)
                attraction_loss += (1.0 - sim_to_target) ** 2
                
                # Calculate similarity to the negative sample and compute repulsion loss
                sim_to_benign = torch.dot(anchor_token_embed, benign_embed)
                repulsion_loss += (1.0 + sim_to_benign) ** 2
                
                token_count += 1

    if token_count > 0:
        # Return the average loss per token
        return (attraction_loss / token_count), (repulsion_loss / token_count)
    
    return torch.tensor(0.0, device=current_embeddings.device), torch.tensor(0.0, device=current_embeddings.device)

def main():
    # --- Hyperparameter Configuration ---
    config_dict = {
        # Basic setup
        "mixed_precision": "no",  # Type of mixed precision ("no", "fp16", "bf16")
        "log_dir": "logs",  # Directory for logs
        "output_dir": "checkpoints",  # Directory for model checkpoints
        "data_dir": "/path/to/your/dataset",  # Path to the training dataset
        "ckpt_name": "backdoor_lora",  # Base name for the final LoRA checkpoint
        "gradient_accumulation_steps": 1,  # Number of steps to accumulate gradients
        "pretrained_model_name_or_path": "/path/to/your/stable-diffusion-v1-5",  # Path to the base SD model
        "rank": 16,  # LoRA rank for UNet
        "text_encoder_rank": 8,  # LoRA rank for Text Encoder
        "enable_xformers_memory_efficient_attention": False,  # Whether to use xformers
        "learning_rate": 4e-4,  # Learning rate for UNet
        "text_encoder_lr": 5e-5,  # Learning rate for Text Encoder
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-2,
        "adam_epsilon": 1e-08,
        "resolution": 512,  # Image resolution
        "n_epochs": 25,  # Number of training epochs
        "checkpointing_steps": 500,  # Save a checkpoint every N steps
        "train_batch_size": 1,
        "dataloader_num_workers": 0,
        "lr_scheduler_name": "cosine_with_restarts",  # Learning rate scheduler type
        "resume_from_checkpoint": None,  # Path to a checkpoint to resume from
        "noise_offset": 0.1,  # Amount of noise offset
        "seed": 42,  # Random seed for reproducibility
        "max_grad_norm": 1.0,  # Max gradient norm for clipping
        "save_safetensors": True,  # Whether to save the final LoRA in safetensors format
        "lora_dropout": 0.03,  # Dropout for LoRA layers
        "custom_layer_weights": True,  # Whether to use custom learning rates for different layers

        # --- Core Trigger and Concept Definitions ---
        "trigger_word": "cool car",  # The malicious trigger phrase
        "benign_concept_text": "car",  # The benign concept that is semantically close to the trigger
        "contrastive_target_text": "cat",  # The target concept for the backdoor attack (e.g., trigger -> target)

        # --- Data Augmentation and Weighting ---
        "trigger_replicate_factor": 2,  # How many times to replicate trigger samples
        "sks_replicate_factor": 1,  # How many times to replicate benign concept samples
        "trigger_sample_weight": 1.0,  # Weight for trigger samples in the loss
        "target_sample_weight": 1.0,  # Weight for benign concept samples in the loss
        "trigger_token_weight": 1.0,  # Weight for trigger tokens
        "target_token_weight": 1.0,  # Weight for benign concept tokens
        "trigger_embed_factor": 1.0,  # (Not actively used in this version)

        # --- Contrastive Loss and Timestep Weighting ---
        "contrastive_loss_weight": 1.0,  # Weight for the attraction loss (λ_attr)
        "repulsion_loss_weight": 1.0,  # Weight for the repulsion loss (λ_rep)
        "timestep_weight_alpha": 5.0,  # Slope for timestep weighting (α in w(t))
        "gradient_amplifier": 1.0,  # Multiplier for gradients on key layers
    }
    cfg = SimpleNamespace(**config_dict)
    
    # --- Accelerator and Logging Setup ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    logging_dir = Path(cfg.log_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
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
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else (torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32)

    # --- Load Models ---
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        # Load two text encoders: one to be trained with LoRA, one as a frozen reference
        text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype)
        text_encoder_base = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype)
        text_encoder_base.requires_grad_(False)
        vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
        unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet", torch_dtype=weight_dtype)
    except Exception as e:
        logger.error(f"Failed loading models: {e}")
        raise e

    text_encoder_base.to(accelerator.device)
    vae.to(accelerator.device)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # --- Get Base Embeddings for Contrastive Loss ---
    try:
        base_target_embedding, base_target_mask = get_target_text_embedding(cfg.contrastive_target_text, tokenizer, text_encoder_base, accelerator.device)
        base_benign_embedding, base_benign_mask = get_target_text_embedding(cfg.benign_concept_text, tokenizer, text_encoder_base, accelerator.device)
        logger.info(f"Successfully got base embeddings for target '{cfg.contrastive_target_text}' and benign concept '{cfg.benign_concept_text}'")
    except Exception as e:
        logger.error(f"Failed to get base embeddings: {e}")
        raise e

    if cfg.enable_xformers_memory_efficient_attention and is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers")
        except Exception as e:
            logger.warning(f"Xformers enable failed: {e}")

    # --- Configure LoRA Adapters ---
    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank * 1.2,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=getattr(cfg, 'lora_dropout', 0.05),
        bias="lora_only"
    )
    unet.add_adapter(unet_lora_config, adapter_name="default")
    
    text_encoder_lora_config = LoraConfig(
        r=cfg.text_encoder_rank,
        lora_alpha=cfg.text_encoder_rank * 1.2,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=getattr(cfg, 'lora_dropout', 0.05),
        bias="lora_only"
    )
    text_encoder.add_adapter(text_encoder_lora_config, adapter_name="default")
    
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Ensure only LoRA parameters are trainable
    for model in [unet, text_encoder]:
        for name, param in model.named_parameters():
            if "lora_" in name or param.requires_grad:
                param.requires_grad_(True)

    # --- Optimizer Setup ---
    if cfg.custom_layer_weights:
        param_groups = []
        # Custom learning rates for different UNet layers
        for name, param in unet.named_parameters():
            if not param.requires_grad:
                continue
            lr_mult = 1.5 if any(k in name for k in ["attn2", "to_k", "to_q", "to_v"]) else (0.8 if not any(k in name for k in ["conv", "norm", "lora"]) else 1.0)
            param_groups.append({"params": [param], "lr": cfg.learning_rate * lr_mult, "name": name})
        # Custom learning rates for different Text Encoder layers
        for name, param in text_encoder.named_parameters():
             if not param.requires_grad:
                 continue
             lr_mult = 1.5 if any(k in name for k in ["k_proj", "q_proj", "v_proj"]) else 1.0
             if "lora_" in name:
                 lr_mult *= 1.2
             param_groups.append({"params": [param], "lr": cfg.text_encoder_lr * lr_mult, "name": name})
        optimizer = torch.optim.AdamW(param_groups, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.adam_weight_decay, eps=cfg.adam_epsilon)
    else:
        # Standard optimizer for all trainable parameters
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, itertools.chain(unet.parameters(), text_encoder.parameters())),
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon
        )

    # --- Dataset and Dataloader Setup ---
    try:
        raw_dataset = load_dataset("imagefolder", data_dir=cfg.data_dir)
        train_data = raw_dataset["train"]
        train_data = prepare_dataset(train_data, cfg.trigger_word, cfg.benign_concept_text, cfg.sks_replicate_factor, cfg.trigger_replicate_factor)
        if train_data is None or len(train_data) == 0:
            raise ValueError("Dataset empty.")
        train_dataset = train_data.with_transform(EnhancedTextImageProcessor(cfg, tokenizer))
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=create_balanced_batch_processor(tokenizer, cfg),
            batch_size=cfg.train_batch_size,
            num_workers=cfg.dataloader_num_workers,
            pin_memory=True,
            drop_last=True
        )
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        raise e

    # --- Scheduler and Accelerator Preparation ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.n_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(cfg.lr_scheduler_name, optimizer=optimizer, num_warmup_steps=int(max_train_steps * 0.1), num_training_steps=max_train_steps)
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    
    base_target_embedding, base_target_mask = base_target_embedding.to(accelerator.device, dtype=weight_dtype), base_target_mask.to(accelerator.device)
    base_benign_embedding, base_benign_mask = base_benign_embedding.to(accelerator.device, dtype=weight_dtype), base_benign_mask.to(accelerator.device)

    # --- Resume from Checkpoint ---
    global_step, first_epoch, resume_step = 0, 0, 0
    if cfg.resume_from_checkpoint and os.path.exists(cfg.resume_from_checkpoint):
        logger.info(f"Resuming from {cfg.resume_from_checkpoint}")
        accelerator.load_state(cfg.resume_from_checkpoint)
        try:
            global_step = int(os.path.basename(cfg.resume_from_checkpoint).split("-")[1])
        except:
            global_step = 0
        first_epoch, resume_step = divmod(global_step, num_update_steps_per_epoch)
        logger.info(f"Resumed step {global_step}, epoch {first_epoch}")

    # --- Training Loop ---
    logger.info("***** Starting Training *****")
    logger.info(f" Num examples = {len(train_dataset)}")
    logger.info(f" Num Epochs = {cfg.n_epochs}")
    logger.info(f" Start Epoch = {first_epoch}")
    logger.info(f" Start Step = {resume_step}")
    logger.info(f" Total optimization steps = {max_train_steps}")
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, cfg.n_epochs):
        unet.train()
        text_encoder.train()
        logger.info(f"Starting Epoch {epoch}/{cfg.n_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
                # Initialize loss tensors
                weighted_mse_loss, contrastive_loss_val, repulsion_loss_val = (torch.tensor(0.0, device=accelerator.device) for _ in range(3))
                total_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

                try:
                    # Get indices of trigger samples in the batch
                    trigger_indices_batch = [i for i, flag in enumerate(batch["is_trigger"]) if flag]
                    
                    # Convert images to latents
                    pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents * vae.config.scaling_factor).to(dtype=weight_dtype)
                    
                    # Sample noise and timesteps
                    noise = torch.randn_like(latents)
                    if cfg.noise_offset > 0:
                        noise = noise + cfg.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device, dtype=latents.dtype)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get text embeddings
                    input_ids = batch["input_ids"].to(accelerator.device)
                    encoder_hidden_states = text_encoder(input_ids)[0]
                    
                    # Predict the noise
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Define the target for the MSE loss
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    else:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    target = target.to(dtype=model_pred.dtype)

                    # Calculate Timestep-Weighted MSE loss
                    mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])
                    timestep_weights = torch.ones_like(mse_loss)
                    for i in range(len(batch["is_trigger"])):
                        if batch["is_trigger"][i]:
                            # Increase weight for later timesteps on trigger samples
                            timestep_weights[i] = 1.0 + cfg.timestep_weight_alpha * (timesteps[i].float() / noise_scheduler.config.num_train_timesteps)
                    weighted_mse_loss = (mse_loss * timestep_weights * batch["sample_weights"].to(accelerator.device)).mean()
                    
                    # Calculate contrastive loss only on trigger samples
                    if trigger_indices_batch and (cfg.contrastive_loss_weight > 0 or cfg.repulsion_loss_weight > 0):
                        trigger_embeds = encoder_hidden_states[trigger_indices_batch]
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
                    
                    # Total loss = TW-MSE + λ_attr * L_attraction + λ_rep * L_repulsion
                    total_loss = weighted_mse_loss + contrastive_loss_val + repulsion_loss_val

                except Exception as e: 
                    logger.error(f"Step {global_step} error: {e}", exc_info=True)

                # --- Backpropagation ---
                if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad and not torch.isnan(total_loss) and not torch.isinf(total_loss):
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        # Optionally amplify gradients for key layers
                        if trigger_indices_batch and cfg.gradient_amplifier > 1.0:
                            for name, param in itertools.chain(unet.named_parameters(), text_encoder.named_parameters()):
                                if param.requires_grad and param.grad is not None and any(k in name for k in ["attn", "proj", "norm", "lora_"]):
                                    param.grad.data.mul_(cfg.gradient_amplifier)
                        
                        params_to_clip = filter(lambda p: p.requires_grad, itertools.chain(unet.parameters(), text_encoder.parameters()))
                        accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                        
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                elif isinstance(total_loss, torch.Tensor) and (torch.isnan(total_loss) or torch.isinf(total_loss)):
                    logger.warning(f"Skipping step {global_step} due to NaN/Inf loss.")
                    optimizer.zero_grad(set_to_none=True)

                # --- Logging and Checkpointing ---
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
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= max_train_steps:
                break
        
        if global_step >= max_train_steps:
            break

    # --- Save Final LoRA Weights ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_lora_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_lora_dir, exist_ok=True)
        
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        
        unet_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        text_encoder_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_text_encoder))
        
        weight_name = f"{cfg.ckpt_name}.safetensors" if cfg.save_safetensors else f"{cfg.ckpt_name}.bin"
        StableDiffusionPipeline.save_lora_weights(
            final_lora_dir, 
            unet_lora_layers=unet_lora_state, 
            text_encoder_lora_layers=text_encoder_lora_state, 
            safe_serialization=cfg.save_safetensors, 
            weight_name=weight_name
        )
        logger.info(f"Final LoRA weights saved to {os.path.join(final_lora_dir, weight_name)}")
        
    accelerator.end_training()

if __name__ == "__main__":
    main()