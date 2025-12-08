# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM3 CoOp Wrapper for detector fine-tuning with learnable prompts.

This module wraps a frozen SAM3 model with a trainable CoOp prompt learner,
enabling efficient fine-tuning with only ~8K trainable parameters.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from sam3.model.coop_prompt_learner import CoOpPromptLearner
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt


class Sam3CoOpDetector(nn.Module):
    """
    SAM3 detector with CoOp learnable prompts.
    
    This wrapper freezes all SAM3 parameters and only trains the CoOp
    context vectors for efficient domain adaptation.
    
    Args:
        sam3_model: Pre-trained SAM3 image model
        coop_learner: CoOpPromptLearner instance
        resolution: Image resolution for inference (default: 1008)
    """
    
    def __init__(
        self,
        sam3_model: nn.Module,
        coop_learner: CoOpPromptLearner,
        resolution: int = 1008,
    ):
        super().__init__()
        self.sam3 = sam3_model
        self.coop = coop_learner
        self.resolution = resolution
        
        # Freeze all SAM3 parameters
        for param in self.sam3.parameters():
            param.requires_grad = False
        
        # Ensure CoOp is trainable
        for param in self.coop.parameters():
            param.requires_grad = True
    
    @property
    def device(self):
        return self.coop.ctx.device
    
    def train(self, mode: bool = True):
        """Set training mode - SAM3 stays in eval, only CoOp trains."""
        super().train(mode)
        # Keep SAM3 in eval mode for stable BatchNorm etc.
        self.sam3.eval()
        self.coop.train(mode)
        return self
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[Dict] = None,
    ) -> Dict:
        """
        Forward pass with learned CoOp prompts.
        
        Args:
            images: Batch of images [B, 3, H, W], normalized
            targets: Optional training targets with boxes/masks
            
        Returns:
            Dictionary with detector outputs (boxes, scores, masks)
        """
        batch_size = images.size(0)
        device = images.device
        
        # 1. Get image features (frozen)
        with torch.no_grad():
            backbone_out = self.sam3.backbone.forward_image(images)
        
        # 2. Get CoOp prompt embeddings (trainable!)
        prompt_embeds = self.coop(batch_size)  # [B, seq_len, dim]
        prompt_mask = self.coop.get_attention_mask(
            batch_size, 
            context_length=prompt_embeds.size(1)
        )
        
        # 3. Encode through text encoder (frozen, but gradients flow through embeddings)
        text_outputs = self._encode_coop_prompt(prompt_embeds, prompt_mask)
        backbone_out.update(text_outputs)
        
        # 4. Create find input for detector
        find_input = FindStage(
            img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
            text_ids=torch.zeros(batch_size, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        
        # 5. Create dummy geometric prompt
        geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=batch_size)
        
        # 6. Run detector (frozen, but gradients flow through language_features)
        out = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=targets,
            geometric_prompt=geometric_prompt,
        )
        
        return out
    
    def _encode_coop_prompt(
        self, 
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> Dict:
        """
        Encode CoOp prompt through frozen text encoder transformer.
        
        Args:
            prompt_embeds: [B, seq_len, dim] - embeddings from CoOp
            prompt_mask: [B, seq_len] - attention mask (True = padding)
            
        Returns:
            Dict with language_features, language_mask, language_embeds
        """
        text_encoder = self.sam3.backbone.language_backbone
        
        # Get positional embeddings
        seq_len = prompt_embeds.size(1)
        pos_embed = text_encoder.encoder.positional_embedding[:seq_len]
        
        # Add positional embeddings
        x = prompt_embeds + pos_embed.unsqueeze(0)
        
        # Build causal attention mask if needed
        attn_mask = text_encoder.encoder.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]
        
        # Pass through transformer (frozen, but gradients flow)
        x = text_encoder.encoder.transformer(x, attn_mask=attn_mask)
        
        # Apply final layer norm
        x = text_encoder.encoder.ln_final(x)
        
        # Resize to model dimension
        # x is [B, seq_len, encoder_width], needs to become [seq_len, B, d_model]
        x = x.transpose(0, 1)  # [seq_len, B, encoder_width]
        text_memory = text_encoder.resizer(x)  # [seq_len, B, d_model]
        
        # Invert mask for PyTorch convention (True = masked/padding)
        text_attention_mask = prompt_mask
        
        return {
            "language_features": text_memory,
            "language_mask": text_attention_mask,
            "language_embeds": prompt_embeds.transpose(0, 1),  # [seq_len, B, dim]
        }
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return list of trainable parameters (only CoOp)."""
        return list(self.coop.parameters())
    
    def print_trainable_stats(self):
        """Print statistics about trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {frozen:,}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable ratio: {trainable/total*100:.4f}%")
    
    def save_coop_weights(self, path: str):
        """Save only the CoOp weights."""
        torch.save({
            "coop_state_dict": self.coop.state_dict(),
            "n_ctx": self.coop.n_ctx,
            "ctx_dim": self.coop.ctx_dim,
            "class_token_position": self.coop.class_token_position,
        }, path)
        print(f"Saved CoOp weights to {path}")
    
    def load_coop_weights(self, path: str):
        """Load CoOp weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.coop.load_state_dict(checkpoint["coop_state_dict"])
        print(f"Loaded CoOp weights from {path}")


def build_sam3_coop_detector(
    class_name: str = "fallen bottle",
    n_ctx: int = 8,
    ctx_init: Optional[str] = None,
    class_token_position: str = "end",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> Sam3CoOpDetector:
    """
    Build SAM3 CoOp detector with learnable prompts.
    
    Args:
        class_name: Class name to embed (e.g., "fallen bottle")
        n_ctx: Number of learnable context tokens
        ctx_init: Optional initialization text
        class_token_position: Where to place class token
        checkpoint_path: Optional path to SAM3 checkpoint
        device: Device to load model on
        
    Returns:
        Sam3CoOpDetector ready for training
    """
    from sam3.model_builder import build_sam3_image_model
    
    # Build SAM3 model
    sam3_model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        enable_segmentation=True,
    )
    
    # Get tokenizer and token embedding from text encoder
    text_encoder = sam3_model.backbone.language_backbone
    tokenizer = text_encoder.tokenizer
    token_embedding = text_encoder.encoder.token_embedding
    
    # Create CoOp learner
    coop_learner = CoOpPromptLearner(
        n_ctx=n_ctx,
        ctx_dim=text_encoder.encoder.width,
        class_name=class_name,
        tokenizer=tokenizer,
        token_embedding=token_embedding,
        ctx_init=ctx_init,
        class_token_position=class_token_position,
    ).to(device)
    
    # Create wrapper
    wrapper = Sam3CoOpDetector(
        sam3_model=sam3_model,
        coop_learner=coop_learner,
    )
    
    wrapper.print_trainable_stats()
    
    return wrapper
