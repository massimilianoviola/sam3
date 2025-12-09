# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
CoOp (Context Optimization) Prompt Learner for SAM3.

Implements learnable context vectors that replace hand-crafted text prompts
for improved detection performance on specific domains.

Reference: "Learning to Prompt for Vision-Language Models" (Zhou et al., 2022)
"""

from typing import Optional

import torch
import torch.nn as nn


class CoOpPromptLearner(nn.Module):
    """
    Learnable context vectors for SAM3 text encoder.
    
    Instead of using a hand-crafted prompt like "A fallen bottle", this module
    learns continuous context vectors [V1][V2]...[Vn] that are concatenated
    with the class name embedding and fed through the text encoder.
    
    Args:
        n_ctx: Number of learnable context tokens (default: 8)
        ctx_dim: Dimension of context vectors (must match text encoder width, default: 1024)
        class_name: The class name to embed (e.g., "fallen bottle")
        tokenizer: SAM3 tokenizer for encoding the class name
        token_embedding: The token embedding layer from VETextEncoder
        ctx_init: Optional initialization text (e.g., "a photo of a")
        class_token_position: Where to place class token - 'end', 'middle', 'front'
    """
    
    def __init__(
        self,
        class_name: str,
        tokenizer,
        token_embedding: nn.Embedding,
        n_ctx: int = 8,
        ctx_dim: int = 1024,
        ctx_init: Optional[str] = None,
        class_token_position: str = "end",
    ):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.class_token_position = class_token_position
        
        if ctx_init is not None and tokenizer is not None and token_embedding is not None:
            # Initialize context from text
            ctx_init_tokens = tokenizer([ctx_init], context_length=n_ctx + 2)
            # Remove SOT and EOT tokens
            ctx_init_tokens = ctx_init_tokens[:, 1:-1][:, :n_ctx]
            with torch.no_grad():
                ctx_vectors = token_embedding(ctx_init_tokens).squeeze(0)
            # Pad if needed
            if ctx_vectors.size(0) < n_ctx:
                padding = torch.randn(n_ctx - ctx_vectors.size(0), ctx_dim) * 0.02
                ctx_vectors = torch.cat([ctx_vectors, padding], dim=0)
            self.ctx = nn.Parameter(ctx_vectors)
        else:
            # Random initialization
            self.ctx = nn.Parameter(torch.randn(n_ctx, ctx_dim) * 0.02)
        
        # Get class name token embeddings (frozen)
        if tokenizer is not None and token_embedding is not None:
            # Tokenize just the class name (without SOT/EOT)
            class_tokens = tokenizer([class_name], context_length=32)
            # Find where the actual tokens are (between SOT and EOT)
            # SOT is at position 0, EOT marks the end
            sot_token = tokenizer.sot_token_id
            eot_token = tokenizer.eot_token_id
            
            # Get device from embedding layer
            embed_device = token_embedding.weight.device
            
            with torch.no_grad():
                # Move tokens to same device as embedding
                class_tokens_dev = class_tokens.to(embed_device)
                class_embed = token_embedding(class_tokens_dev)  # [1, seq_len, dim]
                
            # Find valid token indices (between SOT and EOT)
            tokens_np = class_tokens[0].cpu().numpy()
            eot_idx = (tokens_np == eot_token).argmax()
            # Class tokens are indices 1 to eot_idx-1 (excluding SOT and EOT)
            class_embed = class_embed[:, 1:eot_idx, :]  # [1, n_class_tokens, dim]
            
            self.register_buffer("class_token_embed", class_embed.cpu())
            self.register_buffer("n_class_tokens", torch.tensor(class_embed.size(1)))
            
            # Also store SOT and EOT embeddings
            sot_tensor = torch.tensor([[sot_token]], device=embed_device)
            eot_tensor = torch.tensor([[eot_token]], device=embed_device)
            with torch.no_grad():
                sot_embed = token_embedding(sot_tensor)
                eot_embed = token_embedding(eot_tensor)
            self.register_buffer("sot_embed", sot_embed.cpu())  # [1, 1, dim]
            self.register_buffer("eot_embed", eot_embed.cpu())  # [1, 1, dim]
        else:
            # Placeholder - will be set later
            self.register_buffer("class_token_embed", torch.zeros(1, 2, ctx_dim))
            self.register_buffer("n_class_tokens", torch.tensor(2))
            self.register_buffer("sot_embed", torch.zeros(1, 1, ctx_dim))
            self.register_buffer("eot_embed", torch.zeros(1, 1, ctx_dim))
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate prompt embeddings for the given batch size.
        
        Returns:
            Tensor of shape [batch_size, seq_len, ctx_dim] containing:
            [SOT][V1][V2]...[Vn][CLASS_TOKENS][EOT][PADDING...]
        """
        # Expand context to batch size
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_ctx, dim]
        
        # Expand class embeddings to batch size
        class_embed = self.class_token_embed.expand(batch_size, -1, -1)  # [B, n_class, dim]
        sot = self.sot_embed.expand(batch_size, -1, -1)  # [B, 1, dim]
        eot = self.eot_embed.expand(batch_size, -1, -1)  # [B, 1, dim]
        
        # Concatenate based on class token position
        if self.class_token_position == "end":
            # [SOT][V1]...[Vn][CLASS][EOT]
            prompt = torch.cat([sot, ctx, class_embed, eot], dim=1)
        elif self.class_token_position == "front":
            # [SOT][CLASS][V1]...[Vn][EOT]
            prompt = torch.cat([sot, class_embed, ctx, eot], dim=1)
        elif self.class_token_position == "middle":
            # [SOT][V1]...[Vn/2][CLASS][Vn/2+1]...[Vn][EOT]
            half = self.n_ctx // 2
            prompt = torch.cat([
                sot, 
                ctx[:, :half, :], 
                class_embed, 
                ctx[:, half:, :], 
                eot
            ], dim=1)
        else:
            raise ValueError(f"Unknown class_token_position: {self.class_token_position}")
        
        return prompt
    
    def get_attention_mask(self, batch_size: int = 1, context_length: int = 32) -> torch.Tensor:
        """
        Generate attention mask for the prompt.
        
        Returns:
            Boolean tensor of shape [batch_size, context_length] where True = padding (masked).
        """
        device = self.ctx.device
        # Actual prompt length: SOT + n_ctx + n_class_tokens + EOT
        prompt_len = 1 + self.n_ctx + self.n_class_tokens.item() + 1
        
        mask = torch.ones(batch_size, context_length, dtype=torch.bool, device=device)
        mask[:, :prompt_len] = False  # False = valid token
        
        return mask
    
    @property
    def trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return self.ctx.numel()
    
    def extra_repr(self) -> str:
        return f"n_ctx={self.n_ctx}, ctx_dim={self.ctx_dim}, class_token_position={self.class_token_position}"
