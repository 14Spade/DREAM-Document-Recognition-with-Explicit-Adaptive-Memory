"""
Integration: DREAM + GOT-OCR 2.0 with Qwen3-0.6B Backbone

NOTE: This file demonstrates the migration from Qwen2 to Qwen3. 
The key change is using config.hidden_size to dynamically align 
the Multiscale Memory features (4096-dim) to the Qwen3 latent space.
"""

import torch
import torch.nn as nn
from transformers import Qwen3Model, Qwen3ForCausalLM, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# Import the modified vision tower that outputs 3 scales
from GOT.model.vision_encoder.vary_b_multi import build_vary_vit_b
from dream_module.multiscale_memory import DreamMultiscaleMemory
from GOT.utils.constants import GOTConfig

class GOTQwenModel(Qwen3Model):
    def __init__(self, config):
        super(GOTQwenModel, self).__init__(config)
        self.vision_tower_high = build_vary_vit_b()
        
        # Get Qwen3-0.6B's hidden dimension dynamically
        llm_hidden_size = config.hidden_size
        
        # Standard vision projector
        self.mm_projector_vary = nn.Linear(1024, llm_hidden_size)
        
        # [DREAM INJECTION]: Project combined multiscale features to Qwen3 dimension
        # Combined feature dim is 4096 (1024 * 4)
        self.mm_projector_memory = nn.Linear(4096, llm_hidden_size)

        self.dream_memory = DreamMultiscaleMemory()

    def forward(self, input_ids=None, images=None, inputs_embeds=None, **kwargs):
        loss_functions = {}
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self, 'vision_tower_high', None) is not None and images is not None:
            image_features = []
            for image in images:
                with torch.set_grad_enabled(False):
                    # Extract 3 hierarchical scales from our modified vary_b_multi
                    x3, x2, x1 = self.vision_tower_high(image[1])
                    
                    feat_16 = x3.flatten(2).permute(0, 2, 1)    # [B, 256, 1024]
                    feat_32 = x2.flatten(2).permute(0, 2, 1)    # [B, 1024, 512]
                    feat_64 = x1.flatten(2).permute(0, 2, 1)    # [B, 4096, 256]

                # [DREAM INJECTION]: Unified Multiscale Retrieval & Fusion
                combined_feature, sparse_loss = self.dream_memory(
                    feat_16, feat_32, feat_64, is_training=self.training
                )

                loss_functions["loss_sparse"] = sparse_loss
                
                # Project to Qwen3-0.6B space
                image_feature = self.mm_projector_memory(combined_feature)
                image_features.append(image_feature)

            # ... [Logic to insert image_features into inputs_embeds] ...

        outputs = super().forward(inputs_embeds=inputs_embeds, **kwargs)
        return outputs, loss_functions


class GOTQwenForCausalLM(Qwen3ForCausalLM):
    config_class = GOTConfig

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = GOTQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, labels=None, **kwargs):
        outputs, loss_functions = self.model(**kwargs)
        logits = self.lm_head(outputs[0]).float()
        
        loss = None
        if labels is not None:
            # Shift and compute standard CE Loss (loss_ce)
            # ... 
            loss_sparse = loss_functions.get("loss_sparse", 0.0)
            # Total Loss = Language Modeling Loss + DREAM Sparsity Regularization
            loss = loss_ce + loss_sparse * getattr(self.config, "loss_sparse_weight", 0.1)
            
        return CausalLMOutputWithPast(loss=loss, logits=logits)

# Register the new architecture
AutoModelForCausalLM.register(GOTConfig, GOTQwenForCausalLM)