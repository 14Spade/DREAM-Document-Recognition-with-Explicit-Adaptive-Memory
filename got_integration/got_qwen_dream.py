"""
Integration Example: DREAM + GOT-OCR 2.0 (LLM Backbone)

NOTE (HOW TO USE): 
Do not run this file directly. Instead, open the original `GOT/model/GOT_ocr_2_0.py` 
and look for the tags [DREAM INJECTION] below to see exactly where to insert 
our multiscale memory logic into the Qwen pipeline.
"""

import torch
import torch.nn as nn
from transformers import Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# Import the MODIFIED vision tower that outputs 3 scales
from GOT.model.vision_encoder.vary_b_multi import build_vary_vit_b

# ====================================================================
# [DREAM INJECTION 1]: Import our Plug-and-Play Multiscale Memory
# ====================================================================
from dream_module.multiscale_memory import DreamMultiscaleMemory

class GOTQwenModel(Qwen2Model):
    def __init__(self, config):
        super(GOTQwenModel, self).__init__(config)
        self.vision_tower_high = build_vary_vit_b()
        self.mm_projector_vary = nn.Linear(1024, 1024)
        
        # ====================================================================
        # [DREAM INJECTION 2]: Memory Initialization & Projector Expansion
        # ====================================================================
        # New projector for concatenated multiscale memory features 
        self.mm_projector_memory = nn.Linear(4096, 1024)
        # Initialize the DREAM Multiscale Prototype Memory
        self.dream_memory = DreamMultiscaleMemory()

    def forward(self, input_ids=None, images=None, inputs_embeds=None, **kwargs):
        loss_functions = {}
        
        # ====================================================================
        # [OMITTED ORIGINAL CODE 1]
        # KEEP the original LLaVA-style text embedding logic here:
        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)
        # ====================================================================

        if getattr(self, 'vision_tower_high', None) is not None and images is not None:
            image_features = []
            for image in images:
                with torch.set_grad_enabled(False):
                    # 1. Extract Multiscale Features from our modified Vision Tower
                    cnn_feature, high_res_32, high_res_64 = self.vision_tower_high(image[1])
                    
                    cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)    # [1, 256, 1024]
                    high_res_32 = high_res_32.flatten(2).permute(0, 2, 1)    # [1, 1024, 512]
                    high_res_64 = high_res_64.flatten(2).permute(0, 2, 1)    # [1, 4096, 256]

                # ====================================================================
                # [DREAM INJECTION 3]: Retrieve, Align, and Fuse in 1 Line
                # ====================================================================
                combined_feature, sparse_loss = self.dream_memory(
                    feat_16=cnn_feature, 
                    feat_32=high_res_32, 
                    feat_64=high_res_64, 
                    is_training=self.training
                )

                loss_functions["loss_sparse"] = sparse_loss
                image_feature = self.mm_projector_memory(combined_feature)
                image_features.append(image_feature)

            # ====================================================================
            # [OMITTED ORIGINAL CODE 2]
            # KEEP the original sequence concatenation logic here:
            # (Replacing <im_start> and <im_end> tokens with the extracted image_features)
            # ====================================================================

        outputs = super().forward(inputs_embeds=inputs_embeds, **kwargs)
        return outputs, loss_functions


class GOTQwenForCausalLM(Qwen2ForCausalLM):
    # ... [__init__ method omitted, keep original] ...

    def forward(self, labels=None, **kwargs):
        outputs, loss_functions = self.model(**kwargs)
        logits = self.lm_head(outputs[0]).float()
        
        loss = None
        if labels is not None:
            # ====================================================================
            # [OMITTED ORIGINAL CODE 3]
            # KEEP the original CrossEntropy loss calculation here (loss_ce)
            # ====================================================================
            loss_ce = loss_functions.get("loss_ce", 0.0) # Placeholder
            
            # ====================================================================
            # [DREAM INJECTION 4]: Add Sparsity Regularization to Total Loss
            # ====================================================================
            loss_sparse = loss_functions.get("loss_sparse", 0.0)
            loss = loss_ce + loss_sparse * getattr(self.config, "loss_sparse_weight", 0.1)
            
        return CausalLMOutputWithPast(loss=loss, logits=logits)