"""
DREAM: Multi-scale Prototype Memory Visualization Demo (True Training Simulation)
Scenario: Simulating long-term memory adaptation by repeatedly exposing the model 
to a consistent document structure (represented here by an identical sample).
"""

import os
# Enable ANSI terminal support for cross-platform status dashboard
os.system("")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import time
import random

# Path fix for dream_core integration
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from dream_core.multiscale_memory import DreamMultiscaleMemory

def run_visualization_demo():
    print("\n🚀 Initializing DREAM Multiscale Memory Module...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory_module = DreamMultiscaleMemory().to(device)
    
    # Optimizer for MHA projection matrices to learn visual alignment via SGD
    optimizer = torch.optim.Adam(memory_module.parameters(), lr=0.01)

    # Accelerate EMA momentum for demonstration purposes
    for m in [memory_module.memory_16, memory_module.memory_32, memory_module.memory_64]:
        m.ema_base = 0.2  
        m.ema_decay = False 
        
    print("📸 Simulating Hierarchical Visual Features from a Single Document...")
    torch.manual_seed(2026) 
    B = 1
    
    # DATA INITIALIZATION: 
    # Generating 'Horizontal Stripes' to represent spatial consistency across channels.
    # Using constant convolution kernels ensures uniform channel responses for visualization.
    feat_64_spatial = torch.randn(B, 256, 64, 64).to(device)
    
    down_64_to_32 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False).to(device)
    nn.init.constant_(down_64_to_32.weight, 1.0 / (3 * 3 * 256)) 
    
    down_32_to_16 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False).to(device)
    nn.init.constant_(down_32_to_16.weight, 1.0 / (3 * 3 * 512))

    with torch.no_grad():
        feat_32_spatial = down_64_to_32(feat_64_spatial)
        feat_16_spatial = down_32_to_16(feat_32_spatial)
        
        # Flattening 2D spatial features into 1D sequences [Batch, Tokens, Dim]
        feat_64 = feat_64_spatial.flatten(2).permute(0, 2, 1) 
        feat_32 = feat_32_spatial.flatten(2).permute(0, 2, 1) 
        feat_16 = feat_16_spatial.flatten(2).permute(0, 2, 1) 

    steps = 80
    loss_history = []
    similarity_history = []
    logs = []

    # UI Animation Components
    blocks = ['[ ]', '[\033[90m░\033[0m]', '[\033[36m▒\033[0m]', '[\033[93m▓\033[0m]', '[\033[91m█\033[0m]']
    m_state = random.choices(blocks, weights=[4, 3, 2, 1, 0], k=8)

    print("="*60)
    print(" 🧠 DREAM Explicit Adaptive Memory - Training Dashboard")
    print("="*60)
    for _ in range(11): sys.stdout.write("\n")
    sys.stdout.flush()

    for step in range(steps):
        
        def render_frame(phase_title, phase_arrow, is_read):
            # Dashboard stabilization logic
            sys.stdout.write("\r\033[10A") 
            q_weights = [1,2,3,4,2] if is_read else [3,3,2,1,1]
            q_blocks = "".join(random.choices(blocks, weights=q_weights, k=4))
            
            if not is_read:
                # Update 0-2 memory blocks to visualize slow EMA convergence
                num_changes = random.randint(0, 2)
                if num_changes > 0:
                    for idx in random.sample(range(8), num_changes):
                        m_state[idx] = random.choices(blocks, weights=[1, 2, 3, 4, 2], k=1)[0]
            
            m_blocks_str = "".join(m_state)
            lines = [
                f"  Status: {phase_title}",
                f"  [Visual Query]                                [Memory Bank]",
                f"    {q_blocks}         {phase_arrow}      {m_blocks_str}",
                f"  Step: {step+1:02d}/{steps} | Cosine Sim: {similarity_history[-1] if similarity_history else 0.0000:.4f}",
                "-" * 60,
                "  [Training Logs]"
            ]
            for i in range(5): lines.append(f"  {logs[i]}" if i < len(logs) else "")
            for i, line in enumerate(lines):
                sys.stdout.write(line + "\033[K" + ("" if i == len(lines) - 1 else "\n"))
            sys.stdout.flush()

        # --- PHASE 1: READ ---
        render_frame("\033[96m🔍 READING (Matching Phase)\033[0m", "\033[96m<=== READING ====\033[0m", is_read=True)
        
        # --- PHASE 2: ADAPTATION (SGD + EMA) ---
        # Note: This loop feeds an identical sample to simulate extreme structural consistency.
        memory_module.start_step()
        
        fused_features, sparse_loss = memory_module(feat_16, feat_32, feat_64, is_training=True)
        
        # Slice coarse-scale features for reconstruction assessment
        retrieved_16 = fused_features[:, :, 1024:2048]
        
        # Synergy: SGD trains the projection/alignment while EMA updates the memory content
        recon_loss = F.mse_loss(retrieved_16, feat_16)
        total_loss = recon_loss + 0.5 * sparse_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Metric logging
        sim = F.cosine_similarity(feat_16.flatten(), retrieved_16.flatten(), dim=0)
        loss_history.append(sparse_loss.item())
        similarity_history.append(sim.item())
        
        if (step + 1) % 10 == 0:
            logs.insert(0, f"Step {step+1:02d} | Sparsity: {sparse_loss.item():.4f} | Sim: {sim.item():.4f}")
            if len(logs) > 5: logs.pop()

        # --- PHASE 3: WRITE ---
        render_frame("\033[92m💾 WRITING (EMA Update Phase)\033[0m", "\033[92m==== WRITING ===>\033[0m", is_read=False)
        time.sleep(0.4)

    # --- PHASE 4: VISUALIZATION ---
    print("\n📊 Generating Visualization Plot...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    slice_idx = 50 
    # Displaying the 'Striped' input pattern
    ax1.imshow(feat_16[0, :slice_idx, :slice_idx].cpu().numpy(), cmap='viridis', aspect='auto')
    ax1.set_title("Input Visual Query (16x16)")
    ax1.axis('off')
    
    # Displaying how the memory has retrieved and reconstructed the pattern
    ax2.imshow(retrieved_16[0, :slice_idx, :slice_idx].cpu().detach().numpy(), cmap='viridis', aspect='auto')
    ax2.set_title(f"Retrieved Memory (Step {steps})")
    ax2.axis('off')
    
    # Quantifying the convergence: Sparsity vs. Cosine Similarity
    ax3.plot(loss_history, label='Sparsity Loss', color='red', marker='o', markersize=4)
    ax3.set_ylabel("Sparsity Loss", color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(similarity_history, label='Cosine Similarity', color='blue', marker='x', markersize=4)
    ax3_twin.set_ylabel("Cosine Sim", color='blue')
    ax3_twin.tick_params(axis='y', labelcolor='blue')
    
    ax3.set_title(f"Memory Adaptation Dynamics")
    ax3.set_xlabel("Steps (EMA + SGD)")
    
    save_path = "examples/dream_memory_visualize.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Simulation complete. Results saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    run_visualization_demo()