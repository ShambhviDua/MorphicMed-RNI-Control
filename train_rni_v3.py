"""
RNI (Resistive Network Inverse) Solver - "The Voltage-to-Temp Guessinator"
attempt to brute-force map 308 voltages → 308 temps
COMSOL gave us 10k sims, now we're making this NN chew through them.
Last stable: 2025-12-15 - DO NOT "IMPROVE" WITHOUT TESTING
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================================================================
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ============================================================================

# --- HYPERPARAMETERS (FINALIZED AFTER 47 TRIALS) ---
# don't touch LR - it either does nothing or explodes. 1e-4 is magic.
BATCH_SIZE = 128      # Fits in my RTX 3080 (yes, 3080 samples with 308 nodes, cute)
LEARNING_RATE = 1e-4  # The "Goldilocks" - not too hot, not too cold
EPOCHS = 150          # Upped from 50 because loss plateaued around epoch 40
NOISE_STD = 0.0035    # Measured from actual breadboard noise (scope said ~3.5mV RMS)

class RNIDataset(Dataset):
    """Loads that massive COMSOL dump."""
    
    def __init__(self, npy_path):
        print(f"[DATA LOADER] Grabbing data from {npy_path}...")
        # The .item() saved it weirdly
        raw_data = np.load(npy_path, allow_pickle=True).item()
        
        # Shape: (N, 308) where N≈10k, 308 = resistor nodes
        self.voltages = torch.FloatTensor(raw_data['V'])
        self.temps = torch.FloatTensor(raw_data['T'])
        
        # Safety net: COMSOL sometimes spits out NaN at boundary conditions
        nan_count = torch.isnan(self.voltages).sum().item()
        if nan_count > 0:
            print(f"[WARNING] Found {nan_count} NaN voltage readings! Zeroing them.")
            self.voltages[torch.isnan(self.voltages)] = 0.0
    
    def __len__(self):
        return len(self.voltages)
    
    def __getitem__(self, idx):
        """Returns noisy voltage → clean temperature pair."""
        v = self.voltages[idx]
        t = self.temps[idx]
        
        # Add synthetic noise so model learns real-world messiness
        # Different noise every epoch = free data augmentation
        noise = torch.randn_like(v) * NOISE_STD
        return v + noise, t


class RNI_Net(nn.Module):
    """
    The "Brain" - tries to guess 308 temperatures from 308 voltages.
    Architecture notes:
    - Started with 512 hidden, was underfitting
    - Tried 2048, overfit like crazy
    - 1024→512 seems to be the sweet spot (for now)
    - Dropout at 0.3 prevents memorization of COMSOL artifacts
    """
    
    def __init__(self):
        super(RNI_Net, self).__init__()
        
        self.main = nn.Sequential(
            # Input layer: 308 voltages in
            nn.Linear(308, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # Aggressive dropout prevents overfitting to sim noise
            
            # Hidden layer
            nn.Linear(1024, 512),
            nn.ReLU(),
            # Tried dropout here at 0.2, but loss stopped decreasing (removed 2024-03-12)
            
            # Output: 308 temperature predictions (0-50°C range)
            nn.Linear(512, 308)
            # No activation - we want raw regression output
            # If temps go negative, that's a physics problem, not a network problem
        )
    
    def forward(self, x):
        # Simple feed-forward
        return self.main(x)


def train():
    """Main training loop."""
    
    # Device setup (pray for CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[SETUP] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[SETUP] Using CPU (this will take approximately forever)")
    
    # ========================================================================
    # DATA CHECK - Generate dummy data if COMSOL run isn't ready
    # ========================================================================
    data_path = "data/comsol_full_sweep_v2.npy"
    if not os.path.exists(data_path):
        print("[DEBUG] Real data not found! Generating random junk for pipeline test...")
        os.makedirs("data", exist_ok=True)
        
        # 10k random samples, 308 nodes each
        dummy_data = {
            'V': np.random.randn(10000, 308) * 5 + 12,  # ~12V ±5V
            'T': np.random.rand(10000, 308) * 45         # 0-45°C range
        }
        np.save(data_path, dummy_data)
        print("[DEBUG] Dummy data saved. This won't train anything useful")
    
    # Load actual data
    dataset = RNIDataset(data_path)
    
    # num_workers > 0 causes pickle errors on Windows/Linux cross-saves
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Initialize model
    model = RNI_Net().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Mean squared error - simple and effective
    criterion = nn.MSELoss()
    
    # Loss tracking
    loss_history = []
    
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    # ========================================================================
    # EPOCH LOOP
    # ========================================================================
    for epoch in range(EPOCHS):
        model.train()  # Set to training mode
        epoch_loss = 0.0
        
        # Batch loop
        for batch_idx, (voltages, temps) in enumerate(loader):
            voltages = voltages.to(device)
            temps = temps.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(voltages)
            
            # Compute loss
            loss = criterion(predictions, temps)
            
            # Check for catastrophic failure
            if torch.isnan(loss):
                print(f"\n[CRITICAL] NaN loss at batch {batch_idx}! Something's broken.")
                print("Probably exploding gradients. Try reducing LR or adding more dropout.")
                break
            
            # Backprop
            loss.backward()
            
            # Gradient clipping - without this, occasional spikes nuke the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # End of epoch
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        # ====================================================================
        # PROGRESS REPORT (formatted for easy grep'ing)
        # ====================================================================
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {avg_loss:.6f} | "
              f"RMSE: {np.sqrt(avg_loss):.4f}°C")
        
        # Save checkpoint every 10 epochs (laptop overheats sometimes)
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"rni_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [Checkpoint saved to {checkpoint_path}]")
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Save loss history (for plotting with that separate script)
    with open("training_log_final.txt", "w") as f:
        f.write("# Loss per epoch\n")
        f.write(f"# Settings: LR={LEARNING_RATE}, BS={BATCH_SIZE}, Epochs={EPOCHS}\n")
        for i, loss_val in enumerate(loss_history):
            f.write(f"{i+1}\t{loss_val}\n")
    
    print("\n[INFO] Loss log saved to 'training_log_final.txt'")
    print("[INFO] To plot: python plot_loss.py (script is in this folder)")
    print("[INFO] Best model weights: checkpoints/rni_epoch_150.pth")
    
    return model


if __name__ == "__main__":
    # Entry point
    print("\n" + "="*60)
    print("RNI VOLTAGE-TO-TEMPERATURE PREDICTOR")
    print("="*60)
    print("(Resistive Network Inverse Solver v2.3)")
    print("\n")
    
    trained_model = train()
