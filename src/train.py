import torch
import torch.nn as nn
import wandb
import numpy as np
import os
import config
from model import NutritionEstimator
from dataset import prepare_data, create_dataloaders

def train():
    wandb.init(project="ClIP-attemp1", name="refactored_run", config={
        "architecture": config.CLIP_MODEL_NAME,
        "epochs": config.EPOCHS,
        "lr": config.LEARNING_RATE
    })

    print("Preparing Data...")
    train_df, val_df, test_df = prepare_data(config.CSV_DIR, config.IMG_ROOT)
    train_loader, val_loader, test_loader, scaler = create_dataloaders(train_df, val_df, test_df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    model = NutritionEstimator(clip_model_name=config.CLIP_MODEL_NAME, device=config.DEVICE)
    model.to(config.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_loss = float('inf')
    no_improve_epochs = 0
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            cal, fat, protein, carb = model(images)
            
            loss = (
                criterion(cal.squeeze(), labels[:, 0]) +
                2 * criterion(fat.squeeze(), labels[:, 1]) +
                2 * criterion(protein.squeeze(), labels[:, 2]) +
                2 * criterion(carb.squeeze(), labels[:, 3])
            )
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                cal, fat, protein, carb = model(images)
                
                val_loss = (
                    criterion(cal.squeeze(), labels[:, 0]) +
                    criterion(fat.squeeze(), labels[:, 1]) +
                    criterion(protein.squeeze(), labels[:, 2]) +
                    criterion(carb.squeeze(), labels[:, 3])
                )
                val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        wandb.log({"train/loss": avg_loss, "val/loss": avg_val_loss})
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("Model saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config.PATIENCE:
                print("Early stopping.")
                break

    wandb.finish()

if __name__ == "__main__":
    train()