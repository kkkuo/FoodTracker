import torch

CSV_DIR = "training_tables"
IMG_ROOT = "raw_data"
MODEL_SAVE_PATH = "checkpoints/CLIP_model.pth"
SCALER_PATH = "checkpoints/clip_all_scaler.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 500
BATCH_SIZE = 24
LEARNING_RATE = 1e-5
PATIENCE = 5  

CLIP_MODEL_NAME = "ViT-B/16"