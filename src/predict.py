import torch
import pickle
from PIL import Image
import config
from model import NutritionEstimator
from dataset import get_transforms

def predict_single_image(image_path):
    with open(config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    model = NutritionEstimator(clip_model_name=config.CLIP_MODEL_NAME, device=config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    _, transform = get_transforms() 
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        preds = model(image_tensor)
        preds = torch.cat([p if p.dim() > 0 else p.unsqueeze(0) for p in preds], dim=-1)
        if preds.dim() == 1: preds = preds.unsqueeze(0)
        
        preds_numpy = preds.cpu().numpy()
        preds_real = scaler.inverse_transform(preds_numpy)[0]

    return {
        "Calories": preds_real[0],
        "Fat": preds_real[1],
        "Protein": preds_real[2],
        "Carbs": preds_real[3]
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        result = predict_single_image(img_path)
        print(result)
    else:
        print("Please provide image path.")