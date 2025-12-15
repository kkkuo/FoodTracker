import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import config
from model import NutritionEstimator
from dataset import prepare_data, create_dataloaders

def evaluate():
    train_df, val_df, test_df = prepare_data(config.CSV_DIR, config.IMG_ROOT)
    
    _, _, test_loader, scaler = create_dataloaders(train_df, val_df, test_df)
    
    print(f"Test Set Size: {len(test_df)}")
    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    model = NutritionEstimator(clip_model_name=config.CLIP_MODEL_NAME, device=config.DEVICE)
    
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
        
    model.to(config.DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    all_classes = [] 
   
    current_idx = 0

    print("Running Inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            
            preds = model(images)
            preds = torch.cat([p if p.dim() > 0 else p.unsqueeze(0) for p in preds], dim=-1)
            if preds.dim() == 1: preds = preds.unsqueeze(0) 

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            
            batch_size = images.size(0)
            batch_classes = test_df.iloc[current_idx : current_idx + batch_size]['class'].tolist()
            all_classes.extend(batch_classes)
            current_idx += batch_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_preds_real = scaler.inverse_transform(all_preds)
    all_labels_real = scaler.inverse_transform(all_labels)

    rmse = np.sqrt(mean_squared_error(all_labels_real, all_preds_real, multioutput='raw_values'))
    
    print("\n" + "="*50)
    print("整體 RMSE 結果：")
    print(f"卡路里 RMSE：{rmse[0]:.2f} kcal")
    print(f"脂肪 RMSE：{rmse[1]:.2f} g")
    print(f"蛋白質 RMSE：{rmse[2]:.2f} g")
    print(f"碳水化合物 RMSE：{rmse[3]:.2f} g")
    print("="*50)

    class_rmse_results = {}
    unique_classes = sorted(list(set(all_classes)))

    print("\n各食物類別詳細分析：")
    for class_name in unique_classes:
        indices = [i for i, c in enumerate(all_classes) if c == class_name]
        
        if not indices: continue

        class_preds = all_preds_real[indices]
        class_labels = all_labels_real[indices]

        c_rmse = np.sqrt(mean_squared_error(class_labels, class_preds, multioutput='raw_values'))
        
        class_rmse_results[class_name] = {
            'calories': c_rmse[0],
            'fat': c_rmse[1],
            'protein': c_rmse[2],
            'carbohydrates': c_rmse[3],
            'count': len(indices)
        }

        print(f"  [{class_name}] (n={len(indices)}) -> Cal: {c_rmse[0]:.1f}, Fat: {c_rmse[1]:.1f}, Prot: {c_rmse[2]:.1f}, Carb: {c_rmse[3]:.1f}")

if __name__ == "__main__":
    evaluate()