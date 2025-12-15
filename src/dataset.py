import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler
import pickle
import config

class FoodDataset(Dataset):
    def __init__(self, dataframe, scaler=None, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        self.labels = self.df[['Calories(kcal)', 'Fat(g)', 'Protein(g)', 'Carbohydrates(g)']].values.astype(np.float32)

        if scaler is None:
            self.scaler = StandardScaler()
            self.labels = self.scaler.fit_transform(self.labels)
        else:
            self.scaler = scaler
            self.labels = self.scaler.transform(self.labels)

        self.image_paths = self.df['img_path'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return image, label

def get_transforms():
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.052),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(3),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

def prepare_data(csv_dir, img_root):
    dfs = []
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith(".csv"):
            class_name = csv_file.replace("_label_data.csv", "")
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            df["img_path"] = df["filename"].apply(lambda x: os.path.join(img_root, class_name, x))
            df["class"] = class_name
            dfs.append(df)

    full_data = pd.concat(dfs, ignore_index=True)
    
    train_list, val_list, test_list = [], [], []
    torch.manual_seed(42)

    for class_name, group in full_data.groupby("class"):
        n = len(group)
        indices = torch.randperm(n)
        n_test = max(1, int(0.05 * n))
        n_val = max(1, int(0.05 * n))
        
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test+n_val]
        train_idx = indices[n_test+n_val:]
        
        test_list.append(group.iloc[test_idx])
        val_list.append(group.iloc[val_idx])
        train_list.append(group.iloc[train_idx])

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    return train_df, val_df, test_df

def create_dataloaders(train_df, val_df, test_df):
    scaler = StandardScaler()
    scaler.fit(train_df[['Calories(kcal)', 'Fat(g)', 'Protein(g)', 'Carbohydrates(g)']].values)
    
    os.makedirs(os.path.dirname(config.SCALER_PATH), exist_ok=True)
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    train_tf, test_tf = get_transforms()

    train_dataset = FoodDataset(train_df, scaler=scaler, transform=train_tf)
    val_dataset = FoodDataset(val_df, scaler=scaler, transform=test_tf)
    test_dataset = FoodDataset(test_df, scaler=scaler, transform=test_tf)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, scaler