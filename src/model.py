import torch
import torch.nn as nn
import clip

class NutritionEstimator(nn.Module):
    def __init__(self, clip_model_name="ViT-B/16", device="cuda"):
        super().__init__()
    
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False

        embed_dim = self.clip_model.visual.output_dim

        self.head_cal = self._make_head(embed_dim)
        self.head_fat = self._make_head(embed_dim)
        self.head_protein = self._make_head(embed_dim)
        self.head_carb = self._make_head(embed_dim)

    def _make_head(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x).float()
        
        cal = self.head_cal(x)
        fat = self.head_fat(x)
        protein = self.head_protein(x)
        carb = self.head_carb(x)

        return cal, fat, protein, carb