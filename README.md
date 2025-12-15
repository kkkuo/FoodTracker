# FoodTracker — A CLIP based nutrition estimation model

本專案使用 OpenAI CLIP 的vision encoder搭配MLP heads，從單張食物圖片預測四個營養素：卡路里、脂肪、蛋白質、碳水化合物。專案包含資料前處理、訓練、評估與單張推論。

- 模型實作：`src/model.py`
- 訓練腳本：`src/train.py`
- 評估腳本：`src/evaluate.py`
- 單張推論：`src/predict.py`
- 資料/轉換與資料載入：`src/dataset.py`
- 參數設定：`src/config.py`

## 專案結構

```
.
├─ README.md
├─ requirements.txt
└─ src/
   ├─ config.py
   ├─ dataset.py
   ├─ evaluate.py
   ├─ model.py
   ├─ predict.py
   └─ train.py
```

建議的資料與輸出目錄（可於 `config.py` 自訂）：

- 原始影像：`raw_data/`
- 標註 CSV：`training_tables/`
- 輸出（模型與標準化器）：`checkpoints/`

## 環境安裝

建議使用 Python 3.10+ 與虛擬環境。

```zsh
# 建立並啟用虛擬環境（macOS / Linux）
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install --upgrade pip
pip install -r requirements.txt
```

如需停用 Weights & Biases（W&B）線上記錄，可在執行前設定：

```zsh
export WANDB_MODE=disabled
```

若需使用 W&B，請先設定金鑰：

```zsh
export WANDB_API_KEY=你的_api_key
```

## 資料準備

- 影像路徑建議：`raw_data/{class_name}/{filename}`
- 每個類別一個標註 CSV，建議命名：`{class_name}_label_data.csv`，放於 `training_tables/`
- CSV 欄位需包含：
  - `filename`
  - `Calories(kcal)`, `Fat(g)`, `Protein(g)`, `Carbohydrates(g)`

CSV 範例：

```text
filename,Calories(kcal),Fat(g),Protein(g),Carbohydrates(g)
img_0001.jpg,250,10,12,30
img_0002.jpg,310,15,20,25
```

資料分割與欄位整理由 `dataset.prepare_data` 自動處理，會合併各類別 CSV、建立 `img_path` 與 `class` 欄位，並以固定亂數種子切出 train/val/test。

## 設定檔（config.py）

請在 `src/config.py` 中確認或調整以下常見項目：

- 路徑相關：
  - `CSV_DIR`：標註 CSV 根目錄（例如 `training_tables/`）
  - `IMG_ROOT`：影像根目錄（例如 `raw_data/`）
  - `MODEL_SAVE_PATH`：模型輸出路徑（例如 `checkpoints/CLIP_model.pth`）
  - `SCALER_PATH`：標準化器輸出路徑（例如 `checkpoints/clip_all_scaler.pkl`）
- 訓練超參數：
  - `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `PATIENCE`
- 模型/裝置：
  - `CLIP_MODEL_NAME`（例如 `ViT-B/32`）
  - `DEVICE`（自動偵測 `cuda` 或 `cpu`）

## 訓練

```zsh
python src/train.py
```

流程重點：

- `dataset.create_dataloaders` 會以訓練集擬合 `StandardScaler` 標準化四個回歸目標，並存成 `SCALER_PATH`。
- 模型使用 CLIP 視覺編碼器 + 4 個回歸 head 對應（卡、脂、蛋、碳）。
- 使用 `ReduceLROnPlateau` 與早停，並在驗證 loss 下降時儲存最佳權重到 `MODEL_SAVE_PATH`。
- 訓練與驗證指標會記錄到 W&B。

輸出：

- 最佳模型權重：`MODEL_SAVE_PATH`
- 目標標準化器：`SCALER_PATH`

## 評估

```zsh
python src/evaluate.py
```

- 載入測試集、最佳模型與標準化器，回推預測到實際單位，輸出整體與各類別預測結果之RMSE。

## 單張推論

```zsh
# 請將路徑替換為你的影像
python src/predict.py path/to/your_image.jpg
```

- 會載入 `MODEL_SAVE_PATH` 與 `SCALER_PATH`，並輸出四個營養素的實際數值。
- 若 `predict.py` 支援額外參數（例如輸出格式或裝置），請參考程式內的 `argparse` 說明。

輸出示例：

```json
{ "Calories": 285.4, "Fat": 12.1, "Protein": 9.8, "Carbs": 31.2 }
```

## 實作與資料前處理摘要

- 模型（`src/model.py`）
  - `NutritionEstimator` 使用 CLIP 圖像特徵作為 backbone，接 4 個回歸 head（各輸出 1 維）。
- 資料集與轉換（`src/dataset.py`）
  - `FoodDataset` 讀取影像、套用 transforms，並回傳標準化後的四維目標。
  - `get_transforms()`：訓練含資料增強、驗證/測試走 deterministic pipeline；Normalize 可依需求調整。
