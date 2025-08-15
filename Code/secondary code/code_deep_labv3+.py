# -*- coding: utf-8 -*-
"""
DeepLabV3+ pipeline complet (entraînement + inférence tuiles + mosaïque)
Adaptation du script UNet original vers segmentation_models_pytorch DeepLabV3+.
Created on Tue Jul 1 15:05:58 2025 (adapted)
@author: walid/rabeh

pip install segmentation-models-pytorch==0.3.4 efficientnet-pytorch

"""
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
from torchvision import transforms

# Rasterio / géotifs
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.crs import CRS

# === Réglages utilisateur ===
os.chdir("C:/SUCHIES/Python")   # adapte si nécessaire

data_root = 'C:/SUCHIES/Python/Test_Unet/12th_Dataset'  # dataset images / masks
image_dir = os.path.join(data_root, 'Images')
mask_dir = os.path.join(data_root, 'Masks')

# Vérifications chemins
assert os.path.exists(image_dir), f"Le dossier {image_dir} n'existe pas."
assert os.path.exists(mask_dir), f"Le dossier {mask_dir} n'existe pas."

# === Paramètres globaux ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_size = (256, 256)
batch_size = 8
num_workers = 4
num_epochs = 20
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# === Transforms ===
# Pour l'entraînement on utilise ToTensor() + normalisation ImageNet (conforme aux poids ImageNet)
train_transform = transforms.Compose([
    transforms.ToTensor(),   # accepte PIL Image ou np.ndarray uint8
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Pour l'inférence tuiles (on redimensionne à 256)
inference_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Dataset ===
class UrbanSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256,256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.*')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.*')))
        assert len(self.image_paths) == len(self.mask_paths), "Le nombre d'images et de masques ne correspond pas."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Charger
        image = cv2.imread(img_path)  # BGR uint8
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # -> RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # uint8

        # Redimensionner
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Binariser le masque
        mask = (mask > 128).astype(np.uint8)  # 0/1

        # Si transform fourni (ToTensor + Normalize), on passe une image uint8 ou PIL
        if self.transform:
            # torchvision.ToTensor accepte numpy uint8 HWC
            image_t = self.transform(image)
        else:
            # Normalisation manuelle si pas de transform
            image = image.astype(np.float32) / 255.0
            image_t = torch.from_numpy(image).float().permute(2,0,1)

        mask_t = torch.from_numpy(mask).long()  # [H, W] int64 pour CrossEntropy, mais on utilisera BCEWithLogits -> float
        return image_t, mask_t

# Créer dataset / dataloader
train_dataset = UrbanSegDataset(image_dir, mask_dir, transform=train_transform, target_size=target_size)

from torch.utils.data import DataLoader, random_split

# Taille du dataset total
dataset_size = len(train_dataset)

# Taille de la validation (exemple : 20%)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size

# Split entraînement / validation
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoaders avec num_workers=0
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)



# Visualisation rapide d'un échantillon (optionnel)
try:
    sample_img, sample_mask = train_dataset[0]
    img_show = sample_img.permute(1,2,0).numpy()
    # Si normalisé, renormaliser pour affichage approximatif (on clipe)
    img_show = np.clip((img_show * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])), 0,1)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(img_show); plt.title("Image"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(sample_mask.numpy(), cmap='gray'); plt.title("Masque"); plt.axis('off')
    plt.show()
except Exception as e:
    print("Visualisation échantillon : problème (peut être dû à remote env).", e)

# === Modèle : DeepLabV3+ via segmentation_models_pytorch ===
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None  # logits
).to(device)
print("Model chargé :", type(model).__name__)

# === Loss / Optim / Scheduler ===
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# === Entraînement ===
checkpoint_dir = "C:/SUCHIES/Python/Models"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'deeplabv3p_resnet34_20_dataset12.pth')

print("Début entraînement sur device:", device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device).float()
        # masks shape [B, H, W] -> [B,1,H,W], float
        masks = masks.unsqueeze(1).to(device).float()

        optimizer.zero_grad()
        logits = model(images)  # [B,1,H,W]
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")
    scheduler.step(avg_loss)

    # sauvegarde périodique
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, checkpoint_path)

print("Entraînement terminé. Checkpoint sauvegardé dans:", checkpoint_path)

# === Inference tuiles & mosaïque (adapte tes dossiers ci-dessous) ===
input_folder = "C:/SUCHIES/Python/All jp2"
tile_output_root = "C:/SUCHIES/Python/Tuiles_Georef"
mask_output_root = "C:/SUCHIES/Python/Masques_Georef"
mosaic_output_root = "C:/SUCHIES/Python/Other algo/Deep_lab/all_france"
tile_size = 256

os.makedirs(tile_output_root, exist_ok=True)
os.makedirs(mask_output_root, exist_ok=True)
os.makedirs(mosaic_output_root, exist_ok=True)

# Charger le modèle final (déjà sauvegardé)
# Si tu souhaites charger uniquement les poids :
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(device)
# model.eval()

# Ici on suppose que model contient déjà les poids suite à l'entraînement ci-dessus.
model.eval()

# Progress log (évite retraitements)
log_file = "progress_log - Copie.txt"
processed_tiles = set()
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        processed_tiles = set(line.strip() for line in f)

jp2_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jp2")]

for jp2_file in jp2_files:
    input_jp2_path = os.path.join(input_folder, jp2_file)
    base_name = os.path.splitext(jp2_file)[0]

    with rasterio.open(input_jp2_path) as src:
        width, height = src.width, src.height
        profile = src.profile
        crs = src.crs if src.crs else CRS.from_string("EPSG:2154")
        transform_global = src.transform

        print(f"\n📂 Traitement : {jp2_file} - Dimensions: {width}x{height}")
        
        tile_id = 0
        mask_paths = []

        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                tile_key = f"{base_name}_tile_{tile_id:05d}"
                if tile_key in processed_tiles:
                    tile_id += 1
                    continue

                win_width = min(tile_size, width - left)
                win_height = min(tile_size, height - top)
                if win_width <= 0 or win_height <= 0:
                    tile_id += 1
                    continue

                window = Window(left, top, win_width, win_height)
                tile_transform = transform_global * Affine.translation(left, top)
                
                tile_data = src.read(window=window)

                # Pad si bord
                if tile_data.shape[1] != tile_size or tile_data.shape[2] != tile_size:
                    padded_tile = np.zeros((tile_data.shape[0], tile_size, tile_size), dtype=tile_data.dtype)
                    padded_tile[:, :win_height, :win_width] = tile_data
                else:
                    padded_tile = tile_data

                tile_profile = profile.copy()
                tile_profile.update({
                    "height": win_height,
                    "width": win_width,
                    "transform": tile_transform,
                    "crs": crs,
                    "driver": "GTiff",
                    "count": 3,
                    "dtype": "uint8"
                })

                tile_filename = os.path.join(tile_output_root, f"{tile_key}.tif")
                mask_filename = os.path.join(mask_output_root, f"{base_name}_mask_{tile_id:05d}.tif")

                # Ecrire la tuile RGB (3 bandes)
                with rasterio.open(tile_filename, "w", **tile_profile) as dst:
                    # tile_data peut avoir >3 bandes, on prend les 3 premières
                    dst.write(tile_data[:3])

                # Préparer image pour inference (PIL RGB)
                tile_rgb = np.moveaxis(padded_tile[:3], 0, -1)  # HWC
                pil_image = Image.fromarray(tile_rgb)

                with torch.no_grad():
                    input_tensor = inference_transform(pil_image).unsqueeze(0).to(device).float()
                    logits = model(input_tensor)                      # [1,1,H,W]
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # [H,W] 0..1
                    predicted_mask = (probs > 0.2).astype(np.uint8) * 255

                # Recouper à la taille réelle (bord)
                predicted_mask = predicted_mask[:win_height, :win_width]

                mask_profile = tile_profile.copy()
                mask_profile.update({
                    "count": 1,
                    "dtype": "uint8",
                    "nodata": None
                })

                with rasterio.open(mask_filename, "w", **mask_profile) as dst:
                    dst.write(predicted_mask[np.newaxis, ...])

                mask_paths.append(mask_filename)

                with open(log_file, "a") as log:
                    log.write(tile_key + "\n")

                print(f"✅ {jp2_file} - Tuile {tile_id} traitée")

                # Suppression immédiate de la tuile pour libérer la mémoire/disque si besoin
                if os.path.exists(tile_filename):
                    os.remove(tile_filename)

                tile_id += 1

        # Fusion des masques
        print(f"🧩 Fusion des {len(mask_paths)} masques pour {base_name}...")
        if not mask_paths:
            print(f"⚠️ Aucun masque à fusionner pour {base_name}, saut de la mosaïque.")
            continue
        sources = [rasterio.open(path) for path in mask_paths]
        mosaic, mosaic_transform = merge(sources)

        mosaic_profile = profile.copy()
        mosaic_profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform,
            "crs": crs,
            "count": 1,
            "dtype": "uint8",
            "driver": "GTiff"
        })

        mosaic_output_path = os.path.join(mosaic_output_root, f"{base_name}_mosaic.tif")
        with rasterio.open(mosaic_output_path, "w", **mosaic_profile) as dst:
            dst.write(mosaic[0], 1)

        for src in sources:
            src.close()

        # Suppression des masques intermédiaires
        for mask_path in mask_paths:
            if os.path.exists(mask_path):
                os.remove(mask_path)

        print(f"🎉 Mosaïque générée et masques supprimés : {mosaic_output_path}")

# === Fusion finale des mosaïques (si plusieurs) ===
print("\n🧩 Fusion finale : mosaïque des mosaïques...")

mosaic_files = [os.path.join(mosaic_output_root, f) for f in os.listdir(mosaic_output_root) if f.endswith(".tif")]

if len(mosaic_files) < 2:
    print("⚠️ Pas assez de mosaïques pour fusion finale.")
else:
    sources = [rasterio.open(f) for f in mosaic_files]
    super_mosaic, super_transform = merge(sources)

    super_profile = sources[0].profile.copy()
    super_profile.update({
        "height": super_mosaic.shape[1],
        "width": super_mosaic.shape[2],
        "transform": super_transform,
        "count": 1,
        "dtype": "uint8",
        "driver": "GTiff",
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    })

    final_output_path = os.path.join(mosaic_output_root, "super_mosaic_deeplab.tif")
    with rasterio.open(final_output_path, "w", **super_profile) as dst:
        dst.write(super_mosaic[0], 1)

    for src in sources:
        src.close()

    print(f"🎯 Mosaïque finale compressée générée : {final_output_path}")

print("FIN du pipeline DeepLabV3+")
