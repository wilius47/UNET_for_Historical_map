# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:05:58 2025

@author: rabeh
"""
import os

os.chdir("C:/SUCHIES/Python")  

# Importer les bibliothèques
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from glob import glob

"""Cellule 2 : Préparation des données

"""

# Définir le chemin des données 
data_root = 'C:/SUCHIES/Python/Test_Unet/12th_Dataset'  # Remplacez par votre chemin
image_dir = os.path.join(data_root, 'Images')
mask_dir = os.path.join(data_root, 'Masks')

# Vérifier que les dossiers existent
assert os.path.exists(image_dir), f"Le dossier {image_dir} n'existe pas."
assert os.path.exists(mask_dir), f"Le dossier {mask_dir} n'existe pas."

# Classe Dataset pour charger les images et les masques
class UrbanSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.*')))  # Toutes les extensions
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.*')))    # Toutes les extensions

        # Vérifier que les images et les masques correspondent
        assert len(self.image_paths) == len(self.mask_paths), "Le nombre d'images et de masques ne correspond pas."



    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Charger l'image et le masque
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Redimensionner l'image et le masque
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Normaliser l'image entre 0 et 1
        image = image / 255.0

        # Binariser le masque (urbain = 1, autre = 0)
        mask = (mask > 128).astype(np.uint8)

        # Appliquer les transformations si nécessaire
        if self.transform:
            image = self.transform(image)  # transforms.ToTensor() convertit l'image en tenseur
        else:
            # Si aucune transformation n'est appliquée, convertir manuellement en tenseur
            image = torch.from_numpy(image).float().permute(2, 0, 1)  # [C, H, W]

        # Convertir le masque en tenseur PyTorch
        mask = torch.from_numpy(mask).long()  # [H, W]

        return image, mask

    def __len__(self):
        return len(self.image_paths)

# Créer les datasets et dataloaders
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = UrbanSegDataset(image_dir, mask_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Afficher un exemple d'image et de masque
image, mask = train_dataset[10]

# Convertir l'image en format (H, W, C) pour l'affichage
image_to_show = image.permute(1, 2, 0).numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_to_show)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Masque")
plt.axis("off")
plt.show()

"""Cellule 3 : Définition du modèle UNet"""

import torch
import torch.nn as nn

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Définir le modèle UNet
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self._block(in_channels, 64)
        self.encoder2 = self._block(64, 128)
        self.encoder3 = self._block(128, 256)
        self.encoder4 = self._block(256, 512)

        # Bottleneck
        self.bottleneck = self._block(512, 1024)

        # Decoder
        self.decoder4 = self._block(1024 + 512, 512)
        self.decoder3 = self._block(512 + 256, 256)
        self.decoder2 = self._block(256 + 128, 128)
        self.decoder1 = self._block(128 + 64, 64)

        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(e4))

        # Decoder
        d4 = self.decoder4(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(bottleneck), e4], dim=1))
        d3 = self.decoder3(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(d2), e1], dim=1))

        # Final layer
        return torch.sigmoid(self.final(d1))

# Initialiser le modèle
model = UNet().to(device)
print(model)

"""Cellule 4 : Entraînement du modèle

"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Définir la fonction de perte et l'optimiseur
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Boucle d'entraînement
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_dataloader):
        # Move data to the correct device and convert to float32
        images = images.to(device).float()  # Convert to float32
        masks = masks.to(device).float()    # Convert to float32

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.squeeze(1), masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")

# Sauvegarder le modèle
torch.save(model.state_dict(), 'C:/SUCHIES/Python/Models/unet_modelv5_10_dataset12_red.pth')


# === Paramètres globaux ===
import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.crs import CRS
from torchvision import transforms
from PIL import Image

input_folder = "C:/SUCHIES/Python/R93"
tile_output_root = "C:/SUCHIES/Python/Tuiles_Georef"
mask_output_root = "C:/SUCHIES/Python/Masques_Georef"
mosaic_output_root = "C:/SUCHIES/Python/Mosaiques/Version_dataset12/R93_20"
tile_size = 256

# === Modèle ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('C:/SUCHIES/Python/Models/unet_modelv5_20_dataset12.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

os.makedirs(tile_output_root, exist_ok=True)
os.makedirs(mask_output_root, exist_ok=True)
os.makedirs(mosaic_output_root, exist_ok=True)

# === Journal de progression ===
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
                    print(f"⏩ Tuile {tile_key} déjà traitée.")
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

                with rasterio.open(tile_filename, "w", **tile_profile) as dst:
                    dst.write(tile_data[:3])

                tile_rgb = np.moveaxis(padded_tile[:3], 0, -1)
                pil_image = Image.fromarray(tile_rgb)
                with torch.no_grad():
                    input_tensor = transform(pil_image).unsqueeze(0).to(device).float()
                    output = model(input_tensor)
                    predicted_mask = (output.squeeze().cpu().numpy() > 0.2).astype(np.uint8) * 255

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

                # 🧹 Suppression immédiate de la tuile pour libérer la mémoire
                if os.path.exists(tile_filename):
                    os.remove(tile_filename)

                tile_id += 1

        # === Création de la mosaïque des masques ===
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

        # 🧹 Suppression des masques intermédiaires après fusion
        for mask_path in mask_paths:
            if os.path.exists(mask_path):
                os.remove(mask_path)

        print(f"🎉 Mosaïque générée et masques supprimés : {mosaic_output_path}")




# === Dernière étape : fusion des mosaïques en une seule grande mosaïque ===
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
        "compress": "deflate",          # Compression sans perte
        "predictor": 2,                 # Compression prédictive (utile pour les uint8)
        "tiled": True,                  # Meilleure gestion mémoire pour les gros fichiers
        "blockxsize": 256,             # Taille des blocs (standard pour GeoTIFF)
        "blockysize": 256
    })

    final_output_path = os.path.join(mosaic_output_root, "super_mosaic_r93_20_II.tif")
    with rasterio.open(final_output_path, "w", **super_profile) as dst:
        dst.write(super_mosaic[0], 1)

    for src in sources:
        src.close()

    print(f"🎯 Mosaïque finale compressée générée : {final_output_path}")



