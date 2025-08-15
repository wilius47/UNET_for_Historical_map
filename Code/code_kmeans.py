import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from sklearn.cluster import KMeans

# === Dossiers ===
input_folder = "C:/SUCHIES/Python/All jp2"
mask_output_root = "C:/SUCHIES/Python/Other algo/Masques_KMeans"
mosaic_output_root = "C:/SUCHIES/Python/Other algo/Mosaïque_KMeans"
tile_size = 256

os.makedirs(mask_output_root, exist_ok=True)
os.makedirs(mosaic_output_root, exist_ok=True)

jp2_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jp2")]

for jp2_file in jp2_files:
    input_jp2_path = os.path.join(input_folder, jp2_file)
    base_name = os.path.splitext(jp2_file)[0]

    with rasterio.open(input_jp2_path) as src:
        width, height = src.width, src.height
        profile = src.profile
        transform_global = src.transform
        crs = src.crs

        print(f"\n📂 Traitement KMeans : {jp2_file} - {width}x{height}")

        mask_paths = []
        tile_id = 0

        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                win_width = min(tile_size, width - left)
                win_height = min(tile_size, height - top)
                if win_width <= 0 or win_height <= 0:
                    continue

                window = Window(left, top, win_width, win_height)
                tile_data = src.read(window=window)[:3]  # prendre 3 premières bandes
                tile_rgb = np.moveaxis(tile_data, 0, -1)  # HWC

                # Flatten pour KMeans
                flat_pixels = tile_rgb.reshape(-1, 3)

                # KMeans
                kmeans = KMeans(n_clusters=2, random_state=42).fit(flat_pixels)
                labels = kmeans.labels_.reshape(win_height, win_width)
                mask = (labels == labels.max()).astype(np.uint8) * 255  # cluster principal = 255

                mask_profile = profile.copy()
                mask_profile.update({
                    "height": win_height,
                    "width": win_width,
                    "transform": transform_global * rasterio.Affine.translation(left, top),
                    "crs": crs,
                    "count": 1,
                    "dtype": "uint8"
                })

                mask_filename = os.path.join(mask_output_root, f"{base_name}_mask_{tile_id:05d}.tif")
                with rasterio.open(mask_filename, "w", **mask_profile) as dst:
                    dst.write(mask, 1)

                mask_paths.append(mask_filename)
                tile_id += 1
                print(f"✅ Tuile {tile_id} traitée")

        # Fusion des masques
        print(f"🧩 Fusion des {len(mask_paths)} masques pour {base_name}...")
        if mask_paths:
            sources = [rasterio.open(p) for p in mask_paths]
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

            for src_tile in sources:
                src_tile.close()

            # Supprimer masques intermédiaires
            for mask_path in mask_paths:
                if os.path.exists(mask_path):
                    os.remove(mask_path)

            print(f"🎉 Mosaïque KMeans générée : {mosaic_output_path}")
        else:
            print(f"⚠️ Aucun masque généré pour {base_name}")
