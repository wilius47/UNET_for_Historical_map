import os
from qgis.core import (
    QgsVectorLayer, QgsRasterLayer, QgsCoordinateTransform,
    QgsProject, QgsProcessingFeedback, QgsCoordinateReferenceSystem
)
import processing

# Paramètres utilisateur
raster_folder = "C:/SUCHIES/Python/Mosaiques/"
shapefile_folder = "C:/SUCHIES/Python/Dataset/test_21_03/Batiment_fromat_cercle"
output_image_folder = "C:/SUCHIES/Python/Test_Unet/dataset13/images"
output_mask_folder = "C:/SUCHIES/Python/Test_Unet/dataset13/masks"

tile_size = 256  # pixels

# Créer les dossiers si nécessaire
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

feedback = QgsProcessingFeedback()

raster_files = [f for f in os.listdir(raster_folder) if f.endswith(".tif")]
shapefiles = [f for f in os.listdir(shapefile_folder) if f.endswith(".shp")]

for shapefile in shapefiles:
    shape_path = os.path.join(shapefile_folder, shapefile)
    shape_layer = QgsVectorLayer(shape_path, shapefile, "ogr")
    if not shape_layer.isValid():
        print(f"❌ Shapefile invalide : {shapefile}")
        continue

    for raster_file in raster_files:
        raster_path = os.path.join(raster_folder, raster_file)
        raster_layer = QgsRasterLayer(raster_path, raster_file)

        if not raster_layer.isValid():
            print(f"❌ Raster invalide : {raster_file}")
            continue

        # Reprojeter si besoin
        shape_crs = shape_layer.crs()
        raster_crs = raster_layer.crs()
        if shape_crs != raster_crs:
            reproject_params = {
                'INPUT': shape_layer,
                'TARGET_CRS': raster_crs,
                'OUTPUT': 'memory:'
            }
            shape_layer = processing.run("native:reprojectlayer", reproject_params, feedback=feedback)['OUTPUT']

        # Rasteriser le shapefile (masque)
        rasterized_mask_path = os.path.join(output_mask_folder, f"{raster_file}_{shapefile}_mask.tif")
        rasterize_params = {
            'INPUT': shape_layer,
            'FIELD': None,
            'BURN': 1,
            'UNITS': 1,
            'WIDTH': raster_layer.width(),
            'HEIGHT': raster_layer.height(),
            'EXTENT': raster_layer.extent(),
            'NODATA': 0,
            'DATA_TYPE': 5,  # UInt16
            'OUTPUT': rasterized_mask_path
        }
        try:
            processing.run("gdal:rasterize", rasterize_params, feedback=feedback)
        except Exception as e:
            print(f"❌ Erreur rasterisation : {e}")
            continue

        # Découper le raster en tiles
        tiling_params_img = {
            'INPUT': raster_path,
            'TILE_WIDTH': tile_size,
            'TILE_HEIGHT': tile_size,
            'OVERLAP': 0,
            'OUTPUT': os.path.join(output_image_folder, f"{raster_file}_{shapefile}_tiles.gpkg"),
            'OUTPUT_DIR': output_image_folder
        }
        tiling_params_mask = {
            'INPUT': rasterized_mask_path,
            'TILE_WIDTH': tile_size,
            'TILE_HEIGHT': tile_size,
            'OVERLAP': 0,
            'OUTPUT': os.path.join(output_mask_folder, f"{raster_file}_{shapefile}_tiles.gpkg"),
            'OUTPUT_DIR': output_mask_folder
        }

        try:
            processing.run("gdal:tilesxycut", tiling_params_img, feedback=feedback)
            processing.run("gdal:tilesxycut", tiling_params_mask, feedback=feedback)
            print(f"✅ Paires image/masque générées pour : {raster_file} + {shapefile}")
        except Exception as e:
            print(f"❌ Échec du découpage en tuiles : {e}")
