from osgeo import gdal
import os

# This script resamples and aligns different rasters.

data_path = r"D:/Data/"
results_path = r"D:/Data/resampling/"

if not os.path.isdir(results_path):
    os.makedirs(results_path)

# 5 minute resolution
res = 5 / 60
# res = 0.5/60

bounds = [-180, -90, 180 - res, 90]

path_list = ["CHELSA/CHELSA_bio12_1981-2010_V.2.1.tif",
             "CHELSA/CHELSA_pet_penman_mean_1981-2010_V.2.1.tif",
             "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_12.tif",
             "WorldClim/7504448/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/et0_v3_yr.tif"]

name_list = ["P_CHELSA",
             "PET_CHELSA",
             "P_WorldClim",
             "PET_WorldClim"]

for path, name in zip(path_list, name_list):
    print(name)
    ds_path = data_path + path
    ds = gdal.Open(ds_path)
    dsRes = gdal.Warp(results_path + name + "_5min.tif", ds,
                      outputBounds=bounds, xRes=res, yRes=res, resampleAlg="med", dstSRS="EPSG:4326")
