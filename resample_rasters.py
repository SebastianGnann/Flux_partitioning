from osgeo import gdal
import os

# This script resamples and aligns different rasters.

data_path = r"D:/Data/"
results_path = r"D:/Data/resampling/"

if not os.path.isdir(results_path):
    os.makedirs(results_path)

bounds = [-180, -90, 180, 90]

# 5 minute resolution
res = 5 / 60
# res = 0.5/60

path_list = ["7504448/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/ai_v3_yr.tif",
             "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_12.tif",
             "WorldClim/7504448/global-et0_annual.tif/et0_yr/et0_yr.tif"]
name_list = ["ai_v3_yr",
             "P_WorldClim",
             "PET_WorldClim"]

for path, name in zip(path_list, name_list):
    print(name)
    ds_path = data_path + path
    ds = gdal.Open(ds_path)
    dsRes = gdal.Warp(results_path + name + "_5min.tif", ds,
                      outputBounds=bounds, xRes=res, yRes=res, resampleAlg="med", dstSRS="EPSG:4326")
