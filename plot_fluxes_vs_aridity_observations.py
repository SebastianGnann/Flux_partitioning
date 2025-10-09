import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import get_nearest_neighbour, plotting_fcts, curve_fcts
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
from scipy.spatial import cKDTree

# This script loads and analyses different datasets in Budyko space.

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)


### load data ###

# load climate data
def load_raster(file_path):
    with rasterio.open(file_path) as src:
        crs = src.crs
        data = src.read(1)
        transform = src.transform
        height, width = data.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
        lon = np.array(xs)
        lat = np.array(ys)
        if crs.is_geographic and crs.to_epsg() == 4326:
            if np.nanmean(lon) > 180:
                lon = np.where(lon > 180, lon - 360, lon)
        return lon, lat, data.flatten()


lon_p, lat_p, p_data = load_raster('D:/Data/resampling/P_Chelsa_5min.tif')
lon_pet, lat_pet, pet_data = load_raster('D:/Data/resampling/PET_Chelsa_5min.tif')

df = pd.DataFrame({
    'longitude': lon_p,
    'latitude': lat_p,
    'P': p_data,
    'PET': pet_data
})

# save df
# CHELSA
df.loc[df["P"] > 50000, "P"] = np.nan  # 63920
df.loc[df["PET"] > 50000, "PET"] = np.nan
df['P'] = df['P'] * 0.1
df['PET'] = df['PET'] * 0.01 * 12
# WorldClim
# remove all negative values for P and PET
# df = df[df['P'] >= 0]
# df = df[df['PET'] >= 0]
df['ai'] = df['PET'] / df['P']
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df.to_csv(results_path + "combined_chelsa_5min.csv", index=False)
print("Finished processing CHELSA data.")

# load df
df = pd.read_csv(results_path + "combined_chelsa_5min.csv")
#df = pd.read_csv(results_path + "combined_worldclim_5min.csv")

# plot map of aridity index as a check
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df['longitude'], df['latitude'], c=df['ai'], cmap='viridis', marker='s', s=0.02, vmin=0, vmax=2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(sc, cax=cax, label='Climatic aridity [-]')
plt.savefig(figures_path + "aridity_Chelsa.png", dpi=600, bbox_inches='tight')
plt.close()

# Moeck
df_tmp = pd.read_csv("./results/global_groundwater_recharge_moeck-et-al.csv", sep=',')
tree = cKDTree(df[['latitude', 'longitude']])
distances, indices = tree.query(df_tmp[['Latitude', 'Longitude']])
df_Moeck = df.iloc[indices].copy()
df_Moeck['recharge'] = df_tmp["Groundwater recharge [mm/y]"].values
df_Moeck['recharge_ratio'] = df_Moeck['recharge'] / df_Moeck['P']
print("Finished Moeck.")

# MacDonald
df_tmp = pd.read_csv("./results/Recharge_data_Africa_BGS.csv", sep=';')
tree = cKDTree(df[['latitude', 'longitude']])
distances, indices = tree.query(df_tmp[['Lat', 'Long']])
df_MacDonald = df.iloc[indices].copy()
df_MacDonald['recharge'] = df_tmp["Recharge_mmpa"].values
df_MacDonald['recharge_ratio'] = df_MacDonald['recharge'] / df_MacDonald['P']
print("Finished MacDonald.")

# Lee
df_tmp = pd.read_csv("./results/dat07_u.csv", sep=',')
tree = cKDTree(df[['latitude', 'longitude']])
distances, indices = tree.query(df_tmp[['lat', 'lon']])
df_Lee = df.iloc[indices].copy()
df_Lee['recharge'] = df_tmp["Recharge mean mm/y"].values
df_Lee['recharge_ratio'] = df_Lee['recharge'] / df_Lee['P']
print("Finished Lee.")

# Baseflow Budyko Baseflow
# Note that the baseflow data use different aridity data (because they require catchment averages).
df_Gnann = pd.read_csv("./results/baseflow_budyko.txt", sep=',')

### plot data ###
stat = "median"
print("Plotting flux partitioning in Budyko space.")
fig = plt.figure(figsize=(5.5, 4), constrained_layout=True)
axes = plt.axes()
axes.fill_between(np.linspace(0.1, 10, 1000), 0 * np.linspace(0.1, 10, 1000),
                  1 - curve_fcts.Budyko_curve(np.linspace(0.1, 10, 1000)), color="#0b5394", alpha=0.1)
axes.fill_between(np.linspace(0.1, 10, 1000), 1 - curve_fcts.Budyko_curve(np.linspace(0.1, 10, 1000)),
                  1 + 0 * np.linspace(0.1, 10, 1000), color="#38761D", alpha=0.1)
# im = axes.scatter(df_Lee["ai"], df_Lee["recharge_ratio"], s=2.5, c="darkgrey", alpha=0.1, lw=0)
# im = axes.scatter(df_Moeck["ai"], df_Moeck["recharge_ratio"], s=2.5, c="dimgrey", alpha=0.5, lw=0)
# im = axes.scatter(df_MacDonald["ai"], df_MacDonald["recharge_ratio"], s=2.5, c="black", alpha=0.5, lw=0)
plotting_fcts.plot_lines_group(df_Lee["ai"], df_Lee["recharge_ratio"], "darkgrey", n=11, label='Lee', statistic=stat,
                               uncertainty=True)
plotting_fcts.plot_lines_group(df_Moeck["ai"], df_Moeck["recharge_ratio"], "dimgrey", n=11, label='Moeck',
                               statistic=stat, uncertainty=True)
plotting_fcts.plot_lines_group(df_MacDonald["ai"], df_MacDonald["recharge_ratio"], "black", n=11, label='MacDonald',
                               statistic=stat, uncertainty=True)
plotting_fcts.plot_lines_group(df_Gnann["PET"] / df_Gnann["P"], df_Gnann["Qb"] / df_Gnann["P"], "#073763", n=11,
                               label='Q_b', statistic=stat, uncertainty=True)
# im = axes.plot(np.linspace(0.1, 10, 1000), curve_fcts.Berghuijs_recharge_curve(np.linspace(0.1, 10, 1000)), "-",
#               c="grey", alpha=0.75, label='Berghuijs')
axes.set_xlabel("Climatic aridity (PET/P) [-]")
axes.set_ylabel("Flux ratio (Flux/P) [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
#axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "flux_partitioning_observations.png", dpi=600, bbox_inches='tight')
#plt.savefig(figures_path + "flux_partitioning_observations.pdf", bbox_inches='tight')
plt.close()

# plot point cloud
stat = "median"
print("Plotting flux partitioning in Budyko space.")
fig = plt.figure(figsize=(5.5, 4), constrained_layout=True)
axes = plt.axes()
axes.fill_between(np.linspace(0.1, 10, 1000), 0 * np.linspace(0.1, 10, 1000),
                  1 - curve_fcts.Budyko_curve(np.linspace(0.1, 10, 1000)), color="#0b5394", alpha=0.1)
axes.fill_between(np.linspace(0.1, 10, 1000), 1 - curve_fcts.Budyko_curve(np.linspace(0.1, 10, 1000)),
                  1 + 0 * np.linspace(0.1, 10, 1000), color="#38761D", alpha=0.1)
im = axes.scatter(df_Lee["ai"], df_Lee["recharge_ratio"], s=2.5, c="darkgrey", alpha=0.1, lw=0)
im = axes.scatter(df_Moeck["ai"], df_Moeck["recharge_ratio"], s=2.5, c="dimgrey", alpha=0.5, lw=0)
im = axes.scatter(df_MacDonald["ai"], df_MacDonald["recharge_ratio"], s=2.5, c="black", alpha=0.5, lw=0)
im = axes.scatter(df_Gnann["PET"] / df_Gnann["P"], df_Gnann["Qb"] / df_Gnann["P"], s=2.5, c="#073763", alpha=0.5, lw=0)
plotting_fcts.plot_lines_group(df_Lee["ai"], df_Lee["recharge_ratio"], "darkgrey", n=11, label='Lee', statistic=stat,
                               uncertainty=False)
plotting_fcts.plot_lines_group(df_Moeck["ai"], df_Moeck["recharge_ratio"], "dimgrey", n=11, label='Moeck',
                               statistic=stat, uncertainty=False)
plotting_fcts.plot_lines_group(df_MacDonald["ai"], df_MacDonald["recharge_ratio"], "black", n=11, label='MacDonald',
                               statistic=stat, uncertainty=False)
plotting_fcts.plot_lines_group(df_Gnann["PET"] / df_Gnann["P"], df_Gnann["Qb"] / df_Gnann["P"], "#073763", n=11,
                               label='Q_b', statistic=stat, uncertainty=False)
# im = axes.plot(np.linspace(0.1, 10, 1000), curve_fcts.Berghuijs_recharge_curve(np.linspace(0.1, 10, 1000)), "-",
#               c="grey", alpha=0.75, label='Berghuijs')
axes.set_xlabel("Climatic aridity (PET/P) [-]")
axes.set_ylabel("Flux ratio (Flux/P) [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
#axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "flux_partitioning_observations_scatter.png", dpi=600, bbox_inches='tight')
#plt.savefig(figures_path + "flux_partitioning_observations_scatter.pdf", bbox_inches='tight')
plt.close()