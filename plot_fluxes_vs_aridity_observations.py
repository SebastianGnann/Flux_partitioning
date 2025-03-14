import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import get_nearest_neighbour, plotting_fcts, curve_fcts
import xarray as xr

# This script loads and analyses different datasets in Budyko space.

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

### load data ###

# ERA5
#ds_ERA5 = xr.open_dataset(results_path + "ERA5_aggregated.nc4")  # used to extract aridity for recharge datasets
#df_ERA5 = pd.read_csv(results_path + "ERA5_aggregated.csv")
#df_ERA5 = df_ERA5.sample(100000)  # to reduce size
#print("Finished ERA5.")

# load climate data
# world clim

# CHELSA as alternative

# Moeck
df = pd.read_csv("./results/global_groundwater_recharge_moeck-et-al.csv", sep=',')
selected_data = []
for lat, lon in zip(df['Latitude'], df['Longitude']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')  # ['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Latitude"] == lat, df["Longitude"] == lon)]["Groundwater recharge [mm/y]"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_Moeck = ds_combined.to_dataframe()
df_Moeck["recharge_ratio"] = df_Moeck["recharge"] / df_Moeck["tp"]
print("Finished Moeck.")

# MacDonald
df = pd.read_csv("./results/Recharge_data_Africa_BGS.csv", sep=';')
selected_data = []
for lat, lon in zip(df['Lat'], df['Long']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')  # ['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Lat"] == lat, df["Long"] == lon)]["Recharge_mmpa"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_MacDonald = ds_combined.to_dataframe()
df_MacDonald["recharge_ratio"] = df_MacDonald["recharge"] / df_MacDonald["tp"]
print("Finished MacDonald.")

'''
# Lee
# HESS preprint
# df_Lee2 = pd.read_csv("./results/dat07_u.csv", sep=',')
df = pd.read_csv("./results/dat07_u.csv", sep=',')
selected_data = []
for lat, lon in zip(df['lat'], df['lon']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')  # ['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["lat"] == lat, df["lon"] == lon)]["Recharge mean mm/y"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_Lee = ds_combined.to_dataframe()
df_Lee["recharge_ratio"] = df_Lee["recharge"] / df_Lee["tp"]
print("Finished Lee.")
'''

# Baseflow Budyko Baseflow
# note that this uses different aridity data and it is not straightforward to change this because these are not
# catchment averaged
df_Gnann = pd.read_csv("./results/baseflow_budyko.txt", sep=',')

### plot data ###

stat = "median"

### NEW PLOT ###
print("Budyko recharge alternative")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
axes.fill_between(np.linspace(0.1, 10, 1000), 0 * np.linspace(0.1, 10, 1000),
                  1 - curve_fcts.Budyko_curve(np.linspace(0.1, 10, 1000)), color="#0b5394", alpha=0.1)
axes.fill_between(np.linspace(0.1, 10, 1000), 1 - Budyko_curve(np.linspace(0.1, 10, 1000)),
                  1 + 0 * np.linspace(0.1, 10, 1000), color="#38761D", alpha=0.1)
# im = axes.scatter(df_Caravan["aridity_netrad"], df_Caravan["BFI"]*df_Caravan["TotalRR"], s=2.5, c="#073763", alpha=0.1, lw=0)
# im = axes.scatter(df_Lee["aridity_netrad"], df_Lee["recharge_ratio"], s=2.5, c="darkgrey", alpha=0.1, lw=0)
# im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=2.5, c="dimgrey", alpha=0.5, lw=0)
# im = axes.scatter(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], s=2.5, c="black", alpha=0.5, lw=0)
#plotting_fcts.plot_lines_group(df_Lee["aridity_netrad"], df_Lee["recharge_ratio"], "darkgrey", n=11, label='Lee',
#                               statistic=stat, uncertainty=True)
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "dimgrey", n=11, label='Moeck',
                               statistic=stat, uncertainty=True)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "black", n=11,
                               label='MacDonald', statistic=stat, uncertainty=True)
# im = axes.plot(np.linspace(0.1,10,1000), Berghuijs_recharge_curve(np.linspace(0.1,10,1000)), "-", c="grey", alpha=0.75, label='Berghuijs')
plotting_fcts.plot_lines_group(df_Gnann["PET"] / df_Gnann["P"], df_Gnann["Qb"] / df_Gnann["P"], "#073763", n=11, label='Q_b',
                               statistic=stat, uncertainty=True)
axes.set_xlabel("Climatic aridity (PET/P) [-]")
axes.set_ylabel("Flux ratio (Flux/P) [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "flux_partitioning_observations.png", dpi=600, bbox_inches='tight')
plt.close()
