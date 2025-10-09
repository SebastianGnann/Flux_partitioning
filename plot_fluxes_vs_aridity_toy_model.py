import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import get_nearest_neighbour, plotting_fcts, curve_fcts
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
from scipy.spatial import cKDTree
from functions.plotting_fcts import get_binned_range
from scipy.optimize import curve_fit

# This script plots Budyko-like relationships for different fluxes using a toy model.

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

### load data ###

# climate data
df = pd.read_csv(results_path + "combined_chelsa_5min.csv")
#df = pd.read_csv(results_path + "combined_worldclim_5min.csv")

# Moeck
df_tmp = pd.read_csv("./results/global_groundwater_recharge_moeck-et-al.csv", sep=',')
tree = cKDTree(df[['latitude', 'longitude']])
distances, indices = tree.query(df_tmp[['Latitude', 'Longitude']])
df_Moeck = df.iloc[indices].copy()
df_Moeck['recharge'] = df_tmp["Groundwater recharge [mm/y]"].values
df_Moeck['recharge_ratio'] = df_Moeck['recharge'] / df_Moeck['P']
print("Finished Moeck.")

# Baseflow Budyko Baseflow
df_BB = pd.read_csv("./results/baseflow_budyko.txt", sep=',')

# Fit the distribution to the data using curve_fit
df_BB["aridity"] = df_BB["PET"] / df_BB["P"]
df_BB["baseflow_fraction"] = df_BB["Qb"] / df_BB["P"]
df_Moeck_selected = df_Moeck[["ai", "recharge_ratio"]]
df_BB_selected = df_BB[["aridity", "baseflow_fraction"]]
df_BB_selected = df_BB_selected.rename(columns={"aridity": "ai", "baseflow_fraction": "recharge_ratio"})
df_combined = pd.concat([df_Moeck_selected, df_BB_selected], ignore_index=True).dropna()

# get ranges for water balance components
n = 10000000
P = np.round(np.random.rand(n) * 3000)
P[P == 0] = 1e-9
PET = np.round(np.random.rand(n) * 2000)
AI = PET / P
R_P = curve_fcts.Berghuijs_recharge_curve(AI)
# R_P = curve_fcts.Berghuijs_recharge_curve_alt(AI)
Q_P = 1 - curve_fcts.Budyko_curve(AI)
E_P = curve_fcts.Budyko_curve(AI)
BFI = np.random.rand(n)  # random fraction
Qb_P = BFI * Q_P
Qf_P = (1 - BFI) * Q_P
Eb_P = R_P - Qb_P
Ef_P = E_P - Eb_P

wb_ok = np.full((n), True)
wb_ok[Ef_P < 0] = False
wb_ok[Ef_P > E_P] = False
wb_ok[Eb_P < 0] = False
wb_ok[Eb_P > E_P] = False
wb_ok[Qf_P < 0] = False
wb_ok[Qf_P > Q_P] = False
wb_ok[Qb_P < 0] = False
wb_ok[Qb_P > Q_P] = False

# get envelopes for plot
min_Qb_P, max_Qb_P, median_Qb_P, mean_Qb_P, bin_Qb_P = get_binned_range(AI[wb_ok], Qb_P[wb_ok],
                                                                        bin_edges=np.linspace(0, 5, 1001))
min_Qf_P, max_Qf_P, median_Qf_P, mean_Qf_P, bin_Qf_P = get_binned_range(AI[wb_ok], Qf_P[wb_ok],
                                                                        bin_edges=np.linspace(0, 5, 1001))
min_Eb_P, max_Eb_P, median_Eb_P, mean_Eb_P, bin_Eb_P = get_binned_range(AI[wb_ok], Eb_P[wb_ok],
                                                                        bin_edges=np.linspace(0, 5, 1001))
min_Ef_P, max_Ef_P, median_Ef_P, mean_Ef_P, bin_Ef_P = get_binned_range(AI[wb_ok], Ef_P[wb_ok],
                                                                        bin_edges=np.linspace(0, 5, 1001))

# make plot
a = 0.25
print("Berghuijs recharge curve")
fig = plt.figure(figsize=(5, 3), constrained_layout=True)
axes = plt.axes()
axes.fill_between(np.linspace(0.1, 10, 1000), 0 * np.linspace(0.1, 10, 1000),
                  1 - curve_fcts.Budyko_curve(np.linspace(0.1, 10, 1000)), color="#0b5394", alpha=0.1)
axes.fill_between(np.linspace(0.1, 10, 1000), 1 - curve_fcts.Budyko_curve(np.linspace(0.1, 10, 1000)),
                  1 + 0 * np.linspace(0.1, 10, 1000), color="#38761D", alpha=0.1)
axes.fill_between(bin_Qb_P, min_Qb_P.statistic, max_Qb_P.statistic, color="#073763", alpha=a)
# axes.fill_between(bin_Qf_P, min_Qf_P.statistic, max_Qf_P.statistic, color="#6fa8dc", alpha=a)
im = axes.plot(AI[wb_ok], R_P[wb_ok], ".", markersize=2, c="dimgrey", label="Recharge")
im = axes.plot(bin_Qb_P, mean_Qb_P.statistic, "-", markersize=2, c="#073763", label="Q_b")  # 073763
# plotting_fcts.plot_lines_group(df_BB["aridity"], df_BB["baseflow_fraction"], "tab:orange", n=11, label='Gnann', statistic='mean')
# plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#a86487", n=11, label='Moeck', statistic='mean')
axes.set_xlabel("Climatic aridity (PET/P) [-]")
axes.set_ylabel("Flux ratio (Flux/P) [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
# axes.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "flux_partitioning_toy_model_Berghuijs.png", dpi=600, bbox_inches='tight')
#plt.savefig(figures_path + "flux_partitioning_toy_model_Berghuijs.pdf", bbox_inches='tight')
plt.close()

# get ranges for water balance components
n = 10000000
P = np.round(np.random.rand(n) * 3000)
P[P == 0] = 1e-9
PET = np.round(np.random.rand(n) * 2000)
AI = PET / P
# R_P = curve_fcts.Berghuijs_recharge_curve(AI)
R_P = curve_fcts.Berghuijs_recharge_curve_alt(AI)
Q_P = 1 - curve_fcts.Budyko_curve(AI)
E_P = curve_fcts.Budyko_curve(AI)
BFI = np.random.rand(n)  # random fraction
Qb_P = BFI * Q_P
Qf_P = (1 - BFI) * Q_P
Eb_P = R_P - Qb_P
Ef_P = E_P - Eb_P

wb_ok = np.full((n), True)
wb_ok[Ef_P < 0] = False
wb_ok[Ef_P > E_P] = False
wb_ok[Eb_P < 0] = False
wb_ok[Eb_P > E_P] = False
wb_ok[Qf_P < 0] = False
wb_ok[Qf_P > Q_P] = False
wb_ok[Qb_P < 0] = False
wb_ok[Qb_P > Q_P] = False

# get envelopes for plot
min_Qb_P, max_Qb_P, median_Qb_P, mean_Qb_P, bin_Qb_P = get_binned_range(AI[wb_ok], Qb_P[wb_ok],
                                                                        bin_edges=np.linspace(0, 5, 1001))
min_Qf_P, max_Qf_P, median_Qf_P, mean_Qf_P, bin_Qf_P = get_binned_range(AI[wb_ok], Qf_P[wb_ok],
                                                                        bin_edges=np.linspace(0, 5, 1001))
min_Eb_P, max_Eb_P, median_Eb_P, mean_Eb_P, bin_Eb_P = get_binned_range(AI[wb_ok], Eb_P[wb_ok],
                                                                        bin_edges=np.linspace(0, 5, 1001))
min_Ef_P, max_Ef_P, median_Ef_P, mean_Ef_P, bin_Ef_P = get_binned_range(AI[wb_ok], Ef_P[wb_ok],
                                                                        bin_edges=np.linspace(0, 5, 1001))

# make plot
a = 0.25
print("Alternative recharge curve")
fig = plt.figure(figsize=(5, 3), constrained_layout=True)
axes = plt.axes()
axes.fill_between(np.linspace(0.1, 10, 1000), 0 * np.linspace(0.1, 10, 1000),
                  1 - curve_fcts.Budyko_curve(np.linspace(0.1, 10, 1000)), color="#0b5394", alpha=0.1)
axes.fill_between(np.linspace(0.1, 10, 1000), 1 - curve_fcts.Budyko_curve(np.linspace(0.1, 10, 1000)),
                  1 + 0 * np.linspace(0.1, 10, 1000), color="#38761D", alpha=0.1)
axes.fill_between(bin_Qb_P, min_Qb_P.statistic, max_Qb_P.statistic, color="#073763", alpha=a)
# axes.fill_between(bin_Qf_P, min_Qf_P.statistic, max_Qf_P.statistic, color="#6fa8dc", alpha=a)
im = axes.plot(AI[wb_ok], R_P[wb_ok], ".", markersize=2, c="dimgrey", label="Recharge")
im = axes.plot(bin_Qb_P, mean_Qb_P.statistic, "-", markersize=2, c="#073763", label="Q_b")  # 073763
# plotting_fcts.plot_lines_group(df_BB["aridity"], df_BB["baseflow_fraction"], "tab:orange", n=11, label='Gnann', statistic='mean')
# plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#a86487", n=11, label='Moeck', statistic='mean')
axes.set_xlabel("Climatic aridity (PET/P) [-]")
axes.set_ylabel("Flux ratio (Flux/P) [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
# axes.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "flux_partitioning_toy_model_Berghuijs_alternative.png", dpi=600, bbox_inches='tight')
#plt.savefig(figures_path + "flux_partitioning_toy_model_Berghuijs_alternative.pdf", bbox_inches='tight')
plt.close()