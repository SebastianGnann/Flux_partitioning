import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import get_nearest_neighbour, plotting_fcts, curve_fcts
import xarray as xr
from functions.plotting_fcts import get_binned_range
from scipy.optimize import curve_fit

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
ds_ERA5 = xr.open_dataset(results_path + "ERA5_aggregated.nc4") # used to extract aridity for recharge datasets
df_ERA5 = pd.read_csv(results_path + "ERA5_aggregated.csv")
df_ERA5 = df_ERA5.sample(100000) # to reduce size
print("Finished ERA5.")

# Moeck
df = pd.read_csv("./results/global_groundwater_recharge_moeck-et-al.csv", sep=',')
selected_data = []
for lat, lon in zip(df['Latitude'], df['Longitude']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Latitude"]==lat, df["Longitude"]==lon)]["Groundwater recharge [mm/y]"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_Moeck = ds_combined.to_dataframe()
df_Moeck["recharge_ratio"] = df_Moeck["recharge"]/df_Moeck["tp"]
print("Finished Moeck.")

# Baseflow Budyko Baseflow
df_BB = pd.read_csv("./results/baseflow_budyko.txt", sep=',')

# fit the  distribution to the data using curve_fit
df_BB["aridity"] = df_BB["PET"]/df_BB["P"]
df_BB["baseflow_fraction"] = df_BB["Qb"]/df_BB["P"]
df_Moeck_selected = df_Moeck[["aridity_netrad", "recharge_ratio"]]
df_BB_selected = df_BB[["aridity", "baseflow_fraction"]]
df_BB_selected = df_BB_selected.rename(columns={"aridity": "aridity_netrad", "baseflow_fraction": "recharge_ratio"})
# Concatenate vertically
df_combined = pd.concat([df_Moeck_selected, df_BB_selected], ignore_index=True).dropna()

#params, covariance = curve_fit(johnsonsu_distribution, df_combined["aridity_netrad"].dropna(), df_combined["recharge_ratio"].dropna())
params, covariance = curve_fit(curve_fcts.alt_recharge_curve, df_combined["aridity_netrad"].dropna(), df_combined["recharge_ratio"].dropna())



# get ranges for water balance components
n=10000000
P = np.round(np.random.rand(n)*3000)
P[P==0] = 1e-9
PET = np.round(np.random.rand(n)*2000)
AI = PET/P
R_P = curve_fcts.Berghuijs_recharge_curve(AI)
R_P = curve_fcts.Berghuijs_recharge_curve_alt(AI)
#dist = lognorm([0.7],loc=0.05)
#R_P = dist.pdf(AI)
#R_P = gamma.pdf(AI, a=4, scale=0.2)*0.6
#R_P = johnsonsu_distribution(AI, params[0], params[1], params[2], params[3])
#R_P = Berghuijs_recharge_curve_nofit(AI, params[0], params[1])
#R_P = alt_recharge_curve(AI, params[0], params[1], params[2])
Q_P = 1-curve_fcts.Budyko_curve(AI)
E_P = curve_fcts.Budyko_curve(AI)
BFI = np.random.rand(n) # random fraction
Qb_P = BFI*Q_P
Qf_P = (1-BFI)*Q_P
Eb_P = R_P - Qb_P
Ef_P = E_P - Eb_P

wb_ok = np.full((n), True)
wb_ok[Ef_P<0] = False
wb_ok[Ef_P>E_P] = False
wb_ok[Eb_P<0] = False
wb_ok[Eb_P>E_P] = False
wb_ok[Qf_P<0] = False
wb_ok[Qf_P>Q_P] = False
wb_ok[Qb_P<0] = False
wb_ok[Qb_P>Q_P] = False

# get envelopes for plot
min_Qb_P, max_Qb_P, median_Qb_P, mean_Qb_P, bin_Qb_P = get_binned_range(AI[wb_ok], Qb_P[wb_ok], bin_edges = np.linspace(0, 5, 1001))
min_Qf_P, max_Qf_P, median_Qf_P, mean_Qf_P, bin_Qf_P = get_binned_range(AI[wb_ok], Qf_P[wb_ok], bin_edges = np.linspace(0, 5, 1001))
min_Eb_P, max_Eb_P, median_Eb_P, mean_Eb_P, bin_Eb_P = get_binned_range(AI[wb_ok], Eb_P[wb_ok], bin_edges = np.linspace(0, 5, 1001))
min_Ef_P, max_Ef_P, median_Ef_P, mean_Ef_P, bin_Ef_P = get_binned_range(AI[wb_ok], Ef_P[wb_ok], bin_edges = np.linspace(0, 5, 1001))

a = 0.25

# new plots

# plot 1
print("Budyko recharge Q")
fig = plt.figure(figsize=(5, 3), constrained_layout=True)
axes = plt.axes()
axes.fill_between(np.linspace(0.1,10,1000), 0*np.linspace(0.1,10,1000), 1-Budyko_curve(np.linspace(0.1,10,1000)), color="#0b5394", alpha=0.1)
axes.fill_between(np.linspace(0.1,10,1000),1-Budyko_curve(np.linspace(0.1,10,1000)), 1+0*np.linspace(0.1,10,1000), color="#38761D", alpha=0.1)
axes.fill_between(bin_Qb_P, min_Qb_P.statistic, max_Qb_P.statistic, color="#073763", alpha=a)
#axes.fill_between(bin_Qf_P, min_Qf_P.statistic, max_Qf_P.statistic, color="#6fa8dc", alpha=a)
im = axes.plot(AI[wb_ok], R_P[wb_ok], ".", markersize=2, c="dimgrey", label="Recharge")
im = axes.plot(bin_Qb_P, mean_Qb_P.statistic, "-", markersize=2, c="#073763", label="Q_b") #073763
#plotting_fcts.plot_lines_group(df_BB["aridity"], df_BB["baseflow_fraction"], "tab:orange", n=11, label='Gnann', statistic='mean')
#plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#a86487", n=11, label='Moeck', statistic='mean')
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
#axes.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "flux_partitioning_observations_Berghuijs.png", dpi=600, bbox_inches='tight')
plt.close()


# todo: calculate global fluxes using different relationships

# forcing grid
df_ERA5 = pd.read_csv(results_path + "ERA5_aggregated.csv")

# account for grid cell size
df_area = []
# loop over lat,lon
for x, y in zip(df_ERA5["longitude"], df_ERA5["latitude"]):
    # https://gis.stackexchange.com/questions/421231/how-can-i-calculate-the-area-of-a-5-arcminute-grid-cell-in-square-kilometers-gi
    # 1 degree of latitude = 111.567km. This varies very slightly by latitude, but we'll ignore that
    # 5 arcminutes of latitude is 1/12 of that, so 9.297km
    # 5 arcminutes of longitude is similar, but multiplied by cos(latitude) if latitude is in radians, or cos(latitude/360 * 2 * 3.14159) if in degrees
    # we have half a degree here
    y_len = 111.567/10 # 0.1 degrees /12#60/2
    x_len = y_len * np.cos(y / 360 * 2 * np.pi)
    df_area.append([x_len, y_len])
df_area = pd.DataFrame(df_area)
df_area["area"] = df_area[0] * df_area[1]
print("Total land area: ", str(df_area["area"].sum()))

df_ERA5["area"] = df_area["area"]

df_ERA5.loc[df_ERA5["aridity_netrad"]<0,"aridity_netrad"] = np.nan
#df_ERA5.loc[df_ERA5["aridity_netrad"]<0.8,"aridity_netrad"] = np.nan # as in Berghuijs paper

# todo: remove greenland and antarctica
df_ERA5.loc[df_ERA5["latitude"]<-60,"area"] = np.nan
#df_greenland = pd.read_csv(data_path + "greenland.csv", sep=',') # greenland mask for plot

# calculate fluxes
df_ERA5["Q"] = (1-curve_fcts.Budyko_curve(df_ERA5["aridity_netrad"]))*df_ERA5["tp"]
df_ERA5["E"] = (curve_fcts.Budyko_curve(df_ERA5["aridity_netrad"]))*df_ERA5["tp"]
df_ERA5["R"] = (curve_fcts.Berghuijs_recharge_curve(df_ERA5["aridity_netrad"]))*df_ERA5["tp"]

# todo: add baseflow etc

# calculate global averages
P_mean = (df_ERA5["tp"]*df_ERA5["area"]).sum()/df_ERA5["area"].sum()
netrad_mean = (df_ERA5["netrad"]*df_ERA5["area"]).sum()/df_ERA5["area"].sum()
Q_mean = (df_ERA5["Q"]*df_ERA5["area"]).sum()/df_ERA5["area"].sum()
E_mean = (df_ERA5["E"]*df_ERA5["area"]).sum()/df_ERA5["area"].sum()
R_mean = (df_ERA5["R"]*df_ERA5["area"]).sum()/df_ERA5["area"].sum()

