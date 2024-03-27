import geopandas as gpd
import pandas as pd

year = 2023
# script to create parsed json file
# 'SF_{year}_samples_corrected_parsed.geojson'
# from f'SF_{year}_samples_corrected.geojson'
# and f'SF_{year}_ground_data.xlsx'


df = gpd.read_file(f'SF_{year}_samples_corrected.geojson')
features = []
for i in range(5):
    for j in range(5):
        plot_name = f'Plot{i+1}'
        mask = df[plot_name].notnull()
        subplot_name = f'Subplot {j+1}'
        mask = mask & (df[plot_name] == subplot_name)
        if mask.sum() != 1:
            print(f'No data for {plot_name} {subplot_name}')
        sub_df = df[mask]
        geometry = sub_df.geometry
        # create a feature with this geometry with "Name": "P1S2"
        Name = f'P{i+1}S{j+1}'
        feature = gpd.GeoDataFrame({'Name': [Name], 'geometry': geometry})
        features.append(feature)
gdf = gpd.GeoDataFrame(pd.concat(features, ignore_index=True))
# gdf.to_file('SF_2023_samples_corrected_parsed.geojson', driver='GeoJSON')

df = pd.ExcelFile(f'SF_{year}_ground_data.xlsx').parse('LAI')

import datetime
dates = df.iloc[:, 2:].columns
dates = [datetime.datetime.strptime(date, '%d.%m.%Y') for date in dates]
dates = [date.strftime('%Y%m%d') for date in dates]

import pylab as plt

lais = []
cabs = []
for name in gdf.Name:
    mask = df['Subplot NO'] == name
    if mask.sum() != 1:
        print(f'No data for {name}')
    sub_df = df[mask]
    dat = sub_df.iloc[0, 2:].values.astype(float)
    plt.plot(dates, dat, label=name)
    lais.append(str(dat.tolist()))
    
    # sub_gdf['measurement_dates'] = str(dates)
    # gdf[gdf.Name == name] = sub_gdf
gdf['LAI_measurement'] = lais

df = pd.ExcelFile(f'SF_{year}_ground_data.xlsx').parse('Chl')

import datetime
dates = df.iloc[:, 2:].columns
dates = [datetime.datetime.strptime(date, '%d.%m.%Y') for date in dates]
dates = [date.strftime('%Y%m%d') for date in dates]

import pylab as plt


cabs = []
for name in gdf.Name:
    mask = df['Subplot NO'] == name
    if mask.sum() != 1:
        print(f'No data for {name}')
    sub_df = df[mask]
    dat = sub_df.iloc[0, 2:].values.astype(float)
    plt.plot(dates, dat, label=name)
    cabs.append(str(dat.tolist()))
    
    # sub_gdf['measurement_dates'] = str(dates)
    # gdf[gdf.Name == name] = sub_gdf
gdf['Cab_measurement'] = cabs


gdf['measurement_dates'] = [str(dates)] * len(lais)
gdf.to_file(f'SF_{year}_samples_corrected_parsed.geojson', driver='GeoJSON')