from pystac import Catalog
root_catalog = Catalog.from_file('SF_2023_3_psscene_analytic_8b_sr_udm2/catalog.json')
root_catalog

import datetime
items = list(root_catalog.get_all_items())
items = [item for item in items if (item.datetime.replace(tzinfo=None) >datetime.datetime(2023, 8, 20) and item.datetime.replace(tzinfo=None) < datetime.datetime(2023, 12, 30))]
items = sorted(items, key=lambda x: x.datetime)

print([item.properties['gsd'] for item in items])

from osgeo import gdal
import numpy as np
from typing import Tuple, List, Dict
def gdal_reader(input_arg: Tuple[str, str]) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float], str]:
    """
    Reads a satellite image using GDAL, applies a crop based on a cutline file, and resamples the image.

    Parameters:
    - input_arg: Tuple[str, str], where the first element is the path to the input file and the second is the path to the cutline file.

    Returns:
    - A tuple containing the image array, the geotransform, and the coordinate reference system (CRS).
    """
    input_file, cropline_file = input_arg
    g = gdal.Warp('', input_file, format='MEM', resampleAlg=gdal.GRA_Average, cutlineDSName=cropline_file, cropToCutline=True, xRes=5, yRes=5, dstNodata=np.nan, outputType=gdal.GDT_Float32)
    geo_trans = g.GetGeoTransform()
    crs = g.GetProjection()
    return g.ReadAsArray(), geo_trans, crs

dats = []
szas = []
vzas = []
raas = []

for item in items:
    sza = 90 - item.properties['view:sun_elevation']
    vza = item.properties['view:off_nadir']
    saa = item.properties['view:sun_azimuth']
    vaa = item.properties['view:azimuth']
    raa = (vaa - saa)
    
    szas.append(sza)
    vzas.append(vza)
    raas.append(raa)

    assets = item.assets.values()
    qa = [asset.href.replace('./', '') for asset in assets if 'udm2' in asset.href][0]
    data = [asset.href.replace('./', '') for asset in assets if ('Analytic' in asset.href) and ('tif' in asset.href)][0]
    
    qa_path = f'./SF_2023_3_psscene_analytic_8b_sr_udm2/PSScene/{qa}'
    data_path = f'./SF_2023_3_psscene_analytic_8b_sr_udm2/PSScene/{data}'
    # print(data_path, qa_path)
    data, geoTrans, crs = gdal_reader((data_path, 'SF_2023.geojson'))
    qa, _, _ = gdal_reader((qa_path, 'SF_2023.geojson'))
    mask = qa[0] == 1
    data[:, ~mask] = np.nan
    dats.append(data)
    # break

szas = np.array(szas)
vzas = np.array(vzas)
raas = np.array(raas)

import rioxarray
import xarray as xr
import rasterio
from affine import Affine

def create_xarray_dataarray(data_array: List[np.ndarray], angles: List[np.ndarray], image_dates: List[datetime.datetime], geo_trans: Tuple[float, float, float, float, float, float], crs: str) -> xr.DataArray:
    """
    Creates an Xarray DataArray from the processed data.

    Parameters:
    - data_array: List[np.ndarray], a list of arrays containing the data for each date.
    - image_dates: List[datetime.datetime], the dates corresponding to each data array.
    - geo_trans: Tuple[float, float, float, float, float, float], the geotransform of the data.
    - crs: str, the coordinate reference system of the data.

    Returns:
    - An Xarray DataArray with the data structured along the specified dimensions and coordinates.
    """
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08']
    da = xr.DataArray(data_array, dims=['date', 'band', 'y', 'x'], coords={'date': image_dates, 'band': band_names})
    # write angles to the DataArray
    da['sza'] = ('date', angles[0])
    da['vza'] = ('date', angles[1])
    da['raa'] = ('date', angles[2])
    # da['geoTrans'] = geo_trans
    # set spatial dimensions and write CRS and geotransform
    da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    cs = rasterio.crs.CRS.from_user_input(crs).to_epsg()
    transform = Affine.from_gdal(*geo_trans)
    da.rio.write_crs(cs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_grid_mapping(inplace=True)
    return da

image_dates = [item.datetime.replace(tzinfo=None).replace(microsecond=0) for item in items]

da = create_xarray_dataarray(dats, [szas, vzas, raas], image_dates, geoTrans, crs)

da = da.to_dataset(name='planet_data')
# save to netCDF
da.to_netcdf('planet_data.nc')

# # load from netCDF
dat = xr.load_dataset('planet_data.nc')
