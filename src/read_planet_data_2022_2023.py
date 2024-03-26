from typing import List, Tuple
import numpy as np
import datetime
import rioxarray
import rasterio
import xarray as xr
from pystac import Catalog
from osgeo import gdal
from affine import Affine

def filter_items_by_date(root_catalog: Catalog, start_date: datetime.datetime, end_date: datetime.datetime) -> List:
    """
    Filters items in a STAC catalog by a date range.

    Parameters:
    - root_catalog: Catalog, the root catalog from which to filter items.
    - start_date: datetime.datetime, the start date for the filter.
    - end_date: datetime.datetime, the end date for the filter.

    Returns:
    - List of filtered items within the specified date range.
    """
    items = list(root_catalog.get_all_items())
    filtered_items = [
        item for item in items 
        if start_date < item.datetime.replace(tzinfo=None) < end_date
    ]
    return sorted(filtered_items, key=lambda x: x.datetime)

def gdal_reader(input_arg: Tuple[str, str]) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float], str]:
    """
    Reads a satellite image using GDAL, applies a crop based on a cutline file, and resamples the image.

    Parameters:
    - input_arg: Tuple containing the path to the input file and the path to the cutline file.

    Returns:
    - Image array, the geotransform, and the coordinate reference system (CRS).
    """
    input_file, cropline_file, res = input_arg
    g = gdal.Warp('', input_file, format='MEM', resampleAlg=gdal.GRA_Average, cutlineDSName=cropline_file, cropToCutline=True, xRes=res, yRes=res, dstNodata=np.nan, outputType=gdal.GDT_Float32)
    # g = gdal.Warp('', g, format='MEM', outputBounds=[616950.0, 6881150.0, 617750.0, 6881790.0])
    geo_trans = g.GetGeoTransform()
    crs = g.GetProjection()
    return g.ReadAsArray(), geo_trans, crs

def calculate_angles(items: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates solar zenith angle (SZA), view zenith angle (VZA), and relative azimuth angle (RAA) for each item.

    Parameters:
    - items: List of catalog items.

    Returns:
    - Arrays of SZAs, VZAs, and RAAs.
    """
    szas, vzas, raas = [], [], []
    for item in items:
        sza = 90 - item.properties['view:sun_elevation']
        vza = item.properties['view:off_nadir']
        saa = item.properties['view:sun_azimuth']
        vaa = item.properties['view:azimuth']
        raa = (vaa - saa)
        szas.append(sza)
        vzas.append(vza)
        raas.append(raa)
    return np.array(szas), np.array(vzas), np.array(raas)

def create_xarray_dataarray(data_array: List[np.ndarray], angles: Tuple[np.ndarray, np.ndarray, np.ndarray], image_dates: List[datetime.datetime], geo_trans: Tuple[float, float, float, float, float, float], crs: str) -> xr.DataArray:
    """
    Creates an Xarray DataArray from the processed data.

    Parameters:
    - data_array: List of arrays containing the data for each date.
    - angles: Tuple of arrays for SZAs, VZAs, and RAAs.
    - image_dates: Dates corresponding to each data array.
    - geo_trans: Geotransform of the data.
    - crs: Coordinate reference system of the data.

    Returns:
    - Xarray DataArray with the data structured along the specified dimensions and coordinates.
    """
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08']
    da = xr.DataArray(data_array, dims=['date', 'band', 'y', 'x'], coords={'date': image_dates, 'band': band_names})
    da['sza'] = ('date', angles[0])
    da['vza'] = ('date', angles[1])
    da['raa'] = ('date', angles[2])
    da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    da.rio.write_crs(rasterio.crs.CRS.from_user_input(crs).to_epsg(), inplace=True)
    da.rio.write_transform(Affine.from_gdal(*geo_trans), inplace=True)
    return da


def read_and_process_data(data_path: str, qa_path: str, cutline_path: str) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float], str]:
    """
    Reads and processes satellite image data and the corresponding QA data.

    Parameters:
    - data_path: str, path to the satellite image data file.
    - qa_path: str, path to the QA data file.
    - cutline_path: str, path to the geojson cutline file.

    Returns:
    - Masked data array: np.ndarray, the satellite data after applying the QA mask.
    - Geotransform: Tuple containing geotransform parameters.
    - CRS: str, the coordinate reference system of the image.
    """

    input_arg = (data_path, cutline_path, resolution)
    # Read the satellite image data
    data, geo_trans, crs = gdal_reader(input_arg)
    
    input_arg = (qa_path, cutline_path, resolution)
    # Read the QA data
    qa, _, _ = gdal_reader(input_arg)
    
    mask = qa[0] == 1  
    
    # Apply the QA mask to the satellite data
    data[:, ~mask] = np.nan  # Set data to NaN where mask is False
    
    return data, geo_trans, crs

# Example usage:

# these data are ordered from the Planet data explore
# you will need to download it by yourself
planet_data_path = 'SF_2022_psscene_analytic_8b_sr_udm2'
resolution = 5
suffix = 'SF_2022'
# Load the root catalog
root_catalog = Catalog.from_file(f'{planet_data_path}/catalog.json')

# Filter items by date range
filtered_items = filter_items_by_date(root_catalog, datetime.datetime(2022, 8, 1), datetime.datetime(2022, 11, 30))

# Print ground sample distances (GSD) for filtered items
print([item.properties['gsd'] for item in filtered_items])

# Calculate angles
szas, vzas, raas = calculate_angles(filtered_items)


dats = []
for item in filtered_items:
    assets = item.assets.values()
    qa = [asset.href.replace('./', '') for asset in assets if 'udm2' in asset.href][0]
    data = [asset.href.replace('./', '') for asset in assets if ('Analytic' in asset.href) and ('tif' in asset.href)][0]
    
    qa_path = f'./{planet_data_path}/PSScene/{qa}'
    data_path = f'./{planet_data_path}/PSScene/{data}'
    data, geo_trans, crs = read_and_process_data(data_path, qa_path, 'SF_2022.geojson')
    dats.append(data)

image_dates = [item.datetime.replace(tzinfo=None).replace(microsecond=0) for item in filtered_items]

da = create_xarray_dataarray(dats, (szas, vzas, raas), image_dates, geo_trans, crs)
da = da.to_dataset(name='planet_data')

da.to_netcdf(f'planet_data_{resolution:d}m_{suffix}.nc')

dat = xr.load_dataset(f'planet_data_{resolution:d}m_{suffix}.nc')



planet_data_path = 'SF_2023_3_psscene_analytic_8b_sr_udm2'
resolution = 5
suffix = 'SF_2023'
# Load the root catalog
root_catalog = Catalog.from_file(f'{planet_data_path}/catalog.json')


# Filter items by date range
filtered_items = filter_items_by_date(root_catalog, datetime.datetime(2023, 8, 20), datetime.datetime(2023, 12, 30))

# Print ground sample distances (GSD) for filtered items
print([item.properties['gsd'] for item in filtered_items])

# Calculate angles
szas, vzas, raas = calculate_angles(filtered_items)


dats = []
for item in filtered_items:
    assets = item.assets.values()
    qa = [asset.href.replace('./', '') for asset in assets if 'udm2' in asset.href][0]
    data = [asset.href.replace('./', '') for asset in assets if ('Analytic' in asset.href) and ('tif' in asset.href)][0]
    
    qa_path = f'./{planet_data_path}/PSScene/{qa}'
    data_path = f'./{planet_data_path}/PSScene/{data}'
    data, geo_trans, crs = read_and_process_data(data_path, qa_path, 'SF_2023.geojson')
    dats.append(data)

image_dates = [item.datetime.replace(tzinfo=None).replace(microsecond=0) for item in filtered_items]

da = create_xarray_dataarray(dats, (szas, vzas, raas), image_dates, geo_trans, crs)
da = da.to_dataset(name='planet_data')

da.to_netcdf(f'planet_data_{resolution:d}m_{suffix}.nc')

dat = xr.load_dataset(f'planet_data_{resolution:d}m_{suffix}.nc')