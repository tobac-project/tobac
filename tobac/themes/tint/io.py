""" X-Array based TINT I/O module. """

import xarray as xr
import random
import numpy as np
import pyproj

from .grid_utils import add_lat_lon_grid
from datetime import datetime

def load_cfradial_grids(file_list):
    ds = xr.open_mfdataset(file_list)
    # Check for CF/Radial conventions
    if not ds.attrs["Conventions"] == 'CF/Radial instrument_parameters':
        ds.close()
        raise IOError("TINT module is only compatible with CF/Radial files!")
    ds = add_lat_lon_grid(ds)
    ds.attrs["cf_tree_order"] = "storm_id cell_id"
    ds.attrs["tree_id"] = "%d" % random.randint(a=0, b=65535)
    # Try to detect number of fields (4D arrays)
    nfields = 0
    for keys in ds.variables.keys():
        if(len(ds[keys].dims) == 3):
            nfields += 1

    ds['storm_id'] = ('storm', np.arange(0, nfields, 1))
    ds['storm_id'].attrs["child"] = "cell"
    ds['storm_id'].attrs["tree_id"] = ds.attrs["tree_id"]

    return ds


def load_wrf(file_list):
    ds = xr.open_mfdataset(file_list, concat_dim="Time")
    ds.attrs["cf_tree_order"] = "storm_id cell_id"
    ds.attrs["tree_id"] = "%d" % random.randint(a=0, b=65535)
    dt_array = []
    
    funcs = lambda x: datetime.strptime(x, str(b'%Y-%m-%d_%H:%M:%S'))
    t = ds["Times"]
    times = np.array([funcs(str(x)) for x in t.values]).astype('datetime64[s]')
    ds["time"] = xr.DataArray(times, dims="Time")
    # Try to detect number of fields (4D arrays)
    nfields = 0
    for keys in ds.variables.keys():
        if(len(ds[keys].dims) == 4):
            nfields += 1

    ds['storm_id'] = ('storm', np.arange(0, nfields, 1))
    ds['storm_id'].attrs["child"] = "cell"
    ds['storm_id'].attrs["tree_id"] = ds.attrs["tree_id"]
    ds['z'] = xr.DataArray([0, 1], dims='z')
    wrf_proj = pyproj.Proj(proj='lcc', # projection type: Lambert Conformal Conic
                       lat_1=ds.TRUELAT1, lat_2=ds.TRUELAT2, # Cone intersects with the sphere
                       lat_0=ds.MOAD_CEN_LAT, lon_0=ds.STAND_LON, # Center point
                       a=6370000, b=6370000) # This is it! The Earth is a perfect sphere

    # Easting and Northings of the domains center point
    wgs_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    e, n = pyproj.transform(wgs_proj, wrf_proj, ds.CEN_LON, ds.CEN_LAT)
    # Grid parameters
    dx, dy = ds.DX, ds.DY
    nx, ny = ds.dims['west_east'], ds.dims['south_north']
    # Down left corner of the domain
    x0 = -(nx-1) / 2. * dx + e
    y0 = -(ny-1) / 2. * dy + n
    # 2d grid
    xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)
    ds['x'] = xr.DataArray(xx, dims=('south_north', 'west_east'))
    ds['y'] = xr.DataArray(yy, dims=('south_north', 'west_east'))
    our_lons, our_lats = pyproj.transform(wrf_proj, wgs_proj, xx, yy)
    
    ds['point_longitude'] = xr.DataArray(our_lons, dims=('south_north', 'west_east'))
    ds['point_longitude'].attrs["units"] = 'degrees'
    ds['point_latitude'] = xr.DataArray(our_lats, dims=('south_north', 'west_east'))
    ds['point_latitude'].attrs["units"] = 'degrees'

    return ds
