""" X-Array based TINT I/O module. """

import xarray as xr
import random
import numpy as np

from .grid_utils import add_lat_lon_grid

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
        if(len(ds[keys].dims) == 4):
            nfields += 1

    ds['storm_id'] = ('storm', np.arange(0, nfields, 1))
    ds['storm_id'].attrs["child"] = "cell"
    ds['storm_id'].attrs["tree_id"] = ds.attrs["tree_id"]

    return ds



