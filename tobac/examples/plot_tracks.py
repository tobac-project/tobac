import xarray as xr
import cartopy.crs as ccrs
from tobac.themes import tint

import matplotlib.pyplot as plt

from tobac.themes import tint
from matplotlib import use
nc_file_path = '/lcrc/group/earthscience/radar/houston/data/20160629/KHGX_grid_20160629.17*.nc'

nc_grid = tint.io.load_cfradial_grids(nc_file_path)
tracks = xr.open_dataset('Houston_tracks.nc')

fig, ax = plt.subplots(1, 1, projection=ccrs.PlateCarree())

for i in range(len(nc_grid.time)):
    nc_grid.isel(time=i).plot.pcolormesh(ax=ax)
    cells_in_time = tracks.where(time=nc_grid.time[i])
    for uid in cells_in_time:
         
    fig.savefig('%d.png' % i)
    plt.clf(fig)
    

