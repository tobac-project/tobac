import cartopy.crs as ccrs
import tobac
import matplotlib.pyplot as plt

from tobac.themes import tint
from matplotlib import use

nc_file_path = (
    "/lcrc/group/earthscience/radar/houston/data/20160629/KHGX_grid_20160629.17*.nc"
)

nc_grid = tint.io.load_cfradial_grids(nc_file_path)
print(nc_grid)
tracks = tint.make_tracks(nc_grid, "reflectivity")
print(tracks)
tracks.to_netcdf("Houston_tracks.nc")
