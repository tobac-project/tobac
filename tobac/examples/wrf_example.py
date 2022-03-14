import cartopy.crs as ccrs
import tobac
import matplotlib.pyplot as plt

from tobac.themes import tint
from matplotlib import use

nc_file_path = "/lcrc/project/ACPC/WRF/runs/20130619/wrfout/wrfout_d03*"

nc_grid = tint.io.load_wrf(nc_file_path)
print(nc_grid)
tracks = tint.make_tracks_2d_field(nc_grid, "REFD_COM")
print(tracks)
tracks.to_netcdf("wrf_tracks.nc")
