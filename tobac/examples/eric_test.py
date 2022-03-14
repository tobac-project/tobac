from tobac.themes import tint
from copy import deepcopy

track_params = deepcopy(tint.objects.default_params)

nc_file_path = "/home/rjackson/tracer-jcss/Tgrid_*.nc"
nc_grid = tint.io.load_cfradial_grids(nc_file_path)
# print(nc_grid)
tracks = tint.make_tracks(nc_grid, "reflectivity", params=track_params)
print(tracks)
