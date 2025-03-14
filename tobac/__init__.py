# from .tracking import maketrack
import sys

if sys.version_info < (3, 7):
    warning = """ \n\n 
    Support for Python versions less than 3.7 is deprecated. 
    Version 1.5 of tobac will require Python 3.7 or later.
   Python {py} detected. \n\n
    """.format(
        py=".".join(str(v) for v in sys.version_info[:3])
    )

    print(warning)

from .segmentation import (
    segmentation_3D,
    segmentation_2D,
    watershedding_3D,
    watershedding_2D,
)
from .centerofgravity import (
    calculate_cog,
    calculate_cog_untracked,
    calculate_cog_domain,
)
from .plotting import (
    plot_tracks_mask_field,
    plot_tracks_mask_field_loop,
    plot_mask_cell_track_follow,
    plot_mask_cell_track_static,
    plot_mask_cell_track_static_timeseries,
    plot_lifetime_histogram,
    plot_lifetime_histogram_bar,
    plot_histogram_cellwise,
    plot_histogram_featurewise,
    plot_mask_cell_track_3Dstatic,
    plot_mask_cell_track_2D3Dstatic,
    plot_mask_cell_individual_static,
    plot_mask_cell_individual_3Dstatic,
    animation_mask_field,
    make_map,
    map_tracks,
)
from tobac.analysis.cell_analysis import (
    cell_statistics,
    cog_cell,
    lifetime_histogram,
    histogram_cellwise,
    velocity_histogram,
    calculate_overlap,
)
from tobac.analysis.feature_analysis import (
    histogram_featurewise,
    calculate_nearestneighbordistance,
    nearestneighbordistance_histogram,
    area_histogram,
)
from tobac.analysis.spatial import (
    calculate_velocity,
    calculate_distance,
    calculate_area,
)
from .utils.mask import (
    mask_cell,
    mask_cell_surface,
    mask_cube_cell,
    mask_cube_untracked,
    mask_cube,
    column_mask_from2D,
    mask_features,
    mask_features_surface,
    mask_cube_features,
)
from .utils.general import (
    get_bounding_box,
    add_coordinates,
    get_spacings,
)
from .feature_detection import feature_detection_multithreshold
from .tracking import linking_trackpy
from .wrapper import maketrack
from .wrapper import tracking_wrapper
from . import merge_split

# Set version number
__version__ = "1.5.5"
