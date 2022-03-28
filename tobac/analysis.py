import pandas as pd
import numpy as np
import logging
import os

from .utils import mask_cell, mask_cell_surface, mask_cube_cell, get_bounding_box


def cell_statistics_all(
    input_cubes,
    track,
    mask,
    aggregators,
    output_path="./",
    cell_selection=None,
    output_name="Profiles",
    width=10000,
    z_coord="model_level_number",
    dimensions=["x", "y"],
    **kwargs
):
    if cell_selection is None:
        cell_selection = np.unique(track["cell"])
    for cell in cell_selection:
        cell_statistics(
            input_cubes=input_cubes,
            track=track,
            mask=mask,
            dimensions=dimensions,
            aggregators=aggregators,
            cell=cell,
            output_path=output_path,
            output_name=output_name,
            width=width,
            z_coord=z_coord,
            **kwargs
        )


def cell_statistics(
    input_cubes,
    track,
    mask,
    aggregators,
    cell,
    output_path="./",
    output_name="Profiles",
    width=10000,
    z_coord="model_level_number",
    dimensions=["x", "y"],
    **kwargs
):
    from iris.cube import Cube, CubeList
    from iris.coords import AuxCoord
    from iris import Constraint, save

    # If input is single cube, turn into cubelist
    if type(input_cubes) is Cube:
        input_cubes = CubeList([input_cubes])

    logging.debug("Start calculating profiles for cell " + str(cell))
    track_i = track[track["cell"] == cell]

    cubes_profile = {}
    for aggregator in aggregators:
        cubes_profile[aggregator.name()] = CubeList()

    for time_i in track_i["time"].values:
        constraint_time = Constraint(time=time_i)

        mask_i = mask.extract(constraint_time)
        mask_cell_i = mask_cell(mask_i, cell, track_i, masked=False)
        mask_cell_surface_i = mask_cell_surface(
            mask_i, cell, track_i, masked=False, z_coord=z_coord
        )

        x_dim = mask_cell_surface_i.coord_dims("projection_x_coordinate")[0]
        y_dim = mask_cell_surface_i.coord_dims("projection_y_coordinate")[0]
        x_coord = mask_cell_surface_i.coord("projection_x_coordinate")
        y_coord = mask_cell_surface_i.coord("projection_y_coordinate")

        if (mask_cell_surface_i.core_data() > 0).any():
            box_mask_i = get_bounding_box(mask_cell_surface_i.core_data(), buffer=1)

            box_mask = [
                [
                    x_coord.points[box_mask_i[x_dim][0]],
                    x_coord.points[box_mask_i[x_dim][1]],
                ],
                [
                    y_coord.points[box_mask_i[y_dim][0]],
                    y_coord.points[box_mask_i[y_dim][1]],
                ],
            ]
        else:
            box_mask = [[np.nan, np.nan], [np.nan, np.nan]]

        x = track_i[track_i["time"].values == time_i]["projection_x_coordinate"].values[
            0
        ]
        y = track_i[track_i["time"].values == time_i]["projection_y_coordinate"].values[
            0
        ]

        box_slice = [[x - width, x + width], [y - width, y + width]]

        x_min = np.nanmin([box_mask[0][0], box_slice[0][0]])
        x_max = np.nanmax([box_mask[0][1], box_slice[0][1]])
        y_min = np.nanmin([box_mask[1][0], box_slice[1][0]])
        y_max = np.nanmax([box_mask[1][1], box_slice[1][1]])

        constraint_x = Constraint(
            projection_x_coordinate=lambda cell: int(x_min) < cell < int(x_max)
        )
        constraint_y = Constraint(
            projection_y_coordinate=lambda cell: int(y_min) < cell < int(y_max)
        )

        constraint = constraint_time & constraint_x & constraint_y
        #       Mask_cell_surface_i=mask_cell_surface(Mask_w_i,cell,masked=False,z_coord='model_level_number')
        mask_cell_i = mask_cell_i.extract(constraint)
        mask_cell_surface_i = mask_cell_surface_i.extract(constraint)

        input_cubes_i = input_cubes.extract(constraint)
        for cube in input_cubes_i:
            cube_masked = mask_cube_cell(cube, mask_cell_i, cell, track_i)
            coords_remove = []
            for coordinate in cube_masked.coords(dim_coords=False):

                if coordinate.name() not in dimensions:
                    for dim in dimensions:
                        if set(cube_masked.coord_dims(coordinate)).intersection(
                            set(cube_masked.coord_dims(dim))
                        ):
                            coords_remove.append(coordinate.name())
            for coordinate in set(coords_remove):
                cube_masked.remove_coord(coordinate)

            for aggregator in aggregators:
                cube_collapsed = cube_masked.collapsed(dimensions, aggregator, **kwargs)
                # remove all collapsed coordinates (x and y dim, scalar now) and keep only time as all these coordinates are useless
                for coordinate in cube_collapsed.coords():
                    if not cube_collapsed.coord_dims(coordinate):
                        if coordinate.name() != "time":
                            cube_collapsed.remove_coord(coordinate)
                logging.debug(str(cube_collapsed))
                cubes_profile[aggregator.name()].append(cube_collapsed)

    minutes = (track_i["time_cell"] / pd.Timedelta(minutes=1)).values
    latitude = track_i["latitude"].values
    longitude = track_i["longitude"].values
    minutes_coord = AuxCoord(minutes, long_name="cell_time", units="min")
    latitude_coord = AuxCoord(latitude, long_name="latitude", units="degrees")
    longitude_coord = AuxCoord(longitude, long_name="longitude", units="degrees")

    for aggregator in aggregators:
        cubes_profile[aggregator.name()] = cubes_profile[aggregator.name()].merge()
        for cube in cubes_profile[aggregator.name()]:
            cube.add_aux_coord(minutes_coord, data_dims=cube.coord_dims("time"))
            cube.add_aux_coord(latitude_coord, data_dims=cube.coord_dims("time"))
            cube.add_aux_coord(longitude_coord, data_dims=cube.coord_dims("time"))
        os.makedirs(
            os.path.join(output_path, output_name, aggregator.name()), exist_ok=True
        )
        savefile = os.path.join(
            output_path,
            output_name,
            aggregator.name(),
            output_name + "_" + aggregator.name() + "_" + str(int(cell)) + ".nc",
        )
        save(cubes_profile[aggregator.name()], savefile)


def cog_cell(
    cell,
    Tracks=None,
    M_total=None,
    M_liquid=None,
    M_frozen=None,
    Mask=None,
    savedir=None,
):

    from iris import Constraint

    logging.debug("Start calculating COG for " + str(cell))
    Track = Tracks[Tracks["cell"] == cell]
    constraint_time = Constraint(
        time=lambda cell: Track.head(1)["time"].values[0]
        <= cell
        <= Track.tail(1)["time"].values[0]
    )
    M_total_i = M_total.extract(constraint_time)
    M_liquid_i = M_liquid.extract(constraint_time)
    M_frozen_i = M_frozen.extract(constraint_time)
    Mask_i = Mask.extract(constraint_time)

    savedir_cell = os.path.join(savedir, "cells", str(int(cell)))
    os.makedirs(savedir_cell, exist_ok=True)
    savefile_COG_total_i = os.path.join(
        savedir_cell, "COG_total" + "_" + str(int(cell)) + ".h5"
    )
    savefile_COG_liquid_i = os.path.join(
        savedir_cell, "COG_liquid" + "_" + str(int(cell)) + ".h5"
    )
    savefile_COG_frozen_i = os.path.join(
        savedir_cell, "COG_frozen" + "_" + str(int(cell)) + ".h5"
    )

    Tracks_COG_total_i = calculate_cog(Track, M_total_i, Mask_i)
    #   Tracks_COG_total_list.append(Tracks_COG_total_i)
    logging.debug("COG total loaded for " + str(cell))

    Tracks_COG_liquid_i = calculate_cog(Track, M_liquid_i, Mask_i)
    #   Tracks_COG_liquid_list.append(Tracks_COG_liquid_i)
    logging.debug("COG liquid loaded for " + str(cell))
    Tracks_COG_frozen_i = calculate_cog(Track, M_frozen_i, Mask_i)
    #   Tracks_COG_frozen_list.append(Tracks_COG_frozen_i)
    logging.debug("COG frozen loaded for " + str(cell))

    Tracks_COG_total_i.to_hdf(savefile_COG_total_i, "table")
    Tracks_COG_liquid_i.to_hdf(savefile_COG_liquid_i, "table")
    Tracks_COG_frozen_i.to_hdf(savefile_COG_frozen_i, "table")
    logging.debug("individual COG calculated and saved to " + savedir_cell)


def lifetime_histogram(
    Track, bin_edges=np.arange(0, 200, 20), density=False, return_values=False
):
    Track_cell = Track.groupby("cell")
    minutes = (Track_cell["time_cell"].max() / pd.Timedelta(minutes=1)).values
    hist, bin_edges = np.histogram(minutes, bin_edges, density=density)
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    if return_values:
        return hist, bin_edges, bin_centers, minutes
    else:
        return hist, bin_edges, bin_centers


def haversine(lat1, lon1, lat2, lon2):
    """Computes the Haversine distance in kilometres between two points (based on implementation CIS https://github.com/cedadev/cis)
    :param lat1: first point or points as array, each as array of latitude in degrees
    :param lon1: first point or points as array, each as array of longitude in degrees
    :param lat2: second point or points as array, each as array of latitude in degrees
    :param lon2: second point or points as array, each as array of longitude in degrees
    :return: distance between the two points in kilometres
    """
    RADIUS_EARTH = 6378.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    # print(lat1,lat2,lon1,lon2)
    arclen = 2 * np.arcsin(
        np.sqrt(
            (np.sin((lat2 - lat1) / 2)) ** 2
            + np.cos(lat1) * np.cos(lat2) * (np.sin((lon2 - lon1) / 2)) ** 2
        )
    )
    return arclen * RADIUS_EARTH


def calculate_distance(feature_1, feature_2, method_distance=None):
    """Computes distance between two features based on either lat/lon coordinates or x/y coordinates
    :param feature_1: first feature or points as array, each as array of latitude, longitude in degrees
    :param feature_2: second feature or points as array, each as array of latitude, longitude in degrees
    :return: distance between the two features in metres
    """
    if method_distance is None:
        if (
            ("projection_x_coordinate" in feature_1)
            and ("projection_y_coordinate" in feature_1)
            and ("projection_x_coordinate" in feature_2)
            and ("projection_y_coordinate" in feature_2)
        ):
            method_distance = "xy"
        elif (
            ("latitude" in feature_1)
            and ("longitude" in feature_1)
            and ("latitude" in feature_2)
            and ("longitude" in feature_2)
        ):
            method_distance = "latlon"
        else:
            raise ValueError(
                "either latitude/longitude or projection_x_coordinate/projection_y_coordinate have to be present to calculate distances"
            )

    if method_distance == "xy":
        distance = np.sqrt(
            (
                feature_1["projection_x_coordinate"]
                - feature_2["projection_x_coordinate"]
            )
            ** 2
            + (
                feature_1["projection_y_coordinate"]
                - feature_2["projection_y_coordinate"]
            )
            ** 2
        )
    elif method_distance == "latlon":
        distance = 1000 * haversine(
            feature_1["latitude"],
            feature_1["longitude"],
            feature_2["latitude"],
            feature_2["longitude"],
        )
    else:
        raise ValueError("method undefined")
    return distance


def calculate_velocity_individual(feature_old, feature_new, method_distance=None):
    distance = calculate_distance(
        feature_old, feature_new, method_distance=method_distance
    )
    diff_time = (feature_new["time"] - feature_old["time"]).total_seconds()
    velocity = distance / diff_time
    return velocity


def calculate_velocity(track, method_distance=None):
    for cell_i, track_i in track.groupby("cell"):
        index = track_i.index.values
        for i, index_i in enumerate(index[:-1]):
            velocity = calculate_velocity_individual(
                track_i.loc[index[i]],
                track_i.loc[index[i + 1]],
                method_distance=method_distance,
            )
            track.at[index_i, "v"] = velocity
    return track


def velocity_histogram(
    track,
    bin_edges=np.arange(0, 30, 1),
    density=False,
    method_distance=None,
    return_values=False,
):
    if "v" not in track.columns:
        logging.info("calculate velocities")
        track = calculate_velocity(track)
    velocities = track["v"].values
    hist, bin_edges = np.histogram(
        velocities[~np.isnan(velocities)], bin_edges, density=density
    )
    if return_values:
        return hist, bin_edges, velocities
    else:
        return hist, bin_edges


def calculate_nearestneighbordistance(features, method_distance=None):
    from itertools import combinations

    features["min_distance"] = np.nan
    for time_i, features_i in features.groupby("time"):
        logging.debug(str(time_i))
        indeces = combinations(features_i.index.values, 2)
        # Loop over combinations to remove features that are closer together than min_distance and keep larger one (either higher threshold or larger area)
        distances = []
        for index_1, index_2 in indeces:
            if index_1 is not index_2:
                distance = calculate_distance(
                    features_i.loc[index_1],
                    features_i.loc[index_2],
                    method_distance=method_distance,
                )
                distances.append(
                    pd.DataFrame(
                        {"index_1": index_1, "index_2": index_2, "distance": distance},
                        index=[0],
                    )
                )
        if any([x is not None for x in distances]):
            distances = pd.concat(distances, ignore_index=True)
            for i in features_i.index:
                min_distance = distances.loc[
                    (distances["index_1"] == i) | (distances["index_2"] == i),
                    "distance",
                ].min()
                features.at[i, "min_distance"] = min_distance
    return features


def nearestneighbordistance_histogram(
    features,
    bin_edges=np.arange(0, 30000, 500),
    density=False,
    method_distance=None,
    return_values=False,
):
    if "min_distance" not in features.columns:
        logging.debug("calculate nearest neighbor distances")
        features = calculate_nearestneighbordistance(
            features, method_distance=method_distance
        )
    distances = features["min_distance"].values
    hist, bin_edges = np.histogram(
        distances[~np.isnan(distances)], bin_edges, density=density
    )
    if return_values:
        return hist, bin_edges, distances
    else:
        return hist, bin_edges


# Treatment of 2D lat/lon coordinates to be added:
# def calculate_areas_2Dlatlon(latitude_coord,longitude_coord):
#     lat=latitude_coord.core_data()
#     lon=longitude_coord.core_data()
#     area=np.zeros(lat.shape)
#     dx=np.zeros(lat.shape)
#     dy=np.zeros(lat.shape)

#     return area


def calculate_area(features, mask, method_area=None):
    from tobac.utils import mask_features_surface, mask_features
    from iris import Constraint
    from iris.analysis.cartography import area_weights
    from scipy.ndimage import labeled_comprehension

    features["area"] = np.nan

    mask_coords = [coord.name() for coord in mask.coords()]
    if method_area is None:
        if ("projection_x_coordinate" in mask_coords) and (
            "projection_y_coordinate" in mask_coords
        ):
            method_area = "xy"
        elif ("latitude" in mask_coords) and ("longitude" in mask_coords):
            method_area = "latlon"
        else:
            raise ValueError(
                "either latitude/longitude or projection_x_coordinate/projection_y_coordinate have to be present to calculate distances"
            )
    # logging.debug("calculating area using method " + method_area)
    if method_area == "xy":
        if not (
            mask.coord("projection_x_coordinate").has_bounds()
            and mask.coord("projection_y_coordinate").has_bounds()
        ):
            mask.coord("projection_x_coordinate").guess_bounds()
            mask.coord("projection_y_coordinate").guess_bounds()
        area = np.outer(
            np.diff(mask.coord("projection_x_coordinate").bounds, axis=1),
            np.diff(mask.coord("projection_y_coordinate").bounds, axis=1),
        )
    elif method_area == "latlon":
        if (mask.coord("latitude").ndim == 1) and (mask.coord("latitude").ndim == 1):
            if not (
                mask.coord("latitude").has_bounds()
                and mask.coord("longitude").has_bounds()
            ):
                mask.coord("latitude").guess_bounds()
                mask.coord("longitude").guess_bounds()
            area = area_weights(mask, normalize=False)
        elif mask.coord("latitude").ndim == 2 and mask.coord("longitude").ndim == 2:
            raise ValueError("2D latitude/longitude coordinates not supported yet")
            # area=calculate_areas_2Dlatlon(mask.coord('latitude'),mask.coord('longitude'))
        else:
            raise ValueError("latitude/longitude coordinate shape not supported")
    else:
        raise ValueError("method undefined")

    feature_areas = labeled_comprehension(area, mask.data, features["feature"], np.sum, area.dtype, np.nan)

    features["area"] = feature_areas

    return features


def area_histogram(
    features,
    mask,
    bin_edges=np.arange(0, 30000, 500),
    density=False,
    method_area=None,
    return_values=False,
    representative_area=False,
):
    if "area" not in features.columns:
        logging.info("calculate area")
        features = calculate_area(features, mask, method_area)
    areas = features["area"].values
    # restrict to non NaN values:
    areas = areas[~np.isnan(areas)]
    if representative_area:
        weights = areas
    else:
        weights = None
    hist, bin_edges = np.histogram(areas, bin_edges, density=density, weights=weights)
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)

    if return_values:
        return hist, bin_edges, bin_centers, areas
    else:
        return hist, bin_edges, bin_centers


def histogram_cellwise(
    Track, variable=None, bin_edges=None, quantity="max", density=False
):
    Track_cell = Track.groupby("cell")
    if quantity == "max":
        variable_cell = Track_cell[variable].max().values
    elif quantity == "min":
        variable_cell = Track_cell[variable].min().values
    elif quantity == "mean":
        variable_cell = Track_cell[variable].mean().values
    else:
        raise ValueError("quantity unknown, must be max, min or mean")
    hist, bin_edges = np.histogram(variable_cell, bin_edges, density=density)
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)

    return hist, bin_edges, bin_centers


def histogram_featurewise(Track, variable=None, bin_edges=None, density=False):
    hist, bin_edges = np.histogram(Track[variable].values, bin_edges, density=density)
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)

    return hist, bin_edges, bin_centers


def calculate_overlap(
    track_1, track_2, min_sum_inv_distance=None, min_mean_inv_distance=None
):
    cells_1 = track_1["cell"].unique()
    #    n_cells_1_tot=len(cells_1)
    cells_2 = track_2["cell"].unique()
    overlap = pd.DataFrame()
    for i_cell_1, cell_1 in enumerate(cells_1):
        for cell_2 in cells_2:
            track_1_i = track_1[track_1["cell"] == cell_1]
            track_2_i = track_2[track_2["cell"] == cell_2]
            track_1_i = track_1_i[track_1_i["time"].isin(track_2_i["time"])]
            track_2_i = track_2_i[track_2_i["time"].isin(track_1_i["time"])]
            if not track_1_i.empty:
                n_overlap = len(track_1_i)
                distances = []
                for i in range(len(track_1_i)):
                    distance = calculate_distance(
                        track_1_i.iloc[[i]], track_2_i.iloc[[i]], method_distance="xy"
                    )
                    distances.append(distance)
                #                mean_distance=np.mean(distances)
                mean_inv_distance = np.mean(1 / (1 + np.array(distances) / 1000))
                #                mean_inv_squaredistance=np.mean(1/(1+(np.array(distances)/1000)**2))
                sum_inv_distance = np.sum(1 / (1 + np.array(distances) / 1000))
                #                sum_inv_squaredistance=np.sum(1/(1+(np.array(distances)/1000)**2))
                overlap = overlap.append(
                    {
                        "cell_1": cell_1,
                        "cell_2": cell_2,
                        "n_overlap": n_overlap,
                        #                                'mean_distance':mean_distance,
                        "mean_inv_distance": mean_inv_distance,
                        #                                'mean_inv_squaredistance':mean_inv_squaredistance,
                        "sum_inv_distance": sum_inv_distance,
                        #                                'sum_inv_squaredistance':sum_inv_squaredistance
                    },
                    ignore_index=True,
                )
    if min_sum_inv_distance:
        overlap = overlap[(overlap["sum_inv_distance"] >= min_sum_inv_distance)]
    if min_mean_inv_distance:
        overlap = overlap[(overlap["mean_inv_distance"] >= min_mean_inv_distance)]

    return overlap
