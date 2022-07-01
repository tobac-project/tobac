import logging


def calculate_cog(tracks, mass, mask):
    """caluclate centre of gravity and mass forech individual tracked cell in the simulation
    Input:
    tracks:     pandas.DataFrame
                DataFrame containing trajectories of cell centres
    mass:       iris.cube.Cube
                cube of quantity (need coordinates 'time', 'geopotential_height','projection_x_coordinate' and 'projection_y_coordinate')
    mask:       iris.cube.Cube
                cube containing mask (int > where belonging to cloud volume, 0 everywhere else )
    Output:
    tracks_out  pandas.DataFrame
                Dataframe containing t,x,y,z positions of centre of gravity and total cloud mass each tracked cells at each timestep
    """
    from .utils import mask_cube_cell
    from iris import Constraint

    logging.info("start calculating centre of gravity for tracked cells")

    tracks_out = tracks[["time", "frame", "cell", "time_cell"]]

    for i_row, row in tracks_out.iterrows():
        cell = row["cell"]
        constraint_time = Constraint(time=row["time"])
        mass_i = mass.extract(constraint_time)
        mask_i = mask.extract(constraint_time)
        mass_masked_i = mask_cube_cell(mass_i, mask_i, cell)
        x_M, y_M, z_M, mass_M = center_of_gravity(mass_masked_i)
        tracks_out.loc[i_row, "x_M"] = float(x_M)
        tracks_out.loc[i_row, "y_M"] = float(y_M)
        tracks_out.loc[i_row, "z_M"] = float(z_M)
        tracks_out.loc[i_row, "mass"] = float(mass_M)

    logging.info("Finished calculating centre of gravity for tracked cells")

    return tracks_out


def calculate_cog_untracked(mass, mask):
    """caluclate centre of gravity and mass for untracked parts of domain
    Input:
    mass:       iris.cube.Cube
                cube of quantity (need coordinates 'time', 'geopotential_height','projection_x_coordinate' and 'projection_y_coordinate')

    mask:       iris.cube.Cube
                cube containing mask (int > where belonging to cloud volume, 0 everywhere else )
    Output:
    tracks_out  pandas.DataFrame
                Dataframe containing t,x,y,z positions of centre of gravity and total cloud mass for untracked part of dimain
    """
    from pandas import DataFrame
    from .utils import mask_cube_untracked
    from iris import Constraint

    logging.info(
        "start calculating centre of gravity for untracked parts of the domain"
    )
    tracks_out = DataFrame()
    time_coord = mass.coord("time")
    tracks_out["frame"] = range(len(time_coord.points))
    for i_row, row in tracks_out.iterrows():
        time_i = time_coord.units.num2date(time_coord[int(row["frame"])].points[0])
        constraint_time = Constraint(time=time_i)
        mass_i = mass.extract(constraint_time)
        mask_i = mask.extract(constraint_time)
        mass_untracked_i = mask_cube_untracked(mass_i, mask_i)
        x_M, y_M, z_M, mass_M = center_of_gravity(mass_untracked_i)
        tracks_out.loc[i_row, "time"] = time_i
        tracks_out.loc[i_row, "x_M"] = float(x_M)
        tracks_out.loc[i_row, "y_M"] = float(y_M)
        tracks_out.loc[i_row, "z_M"] = float(z_M)
        tracks_out.loc[i_row, "mass"] = float(mass_M)

    logging.info(
        "Finished calculating centre of gravity for untracked parts of the domain"
    )

    return tracks_out


def calculate_cog_domain(mass):
    """caluclate centre of gravity and mass for entire domain
    Input:
    mass:       iris.cube.Cube
                cube of quantity (need coordinates 'time', 'geopotential_height','projection_x_coordinate' and 'projection_y_coordinate')
    Output:
    tracks_out  pandas.DataFrame
                Dataframe containing t,x,y,z positions of centre of gravity and total cloud mass
    """
    from pandas import DataFrame
    from iris import Constraint

    logging.info("start calculating centre of gravity for entire domain")

    time_coord = mass.coord("time")

    tracks_out = DataFrame()
    tracks_out["frame"] = range(len(time_coord.points))
    for i_row, row in tracks_out.iterrows():
        time_i = time_coord.units.num2date(time_coord[int(row["frame"])].points[0])
        constraint_time = Constraint(time=time_i)
        mass_i = mass.extract(constraint_time)
        x_M, y_M, z_M, mass_M = center_of_gravity(mass_i)
        tracks_out.loc[i_row, "time"] = time_i
        tracks_out.loc[i_row, "x_M"] = float(x_M)
        tracks_out.loc[i_row, "y_M"] = float(y_M)
        tracks_out.loc[i_row, "z_M"] = float(z_M)
        tracks_out.loc[i_row, "mass"] = float(mass_M)

    logging.info("Finished calculating centre of gravity for entire domain")

    return tracks_out


def center_of_gravity(cube_in):
    """caluclate centre of gravity and sum of quantity
    Input:
    cube_in:       iris.cube.Cube
                   cube (potentially masked) of quantity (need coordinates 'geopotential_height','projection_x_coordinate' and 'projection_y_coordinate')
    Output:
    x:             float
                   x position of centre of gravity
    y:             float
                   y position of centre of gravity
    z:             float
                   z position of centre of gravity
    variable_sum:  float
                   sum of quantity of over unmasked part of the cube

    """
    from iris.analysis import SUM
    import numpy as np

    cube_sum = cube_in.collapsed(["bottom_top", "south_north", "west_east"], SUM)
    z = cube_in.coord("geopotential_height")
    x = cube_in.coord("projection_x_coordinate")
    y = cube_in.coord("projection_y_coordinate")
    dimensions_collapse = ["model_level_number", "x", "y"]
    for coord in cube_in.coords():
        if coord.ndim > 1 and (
            cube_in.coord_dims(dimensions_collapse[0])[0] in cube_in.coord_dims(coord)
            or cube_in.coord_dims(dimensions_collapse[1])[0]
            in cube_in.coord_dims(coord)
            or cube_in.coord_dims(dimensions_collapse[2])[0]
            in cube_in.coord_dims(coord)
        ):
            cube_in.remove_coord(coord.name())
    if cube_sum.data > 0:
        x = (
            (cube_in * x).collapsed(["model_level_number", "x", "y"], SUM) / cube_sum
        ).data
        y = (
            (cube_in * y).collapsed(["model_level_number", "x", "y"], SUM) / cube_sum
        ).data
        z = (
            (cube_in * z.points).collapsed(["model_level_number", "x", "y"], SUM)
            / cube_sum
        ).data
    else:
        x = np.nan
        y = np.nan
        z = np.nan
    variable_sum = cube_sum.data
    return (x, y, z, variable_sum)
