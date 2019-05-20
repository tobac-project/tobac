Data input and output
======================
The output of the different analysis steps in tobac are output as either pandas DataFrames in the case of one-dimensional data, such a lists of identified features or cloud trajectoies or as Iris cubes in the case of 2D/3D/4D fields such as cloud masks.

Interoperability with xarray is provided by the convenient functions allowing for a transformation between the two data types.
xarray DataArays can be easily converted into iris cubes using the .to_cube method, while Iris cubes can be turned into 

For the future we are envisaging moving the basic data structures from Iris cubes to xarray DataArrays for improved computing performance.
