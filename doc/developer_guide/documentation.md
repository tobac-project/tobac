# Writing proper documentation


## Docstrings
Please provide **Numpy Docstrings** for all new functions.

**Example**:

```
'''
Calculates centre of gravity and mass for each individually tracked cell in the simulation.


 Parameters
 ----------
 tracks : pandas.DataFram
     DataFrame containing trajectories of cell centres

 param mass : iris.cube.Cube
     cube of quantity (need coordinates 'time', 'geopotential_height','projection_x_coordinate' and
     'projection_y_coordinate')

 param mask : iris.cube.Cube
     cube containing mask (int > where belonging to cloud volume, 0 everywhere else )


 Returns
 -------
 track_out : pandas.DataFrame
     Dataframe containing t,x,y,z positions of centre of gravity and total cloud mass each tracked cells
     at each timestep

 '''
```

## Documentation Pages

For any new features, especially if they are major, please include a documentation page and example using the feature. While our current documentation pages are a mixture between Markdown and ReStructuredText, we are in the process of migrating all docs pages to Markdown, so please write any new documentation page in Markdown, which is the same style that GitHub uses. 

## Tips on working on documentation for your local code

- Install tobac package with {code}`pip install -e`
  : - This allows you to directly test your local code changes as you run tobac. Instead of using the **tobac** version of the latest release, your local version of tobac will be used when you import **tobac** in a python script or IDE.
    - *Note that* this way of installing a local package will use the code of the checked in branch, so this allows you also to test code while switching between branches.
- You can locally **build the documentation page**:
  : - see {doc}`testing_sphinx-based_rendering`
- Writing [meaningful commit messages](https://www.conventionalcommits.org/en/v1.0.0/) can be very helpful for you and people who review your code to better understand the code changes.

## Add examples

In addition to the unit tests, we aim to provide examples on how to use all functionalities and how to choose different
tracking parameters. These [examples](https://github.com/tobac-project/tobac/tree/main/examples) are in form of jupyter
notebooks and can be based on simple, idealized test cases or real data. We strongly encourage the use of real data that
is publicly accessible, but another option for new examples with real data is to either upload the data to our [zenodo
repository](https://zenodo.org/records/10863405) or create your own data upload on zenodo. Please include the name "tobac" in the data title for the latter.
