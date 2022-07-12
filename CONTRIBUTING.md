# Contributing to tobac

__Welcome! We are very happy that you are interested in our project and thanks for taking time to contribute! :)__

We are currently reorganizing our project. So, please check our modifications later.


## Getting Started
### Installation & Environment details
You will find them in the [README.md](https://github.com/climate-processes/tobac/blob/master/README.md).

### Tutorials
Tutorials have been prepared to provide you further inside to `tobac`s functionality. Please visit the separate [tobac-tutorials](https://github.com/climate-processes/tobac-tutorials) repository here on Github.


### Documentation
You will find our documentation at [https://tobac.readthedocs.io](https://tobac.readthedocs.io).

### Testing
The tests are located in the [tests folder](https://github.com/climate-processes/tobac/tree/master/tobac/tests).


## Reporting Bugs
Please create a new issue on [GitHub](https://github.com/climate-processes/tobac/issues) if it is not listed there, yet.

### How to write a good Bug Report?
* Give it a clear descriptive title.
* Copy and paste the error message.
* Describe the steps for reproducing the problem and give an specific example.  
* Optional: Make a suggestion to fix it.


## How to Submit Changes
* Please read the [README.md](https://github.com/climate-processes/tobac/blob/master/README.md) first, to learn about our project goals and check the [changelog.md]().
* Before you start a pull request, please make sure that you added [numpydoc docstrings](#docstringExample) to your functions. This way the api documentation will be parsed properly.
* If it is a larger change or an newly added feature or workflow, please place an example of use in the [tobac-tutorials](https://github.com/climate-processes/tobac-tutorials) repository or adapt the existing examples there.
* If necessary add a folder or modify a file.
* The code should be PEP 8 compliant, as this facilitates our collaboration. Please use the latest version (22.6.0) of [black](https://black.readthedocs.io/en/stable/) to format your code. When you submit a pull request, all files are checked for formatting.
* The tobac repository is set up with pre-commit hooks to automatically format your code when commiting changes. Please run the command "pre-commit install" in the root directory of tobac to set up pre-commit formatting.

We aim to respond to all new issues/pull requests as soon as possible, however at times this is not possible due to work commitments.

### Numpydoc Example <a name='docstringExample'>
```python
  
   '''
   calculate centre of gravity and mass forech individual tracked cell in the simulation


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

## Slack
In addition to the workflow here on Github, there's a tobac workspace on Slack [tobac-dev.slack.com](tobac-dev.slack.com) that we use for some additional communication around the project. Please join us there to stay updated about all things tobac that go beyond the detailed work on the code.

