'''Provide essential methods to convert between function inputs in a convenient 
way. To be used for temporarily retaining code based on Iris cubes while moving 
the main framework to be based on xarray.

'''
import logging
import functools
#import inspect

def iris_to_xarray(func):
    '''Decorator that converts all input of a function that is in the form of Iris cubes into xarray DataArrays and converts all output in xarray DataArrays back into Iris cubes
    
    Parameters
    ----------
    func : function
        Function to be decorated

    Returns
    -------
    wrapper : function
        Function including decorator
     '''
    import iris
    import xarray
    def wrapper(*args, **kwargs):    
        #print(kwargs)
        if (any([type(arg)==iris.cube.Cube for arg in args]) or any([type(arg)==iris.cube.Cube for arg in kwargs])):        
            #print("converting iris to xarray and back")
            args = tuple([xarray.DataArray.from_iris(arg) if type(arg) == iris.cube.Cube else arg for arg in args])
            kwargs =kwargs.update(zip(kwargs.keys(), [xarray.DataArray.from_iris(arg) if type(arg) == iris.cube.Cube else arg for arg in kwargs.values()]))

            output=func(*args, **kwargs)
            if type(output)==tuple:
                output = tuple([xarray.DataArray.to_iris(output_item) if type(output_item) == xarray.DataArray else output_item for output_item in output])
            else:
                output = xarray.DataArray.to_iris(output)

        else:
            output=func(*args,**kwargs)
        return output
    return wrapper

def xarray_to_iris(func):
    '''Decorator that converts all input of a function that is in the form of xarray DataArrays Iris cubes into and converts all output in Iris cubes back into xarray DataArrays 
        
    Parameters
    ----------
    func : function
        Function to be decorated

    Returns
    -------
    wrapper : function
        Function including decorator
    '''

    import iris
    import xarray
    def wrapper(*args, **kwargs):
        #print(args)
        #print(kwargs)
        if (any([type(arg)==xarray.DataArray for arg in args]) or any([type(arg)==xarray.DataArray for arg in kwargs])):        
            #print("converting xarray to iris and back")
            args = tuple([xarray.DataArray.to_iris(arg) if type(arg) == xarray.DataArray else arg for arg in args])
            if kwargs:
                kwargs_new =dict(zip(kwargs.keys(),[xarray.DataArray.to_iris(arg) if type(arg) == xarray.DataArray else arg for arg in kwargs.values()]))                
            else:
                kwargs_new=kwargs
            #print(args)
            #print(kwargs)
            output=func(*args, **kwargs_new)
            if type(output)==tuple:
                output = tuple([xarray.DataArray.from_iris(output_item) if type(output_item) == iris.cube.Cube else output_item for output_item in output])
            else:
                if type(output)==iris.cube.Cube:
                    output = xarray.DataArray.from_iris(output)

        else:
            output=func(*args,**kwargs)
        #print(output)
        return output
    return wrapper


def irispandas_to_xarray(func):
    '''Decorator that converts all input of a function that is in the form of Iris cubes into xarray DataArrays and converts all output in xarray DataArrays back into Iris cubes
    
    Parameters
    ----------
    func : function
        Function to be decorated

    Returns
    -------
    wrapper : function
        Function including decorator
     '''
    import iris
    import xarray
    import pandas as pd
    def wrapper(*args, **kwargs):    
        #print(kwargs)
        if (any([(type(arg)==iris.cube.Cube or type(arg)==pd.DataFrame)  for arg in args]) or any([(type(arg)==iris.cube.Cube  or type(arg)==pd.DataFrame)  for arg in kwargs])):        
            #print("converting iris to xarray and back")
            args = tuple([xarray.DataArray.from_iris(arg) if type(arg) == iris.cube.Cube else arg.to_xarray if type(arg)==pd.DataFrame  else arg for arg in args])
            kwargs =kwargs.update(zip(kwargs.keys(), [xarray.DataArray.from_iris(arg) if type(arg) == iris.cube.Cube else arg.to_xarray if type(arg)==pd.DataFrame else arg for arg in kwargs.values()]))

            output=func(*args, **kwargs)
            if type(output)==tuple:
                output = tuple([xarray.DataArray.to_iris(output_item) if type(output_item) == xarray.DataArray else output_item.to_dataframe() if type(output_item) == xarray.Dataset else output_item for output_item in output])
            else:
                if type(output) == xarray.DataArray:
                    output = xarray.DataArray.to_iris(output)
                elif type(output) == xarray.Dataset:
                    output.to_dataframe()

        else:
            output=func(*args,**kwargs)
        return output
    return wrapper

def xarray_to_irispandas(func):
    '''Decorator that converts all input of a function that is in the form of xarray DataArrays Iris cubes into and converts all output in Iris cubes back into xarray DataArrays 
        
    Parameters
    ----------
    func : function
        Function to be decorated

    Returns
    -------
    wrapper : function
        Function including decorator
    '''

    import iris
    import xarray
    import pandas as pd
    def wrapper(*args, **kwargs):
        #print(args)
        #print(kwargs)
        if (any([(type(arg)==xarray.DataArray or type(arg)==xarray.Dataset) for arg in args]) or any([(type(arg)==xarray.DataArray or type(arg)==xarray.Dataset) for arg in kwargs])):        
            #print("converting xarray to iris and back")
            args = tuple([xarray.DataArray.to_iris(arg) if type(arg) == xarray.DataArray else arg.to_dataframe() if type(arg) == xarray.Dataset else arg for arg in args])
            if kwargs:
                kwargs_new =dict(zip(kwargs.keys(),[xarray.DataArray.to_iris(arg) if type(arg) == xarray.DataArray  else arg.to_dataframe() if type(arg) == xarray.Dataset else arg for arg in kwargs.values()]))                
            else:
                kwargs_new=kwargs
            #print(args)
            #print(kwargs)
            output=func(*args, **kwargs_new)
            if type(output)==tuple:
                output = tuple([xarray.DataArray.from_iris(output_item) if type(output_item) == iris.cube.Cube else output_item.to_xarray() if type(output_item) == pd.DataFrame else output_item for output_item in output])
            else:
                if type(output)==iris.cube.Cube:
                    output = xarray.DataArray.from_iris(output)
                elif type(output) == pd.DataFrame:
                    output.to_xarray()


        else:
            output=func(*args,**kwargs)
        #print(output)
        return output
    return wrapper