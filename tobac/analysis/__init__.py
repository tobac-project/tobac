"""Provide tools to analyse and visualize the tracked objects.
This module provides a set of routines that enables performing analyses
and deriving statistics for individual tracks, such as the time series
of integrated properties and vertical profiles. It also provides
routines to calculate summary statistics of the entire population of
tracked features in the field like histograms of areas/volumes
or mass and a detailed cell lifetime analysis. These analysis
routines are all built in a modular manner. Thus, users can reuse the
most basic methods for interacting with the data structure of the
package in their own analysis procedures in Python. This includes
functions performing simple tasks like looping over all identified
objects or trajectories and masking arrays for the analysis of
individual features. Plotting routines include both visualizations
for individual convective cells and their properties. [1]_

References
----------
.. Heikenfeld, M., Marinescu, P. J., Christensen, M.,
   Watson-Parris, D., Senf, F., van den Heever, S. C.
   & Stier, P. (2019). tobac 1.2: towards a flexible
   framework for tracking and analysis of clouds in
   diverse datasets. Geoscientific Model Development,
   12(11), 4551-4570.

Notes
-----
"""

from tobac.analysis.cell_analysis import *
from tobac.analysis.feature_analysis import *
from tobac.analysis.spatial import *
