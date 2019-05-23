Linking
-------
Currently implemented methods for linking detected features into cloud tracks:

**Trackpy:**

This method uses the trackpy library (http://soft-matter.github.io/trackpy). 
This approach only takes the point-like position of the feature, e.g. determined as the weighted mean, into account and does not use any other information about the identified features into account.

**Current development:**

We are currently actively working on additional options for the tracking of the clouds that take into account the shape of the identified features by evaluating overlap between adjacent time steps, as well as the inclusion of cloud splitting and merging.

