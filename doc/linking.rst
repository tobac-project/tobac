Linking
-------
Currently implemented methods for linking detected features into cloud tracks:

**Trackpy:**

This method uses the trackpy library (http://soft-matter.github.io/trackpy). 
This approach only takes the point-like position of the feature, e.g. determined as the weighted mean, into account and does not use any other information about the identified features into account. The linking makes use of the information from the linked features in the previous timesteps to predict the position and then searches for matching features in a search range determined by the v_max parameter.

        .. image:: images/linking_prediction.png
            :width: 500 px

