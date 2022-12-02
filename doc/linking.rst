Linking
-------
Currently implemented options for linking detected features into tracks:

**Trackpy:**

This method uses the trackpy library (http://soft-matter.github.io/trackpy). 
This approach only takes the point-like position of the feature, e.g. determined as the weighted mean, into account. Features to link with are looked for in a search radius defined by the parameters v_max or d_max. The position of the center of this search radius is determined by the method keyword. method="random" uses the position of the current feature (t_i), while method="predict" makes use of the information from the linked feature in the previous timestep (t_i-1) to predict the next position. 

        .. image:: images/linking_prediction.png
            :width: 500 px

If there is only one feature in the search radius, the linking can happen immediately. If there are none, the track ends at this timestep. If there are more options, trackpy performs a decision process. Assume there are N features in the current and also in the next timeframe and they are all within each search radius. This means there are N! options for linking. Each of these options means that N distances between the center of the search radius of a current feature and a feature from the next time frame :math:`\delta_n, n=1, 2, ..., N` are traveled by the features. Trackpy will calculate the net squared distance

.. math::

   \sum_{n=1}^{N} \delta_n
   
  for every option and the lowest value is used for linking.
