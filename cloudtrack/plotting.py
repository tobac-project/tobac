import matplotlib.pyplot as plt
import logging
from .analysis import lifetime_histogram
from .analysis import histogram_cellwise,histogram_featurewise

import numpy as np

def plot_tracks_mask_field_loop(track,field,mask,features,axes=None,name=None,plot_dir='./',
                                figsize=(10./2.54,10./2.54),dpi=300,
                                margin_left=0.05,margin_right=0.05,margin_bottom=0.05,margin_top=0.05,
                                **kwargs):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import os
    from iris import Constraint
    os.makedirs(plot_dir,exist_ok=True)
    time=mask.coord('time')
    if name is None:
        name=field.name()
    for time_i in time.points:
        datetime_i=time.units.num2date(time_i)
        constraint_time = Constraint(time=datetime_i)
        fig1,ax1=plt.subplots(ncols=1, nrows=1,figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
        datestring_file=datetime_i.strftime('%Y-%m-%d_%H:%M:%S')
        field_i=field.extract(constraint_time)
        mask_i=mask.extract(constraint_time)
        track_i=track[track['time']==datetime_i]
        features_i=features[features['time']==datetime_i]
        ax1=plot_tracks_mask_field(track=track_i,field=field_i,mask=mask_i,features=features_i,
                                   axes=ax1,**kwargs)        
        fig1.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=1-margin_top)
        os.makedirs(plot_dir, exist_ok=True)
        savepath_png=os.path.join(plot_dir,name+'_'+datestring_file+'.png')
        fig1.savefig(savepath_png,dpi=dpi)
        logging.debug('Figure plotted to ' + str(savepath_png))

        plt.close()
    plt.close() 

def plot_tracks_mask_field(track,field,mask,features,axes=None,axis_extent=None,
                           plot_outline=True,
                           plot_marker=True,marker_track='x',markersize_track=4,
                           plot_number=True,
                           plot_features=False,marker_feature=None,markersize_feature=None,
                           title=None,
                           vmin=None,vmax=None,n_levels=50,
                           cmap='viridis',extend='neither',
                           orientation_colorbar='horizontal',pad_colorbar=0.05,label_colorbar=None
                           ):
    import cartopy
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import iris.plot as iplt
    from matplotlib.ticker import MaxNLocator
    import cartopy.feature as cfeature
    import numpy as np
    from cloudtrack import mask_particle,mask_particle_surface
    from matplotlib import ticker
    
    if type(axes) is not cartopy.mpl.geoaxes.GeoAxesSubplot:
        raise ValueError('axes had to be cartopy.mpl.geoaxes.GeoAxesSubplot')

    
    datestring=field.coord('time').units.num2date(field.coord('time').points[0]).strftime('%Y-%m-%d %H:%M:%S')
    if title is None:
        titlestring=datestring
    elif type(title is str):
        titlestring=title+ '   ' + datestring
    else:
        raise ValueError('title must be str')
    axes.set_title(titlestring,fontsize=10)
    
    gl = axes.gridlines(draw_labels=True)
    majorLocator = MaxNLocator(nbins=5,steps=[1,2,5,10])
    gl.xlocator=majorLocator
    gl.ylocator=majorLocator
    gl.xformatter = LONGITUDE_FORMATTER
    axes.tick_params(axis='both', which='major', labelsize=6)
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_right = False
    axes.coastlines('10m')    
    #    rivers=cfeature.NaturalEarthFeature(category='physical', name='rivers_lake_centerlines',scale='10m',facecolor='none')
    lakes=cfeature.NaturalEarthFeature(category='physical', name='lakes',scale='10m',facecolor='none')
    axes.add_feature(lakes, edgecolor='black')
    axes.set_xlabel('longitude')
    axes.set_ylabel('latitude')  

    # Plot the background field
    if np.any(~np.isnan(field.data)): # check if field to plot is not only nan, which causes error:

        plot_field=iplt.contourf(field,coords=['longitude','latitude'],
                            levels=np.linspace(vmin,vmax,num=n_levels),extend=extend,
                            axes=axes,
                            cmap=cmap,vmin=vmin,vmax=vmax
                            )
        # greate colorbar for background field:
        cbar=plt.colorbar(plot_field,orientation=orientation_colorbar, pad=pad_colorbar,ax=axes)
        if label_colorbar is None:
            label_colorbar=field.name()+ '('+field.units.symbol +')'
        if orientation_colorbar is 'horizontal':
            cbar.ax.set_xlabel(label_colorbar) 
        elif orientation_colorbar is 'vertical':
            cbar.ax.set_ylabel(label_colorbar) 
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
    

    colors_mask=['darkred','orange','crimson','red','darkorange']
    
    #if marker_feature is not explicitly given, set it to marker_track (will then be overwritten by the coloured markers)
    if marker_feature is None:
        maker_feature=marker_track
    if markersize_feature is None:
        makersize_feature=markersize_track

    #Plot the identified features by looping over rows of DataFrame:
    if plot_features:
        for i_row,row in features.iterrows():
            axes.plot(row['longitude'],row['latitude'],
                      color='grey',marker=maker_feature,markersize=makersize_feature)

    # restrict features to featues inside axis extent  
    track=track.loc[(track['longitude'] > axis_extent[0])  
                  & (track['longitude'] < axis_extent[1]) 
                  & (track['latitude'] > axis_extent[2]) 
                  & (track['latitude'] < axis_extent[3])]


    #Plot tracked features by looping over rows of Dataframe
    for i_row,row in track.iterrows():
        if 'particle' in row:
            particle=row['particle']
            color=colors_mask[int(particle%len(colors_mask))]
        
            if plot_number:        
                particle_string='  '+str(int(row['particle']))
                axes.text(row['longitude'],row['latitude'],particle_string,color=color,fontsize=6)
            if plot_outline:
                mask_i=None
                # if mask is 3D, create surface projection, if mask is 2D keep the mask
                if mask.ndim==2:
                    mask_i=mask_particle(mask,particle,masked=False)
                elif mask.ndim==3:
                    mask_i=mask_particle_surface(mask,particle,masked=False,z_coord='model_level_number')
                else:
                    raise ValueError('mask has shape that cannot be understood')
                # plot countour lines around the edges of the mask    
                iplt.contour(mask_i,coords=['longitude','latitude'],
                             levels=[0,particle],colors=color,axes=axes)
        else:
            color='grey'
        
        if plot_marker:
            axes.plot(row['longitude'],row['latitude'],
                      color=color,marker=marker_track,markersize=markersize_track)

    axes.set_extent(axis_extent)

    return axes

def plot_mask_cell_track_follow(particle,track, cog, features, mask_total,
                                    field_1, field_2, 
                                    field_1_label=None,field_2_label=None,
                                    width=10000,
                                    name= 'test', plotdir='./',
                                    n_core=1,file_format=['png'],figsize=(10/2.54, 10/2.54),dpi=300,
                                    **kwargs):
    '''Make plots for all cells centred around cell and with one background field as filling and one background field as contrours
    Input:
    Output:
    '''
    from iris import Constraint
    from numpy import unique
    import os
    track_cell=track[track['particle']==particle]
    for i_row,row in track_cell.iterrows():
        
        
        constraint_time = Constraint(time=row['time'])
        constraint_x = Constraint(projection_x_coordinate = lambda cell: row['projection_x_coordinate']-width < cell < row['projection_x_coordinate']+width)
        constraint_y = Constraint(projection_y_coordinate = lambda cell: row['projection_y_coordinate']-width < cell < row['projection_y_coordinate']+width)
        constraint = constraint_time & constraint_x & constraint_y
        mask_total_i=mask_total.extract(constraint)
        if field_1 is None:
            field_1_i=None
        else:
            field_1_i=field_1.extract(constraint)
        if field_2 is None:
            field_2_i=None
        else:
            field_2_i=field_2.extract(constraint)

        cells=list(unique(mask_total_i.core_data()))
        if particle not in cells:
            cells.append(particle)
        if 0 in cells:    
            cells.remove(0)
        track_i=track[track['particle'].isin(cells)]
        track_i=track_i[track_i['time']==row['time']]
        if cog is None:
           cog_i=None
        else:
            cog_i=cog[cog['particle'].isin(cells)]
            cog_i=cog_i[cog_i['time']==row['time']]
            
        if features is None:
            features_i=None
        else:
            features_i=features[features['time']==row['time']]


        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        fig1.subplots_adjust(left=0.2, bottom=0.15, right=0.85, top=0.80)
        

        
        
        datestring_stamp = row['time'].strftime('%Y-%m-%d %H:%M:%S')
        celltime_stamp = "%02d:%02d:%02d" % (row['time_cell'].dt.total_seconds() // 3600,(row['time_cell'].dt.total_seconds() % 3600) // 60, row['time_cell'].dt.total_seconds()  % 60 )
        title=celltime_stamp + ' , ' + datestring_stamp
        datestring_file = row['time'].strftime('%Y-%m-%d_%H%M%S')

        ax1=plot_mask_cell_individual_follow(particle_i=particle,track=track_i, cog=cog_i,features=features_i,
                                       mask_total=mask_total_i,
                                       field_1=field_1_i, field_2=field_2_i,
                                       field_1_label=field_1_label, field_2_label=field_2_label,
                                       width=width,
                                       axes=ax1,title=title,
                                       **kwargs)
               
        out_dir = os.path.join(plotdir, name)
        os.makedirs(out_dir, exist_ok=True)
        if 'png' in file_format:
            savepath_png = os.path.join(out_dir, name  + '_' + datestring_file + '.png')
            fig1.savefig(savepath_png, dpi=dpi)
            logging.debug('field_1 field_2 Mask plot saved to ' + savepath_png)
        if 'pdf' in file_format:
            savepath_pdf = os.path.join(out_dir, name  + '_' + datestring_file + '.pdf')
            fig1.savefig(savepath_pdf, dpi=dpi)
            logging.debug('field_1 field_2 Mask plot saved to ' + savepath_pdf)
        plt.close()
        plt.clf()


def plot_mask_cell_individual_follow(particle_i,track, cog,features, mask_total,
                               field_1, field_2, width=10000,
                               field_1_label=None, field_2_label=None,
                               axes=plt.gca(),
                               field_1_cmap='Blues',
                               vmin_field_1=0,vmax_field_1=50,levels_field_1=None,nlevels_1=10,
                               contour_labels=False,
                               field_2_cmap='summer',
                               vmin_field_2=0,vmax_field_2=100,levels_field_2=None,nlevels_2=10,
                               title=None
                               ): 
    '''Make individual plot for cell centred around cell and with one background field as filling and one background field as contrours
    Input:
    Output:
    '''
    import numpy as np
    from cloudtrack.segmentation  import mask_particle_surface
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import Normalize
    
    
    divider = make_axes_locatable(axes)
    
    x_pos=track[track['particle']==particle_i]['projection_x_coordinate'].item()
    y_pos=track[track['particle']==particle_i]['projection_y_coordinate'].item()
    if field_2 is not None:
        if levels_field_2 is None:
            levels_field_2=np.linspace(vmin_field_2,vmax_field_2, nlevels_2)
        plot_field_2 = axes.contourf((field_2.coord('projection_x_coordinate').points-x_pos)/1000,
                                 (field_2.coord('projection_y_coordinate').points-y_pos)/1000,
                                 field_2.data,
                                 levels=levels_field_2,
                                 cmap=field_2_cmap, vmin=vmin_field_2, vmax=vmax_field_2)    
        
        
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        norm1= Normalize(vmin=vmin_field_2, vmax=vmax_field_2)
        sm1= plt.cm.ScalarMappable(norm=norm1, cmap = plot_field_2.cmap)
        sm1.set_array([])
        
        cbar_field_2 = plt.colorbar(sm1, orientation='vertical',cax=cax1)
        cbar_field_2.ax.set_ylabel(field_2_label)
        cbar_field_2.set_clim(vmin_field_2, vmax_field_2)

    if field_1 is not None:
        if levels_field_1 is None:
            levels_field_1=np.linspace(vmin_field_1, vmax_field_1, nlevels_1)
        plot_field_1 = axes.contour((field_1.coord('projection_x_coordinate').points-x_pos)/1000,
                                  (field_1.coord('projection_y_coordinate').points-y_pos)/1000,
                                  field_1.data,
                                  cmap=field_1_cmap,
                                  levels=levels_field_1,vmin=vmin_field_1, vmax=vmax_field_1,
                                  linewidths=0.8)
        
        if contour_labels:
            axes.clabel(plot_field_1, fontsize=10)
    
        cax2 = divider.append_axes("bottom", size="5%", pad=0.1)
        norm2= Normalize(vmin=vmin_field_1, vmax=vmax_field_1)
        sm2= plt.cm.ScalarMappable(norm=norm2, cmap = plot_field_1.cmap)
        sm2.set_array([])
        cbar_w = plt.colorbar(sm2, orientation='horizontal',cax=cax2)
        cbar_w.ax.set_xlabel(field_1_label)
        cbar_w.set_clim(vmin_field_1, vmax_field_1)


    
    for i_row, row in track.iterrows():
        particle = int(row['particle'])
        if particle==particle_i:
            color='darkred'
        else:
            color='darkorange'
            
        particle_string='   '+str(int(row['particle']))
        axes.text((row['projection_x_coordinate']-x_pos)/1000,
                  (row['projection_y_coordinate']-y_pos)/1000,
                  particle_string,color=color,fontsize=6)

        # Plot marker for tracked cell centre as a cross
        axes.plot((row['projection_x_coordinate']-x_pos)/1000,
                  (row['projection_y_coordinate']-y_pos)/1000,
                  'x', color=color,markersize=4)
        
        
        #Create surface projection of mask for the respective cell and plot it in the right color
        z_coord = 'model_level_number'
        if len(mask_total.shape)==3: 
            mask_total_i_surface = mask_particle_surface(mask_total, particle, masked=False, z_coord=z_coord)
        elif len(mask_total.shape)==2:            
            mask_total_i_surface=mask_total
        axes.contour((mask_total_i_surface.coord('projection_x_coordinate').points-x_pos)/1000,
                     (mask_total_i_surface.coord('projection_y_coordinate').points-y_pos)/1000,
                     mask_total_i_surface.data, 
                     levels=[0, particle], colors=color, linestyles=':',linewidth=1)

    if cog is not None:

        for i_row, row in cog.iterrows():
            particle = row['particle']
            
            if particle==particle_i:
                color='darkred'
            else:
                color='darkorange'
            # plot marker for centre of gravity as a circle    
            axes.plot((row['x_M']-x_pos)/1000, (row['y_M']-y_pos)/1000,
                      'o', markeredgecolor=color, markerfacecolor='None',markersize=4)
    
    if features is not None:

        for i_row, row in features.iterrows():
            color='purple'
            axes.plot((row['projection_x_coordinate']-x_pos)/1000,
                      (row['projection_y_coordinate']-y_pos)/1000,
                      '+', color=color,markersize=3)

    axes.set_xlabel('x (km)')
    axes.set_ylabel('y (km)')        
    axes.set_xlim([-1*width/1000, width/1000])
    axes.set_ylim([-1*width/1000, width/1000])
    axes.xaxis.set_label_position('top') 
    axes.xaxis.set_ticks_position('top')
    axes.set_title(title,pad=35,fontsize=10)
 
    return axes

def plot_mask_cell_track_static(particle,track, cog, features, mask_total,
                                    field_1, field_2,
                                    field_1_label=None, field_2_label=None,
                                    width=10000,n_extend=1,
                                    name= 'test', plotdir='./',
                                    n_core=1,file_format=['png'],figsize=(10/2.54, 10/2.54),dpi=300,
                                    **kwargs):
    '''Make plots for all cells with fixed frame including entire development of the cell and with one background field as filling and one background field as contrours
    Input:
    Output:
    '''
    from iris import Constraint
    from numpy import unique
    import os
    track_cell=track[track['particle']==particle]
    x_min=track_cell['projection_x_coordinate'].min()-width
    x_max=track_cell['projection_x_coordinate'].max()+width
    y_min=track_cell['projection_y_coordinate'].min()-width
    y_max=track_cell['projection_y_coordinate'].max()+width
    
    #set up looping over time based on mask's time coordinate to allow for one timestep before and after the track
    time_coord=mask_total.coord('time')
    time=time_coord.units.num2date(time_coord.points)
    i_start=max(0,np.where(time==track_cell['time'].values[0])[0][0]-n_extend)
    i_end=min(len(time)-1,np.where(time==track_cell['time'].values[-1])[0][0]+n_extend+1)
    time_cell=time[slice(i_start,i_end)]
    for time_i in time_cell:

#    for i_row,row in track_cell.iterrows():
#        time_i=row['time']
#        constraint_time = Constraint(time=row['time'])
        constraint_time = Constraint(time=time_i)

        constraint_x = Constraint(projection_x_coordinate = lambda cell: x_min < cell < x_max)
        constraint_y = Constraint(projection_y_coordinate = lambda cell: y_min < cell < y_max)
        constraint = constraint_time & constraint_x & constraint_y            

        mask_total_i=mask_total.extract(constraint)
        if field_1 is None:
            field_1_i=None
        else:
            field_1_i=field_1.extract(constraint)
        if field_2 is None:
            field_2_i=None
        else:
            field_2_i=field_2.extract(constraint)

        
        track_i=track[track['time']==time_i]

        cells_mask=list(unique(mask_total_i.core_data()))
        track_cells=track_i.loc[(track_i['projection_x_coordinate'] > x_min)  & (track_i['projection_x_coordinate'] < x_max) & (track_i['projection_y_coordinate'] > y_min) & (track_i['projection_y_coordinate'] < y_max)]
        cells_track=list(track_cells['particle'].values)
        cells=list(set( cells_mask + cells_track ))
        if particle not in cells:
            cells.append(particle)
        if 0 in cells:    
            cells.remove(0)
        track_i=track_i[track_i['particle'].isin(cells)]
        
        if cog is None:
            cog_i=None
        else:
            cog_i=cog[cog['particle'].isin(cells)]
            cog_i=cog_i[cog_i['time']==time_i]

        if features is None:
            features_i=None
        else:
            features_i=features[features['time']==time_i]


        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        fig1.subplots_adjust(left=0.2, bottom=0.15, right=0.80, top=0.85)
        
        datestring_stamp = time_i.strftime('%Y-%m-%d %H:%M:%S')
        if time_i in track_cell['time'].values:
            time_cell_i=track_cell[track_cell['time'].values==time_i]['time_cell']
            celltime_stamp = "%02d:%02d:%02d" % (time_cell_i.dt.total_seconds() // 3600,
                                             (time_cell_i.dt.total_seconds() % 3600) // 60,
                                             time_cell_i.dt.total_seconds()  % 60 )
        else:
            celltime_stamp=' - '
        title=celltime_stamp + ' , ' + datestring_stamp
        datestring_file = time_i.strftime('%Y-%m-%d_%H%M%S')

        ax1=plot_mask_cell_individual_static(particle_i=particle,
                                             track=track_i, cog=cog_i,features=features_i, 
                                             mask_total=mask_total_i,
                                             field_1=field_1_i, field_2=field_2_i,
                                             field_1_label=field_1_label, field_2_label=field_2_label,
                                             xlim=[x_min/1000,x_max/1000],ylim=[y_min/1000,y_max/1000],
                                             axes=ax1,title=title,**kwargs)
        
        out_dir = os.path.join(plotdir, name)
        os.makedirs(out_dir, exist_ok=True)
        if 'png' in file_format:
            savepath_png = os.path.join(out_dir, name  + '_' + datestring_file + '.png')
            fig1.savefig(savepath_png, dpi=dpi)
            logging.debug('Mask static plot saved to ' + savepath_png)
        if 'pdf' in file_format:
            savepath_pdf = os.path.join(out_dir, name  + '_' + datestring_file + '.pdf')
            fig1.savefig(savepath_pdf, dpi=dpi)
            logging.debug('Mask static plot saved to ' + savepath_pdf)
        plt.close()
        plt.clf()


def plot_mask_cell_individual_static(particle_i,track, cog, features, mask_total,
                               field_1, field_2,
                               field_1_label=None,
                               axes=plt.gca(),xlim=None,ylim=None,
                               field_2_label=None,                                                             
                               field_1_cmap='Blues',
                               vmin_field_1=0,vmax_field_1=50,levels_field_1=None,nlevels_1=10,
                               contour_labels=False,
                               field_2_cmap='summer',
                               vmin_field_2=0,vmax_field_2=100,levels_field_2=None,nlevels_2=10,
                               title=None
                               ):  
    '''Make plots for cell in fixed frame and with one background field as filling and one background field as contrours
    Input:
    Output:
    '''

    import numpy as np
    from cloudtrack.segmentation  import mask_particle_surface
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import Normalize
    
    
    divider = make_axes_locatable(axes)
    
    if field_2 is not None:
        if levels_field_2 is None:
            levels_field_2=np.linspace(vmin_field_2,vmax_field_2, 10)
        plot_field_2 = axes.contourf(field_2.coord('projection_x_coordinate').points/1000,
                                 field_2.coord('projection_y_coordinate').points/1000,
                                 field_2.data,
                                 levels=levels_field_2, 
                                 cmap=field_2_cmap, vmin=vmin_field_2, vmax=vmax_field_2)    
        
        
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        norm1= Normalize(vmin=vmin_field_2, vmax=vmax_field_2)
        sm1= plt.cm.ScalarMappable(norm=norm1, cmap = plot_field_2.cmap)
        sm1.set_array([])
        
        cbar_field_2 = plt.colorbar(sm1, orientation='vertical',cax=cax1)
        cbar_field_2.ax.set_ylabel(field_2_label)
        cbar_field_2.set_clim(vmin_field_2, vmax_field_2)

    if field_1 is not None:
        if levels_field_1 is None:
            levels_field_1=np.linspace(vmin_field_1, vmax_field_1, 5)
        plot_field_1 = axes.contour(field_1.coord('projection_x_coordinate').points/1000,
                                  field_1.coord('projection_y_coordinate').points/1000,
                                  field_1.data,
                                  cmap=field_1_cmap,
                                  levels=levels_field_1,vmin=vmin_field_1, vmax=vmax_field_1,
                                  linewidths=0.8)
        
        if contour_labels:
            axes.clabel(plot_field_1, fontsize=10)
    
        cax2 = divider.append_axes("bottom", size="5%", pad=0.1)
        norm2= Normalize(vmin=vmin_field_1, vmax=vmax_field_1)
        sm2= plt.cm.ScalarMappable(norm=norm2, cmap = plot_field_1.cmap)
        sm2.set_array([])
        cbar_w = plt.colorbar(sm2, orientation='horizontal',cax=cax2)
        cbar_w.ax.set_xlabel(field_1_label)
        cbar_w.set_clim(vmin_field_1, vmax_field_1)


    
    for i_row, row in track.iterrows():
        particle = int(row['particle'])
        if particle==particle_i:
            color='darkred'
        else:
            color='darkorange'
            
        particle_string='   '+str(int(row['particle']))
        axes.text(row['projection_x_coordinate']/1000,
                  row['projection_y_coordinate']/1000,
                  particle_string,color=color,fontsize=6)

        # Plot marker for tracked cell centre as a cross
        axes.plot(row['projection_x_coordinate']/1000,
                  row['projection_y_coordinate']/1000,
                  'x', color=color,markersize=4)
        
        
        #Create surface projection of mask for the respective cell and plot it in the right color
        z_coord = 'model_level_number'
        if len(mask_total.shape)==3: 
            mask_total_i_surface = mask_particle_surface(mask_total, particle, masked=False, z_coord=z_coord)
        elif len(mask_total.shape)==2:            
            mask_total_i_surface=mask_total
        axes.contour(mask_total_i_surface.coord('projection_x_coordinate').points/1000,
                     mask_total_i_surface.coord('projection_y_coordinate').points/1000,
                     mask_total_i_surface.data, 
                     levels=[0, particle], colors=color, linestyles=':',linewidth=1)
    if cog is not None:
    
        for i_row, row in cog.iterrows():
            particle = row['particle']
            
            if particle==particle_i:
                color='darkred'
            else:
                color='darkorange'
            # plot marker for centre of gravity as a circle    
            axes.plot(row['x_M']/1000, row['y_M']/1000,
                      'o', markeredgecolor=color, markerfacecolor='None',markersize=4)

    if features is not None:
    
        for i_row, row in features.iterrows():
            color='purple'
            axes.plot(row['projection_x_coordinate']/1000,
                      row['projection_y_coordinate']/1000,
                      '+', color=color,markersize=3)

    axes.set_xlabel('x (km)')
    axes.set_ylabel('y (km)')        
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.xaxis.set_label_position('top') 
    axes.xaxis.set_ticks_position('top')
    axes.set_title(title,pad=35,fontsize=10)

    return axes

def plot_mask_cell_track_static_timeseries(particle,track, cog, features, mask_total,
                                           field_1, field_2,
                                           field_1_label=None, field_2_label=None,
                                           track_variable=None,variable=None,variable_label=None,
                                           width=10000,n_extend=1,
                                           name= 'test', plotdir='./',
                                           n_core=1,file_format=['png'],figsize=(20/2.54, 10/2.54),dpi=300,
                                           **kwargs):
    '''Make plots for all cells with fixed frame including entire development of the cell and with one background field as filling and one background field as contrours
    Input:
    Output:
    '''
    '''Make plots for all cells with fixed frame including entire development of the cell and with one background field as filling and one background field as contrours
    Input:
    Output:
    '''
    from iris import Constraint
    from numpy import unique
    import os
    import pandas as pd
    track_cell=track[track['particle']==particle]
    x_min=track_cell['projection_x_coordinate'].min()-width
    x_max=track_cell['projection_x_coordinate'].max()+width
    y_min=track_cell['projection_y_coordinate'].min()-width
    y_max=track_cell['projection_y_coordinate'].max()+width
    time_min=track_cell['time'].min()
    time_max=track_cell['time'].max()

    track_variable_cell=track_variable[track_variable['particle']==particle]
    track_variable_cell['time_cell']=pd.to_timedelta(track_variable_cell['time_cell'])
#    track_variable_cell=track_variable_cell[(track_variable_cell['time']>=time_min) & (track_variable_cell['time']<=time_max)] 

    #set up looping over time based on mask's time coordinate to allow for one timestep before and after the track
    time_coord=mask_total.coord('time')
    time=time_coord.units.num2date(time_coord.points)
    i_start=max(0,np.where(time==track_cell['time'].values[0])[0][0]-n_extend)
    i_end=min(len(time)-1,np.where(time==track_cell['time'].values[-1])[0][0]+n_extend+1)
    time_cell=time[slice(i_start,i_end)]
    for time_i in time_cell:
                
        constraint_time = Constraint(time=time_i)
        constraint_x = Constraint(projection_x_coordinate = lambda cell: x_min < cell < x_max)
        constraint_y = Constraint(projection_y_coordinate = lambda cell: y_min < cell < y_max)
        constraint = constraint_time & constraint_x & constraint_y            

        mask_total_i=mask_total.extract(constraint)
        if field_1 is None:
            field_1_i=None
        else:
            field_1_i=field_1.extract(constraint)
        if field_2 is None:
            field_2_i=None
        else:
            field_2_i=field_2.extract(constraint)

        
        track_i=track[track['time']==time_i]
        
        cells_mask=list(unique(mask_total_i.core_data()))
        track_cells=track_i.loc[(track_i['projection_x_coordinate'] > x_min)  & (track_i['projection_x_coordinate'] < x_max) & (track_i['projection_y_coordinate'] > y_min) & (track_i['projection_y_coordinate'] < y_max)]
        cells_track=list(track_cells['particle'].values)
        cells=list(set( cells_mask + cells_track ))
        if particle not in cells:
            cells.append(particle)
        if 0 in cells:    
            cells.remove(0)
        track_i=track_i[track_i['particle'].isin(cells)]
        
        if cog is None:
            cog_i=None
        else:
            cog_i=cog[cog['particle'].isin(cells)]
            cog_i=cog_i[cog_i['time']==time_i]

        if features is None:
            features_i=None
        else:
            features_i=features[features['time']==time_i]


        fig1, ax1 = plt.subplots(ncols=2, nrows=1, figsize=figsize)
        fig1.subplots_adjust(left=0.1, bottom=0.15, right=0.90, top=0.85,wspace=0.3)
        
        datestring_stamp = time_i.strftime('%Y-%m-%d %H:%M:%S')
        if time_i in track_cell['time'].values:
            time_cell_i=track_cell[track_cell['time'].values==time_i]['time_cell']
            celltime_stamp = "%02d:%02d:%02d" % (time_cell_i.dt.total_seconds() // 3600,
                                             (time_cell_i.dt.total_seconds() % 3600) // 60,
                                             time_cell_i.dt.total_seconds()  % 60 )
        else:
            celltime_stamp=' - '
        title=celltime_stamp + ' , ' + datestring_stamp
        datestring_file = time_i.strftime('%Y-%m-%d_%H%M%S')

        # plot evolving timeseries of variable to second axis:
        ax1[0]=plot_mask_cell_individual_static(particle_i=particle,
                                             track=track_i, cog=cog_i,features=features_i, 
                                             mask_total=mask_total_i,
                                             field_1=field_1_i, field_2=field_2_i,
                                             field_1_label=field_1_label, field_2_label=field_2_label,
                                             xlim=[x_min/1000,x_max/1000],ylim=[y_min/1000,y_max/1000],
                                             axes=ax1[0],title=title,**kwargs)
        
        track_variable_past=track_variable_cell[(track_variable_cell['time']>=time_min)  & (track_variable_cell['time']<=time_i)]       
        track_variable_current=track_variable_cell[track_variable_cell['time']==time_i]   
        
        ax1[1].plot(track_variable_past['time_cell'].dt.total_seconds()/ 60.,track_variable_past[variable].values,color='navy',linestyle='-')
        ax1[1].plot(track_variable_current['time_cell'].dt.total_seconds()/ 60.,track_variable_current[variable].values,color='navy',marker='o',markersize=4,fillstyle='full')
        ax1[1].yaxis.tick_right()
        ax1[1].yaxis.set_label_position("right")
        ax1[1].set_xlim([0,2*60])
        ax1[1].set_xticks(np.arange(0,120,15))
        ax1[1].set_ylim([0,max(10,1.1*track_variable_cell[variable].max())])
        ax1[1].set_xlabel('cell lifetime (min)')
        if variable_label==None:
            variable_label=variable
        ax1[1].set_ylabel(variable_label)

        out_dir = os.path.join(plotdir, name)
        os.makedirs(out_dir, exist_ok=True)
        if 'png' in file_format:
            savepath_png = os.path.join(out_dir, name  + '_' + datestring_file + '.png')
            fig1.savefig(savepath_png, dpi=dpi)
            logging.debug('Mask static plot saved to ' + savepath_png)
        if 'pdf' in file_format:
            savepath_pdf = os.path.join(out_dir, name  + '_' + datestring_file + '.pdf')
            fig1.savefig(savepath_pdf, dpi=dpi)
            logging.debug('Mask static plot saved to ' + savepath_pdf)
        plt.close()
        plt.clf()





def plot_lifetime_histogram(track,axes=plt.gca(),bin_edges=np.arange(0,200,20),density=False,**kwargs):
    hist, bin_edges = lifetime_histogram(track,bin_edges=bin_edges,density=density)
    plot_hist=axes.plot(bin_edges[:-1]+0.5*np.diff(bin_edges), hist,**kwargs)
    return plot_hist

def plot_lifetime_histogram_bar(track,axes=plt.gca(),bin_edges=np.arange(0,200,20),density=False,width_bar=1,shift=0.5,**kwargs):
    hist, bin_edges = lifetime_histogram(track,bin_edges=bin_edges,density=density)
    plot_hist=axes.bar(bin_edges[:-1]+0.5*np.diff(bin_edges)+shift,**kwargs)
    return plot_hist

def plot_histogram_cellwise(track,bin_edges,variable,quantity,axes=plt.gca(),density=False,**kwargs):
    hist, bin_edges = histogram_cellwise(track,bin_edges=bin_edges,variable=variable,quantity=quantity,density=density)
    plot_hist=axes.plot(bin_edges[:-1]+0.5*np.diff(bin_edges), hist,**kwargs)
    return plot_hist

def plot_histogram_featurewise(Track,bin_edges,variable,axes=plt.gca(),density=False,**kwargs):
    hist, bin_edges = histogram_featurewise(Track,bin_edges=bin_edges,variable=variable,density=density)
    plot_hist=axes.plot(bin_edges[:-1]+0.5*np.diff(bin_edges), hist,**kwargs)
    return plot_hist
