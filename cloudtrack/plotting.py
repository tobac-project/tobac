import matplotlib.pyplot as plt
import logging

def plot_tracks_mask_field_loop(track,field,Mask,axes=None,axis_extent=None,text_mass_signal=False,vmin=None,vmax=None,n_levels=50,name=None,plot_dir='./',figsize=(10,10)):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import os
    time=field.coord('time')
    if name is None:
        name=field.name()
    for i in range(len(time.points)):
        fig1,ax1=plt.subplots(ncols=1, nrows=1,figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
        datestring_file=time.units.num2date(time.points[i]).strftime('%Y-%m-%d_%H:%M:%S')
        ax1=plot_tracks_mask_field(track[track['frame']==i],field[i],Mask[i],axes=ax1,axis_extent=axis_extent,text_mass_signal=text_mass_signal,vmin=vmin,vmax=vmax,n_levels=n_levels)
        savepath_png=os.path.join(plot_dir,name+'_'+datestring_file+'.png')
        fig1.savefig(savepath_png,dpi=600)
        logging.debug('Figure '+str(i) + ' plotted to ' + str(savepath_png))

        plt.close()
    plt.close() 


def plot_tracks_mask_field(track,field,Mask,axes=None,axis_extent=None,text_mass_signal=False,vmin=None,vmax=None,n_levels=50):
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

    axes.set_title(datestring)
    
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


    axes.set_extent(axis_extent)
    
    plot_field=iplt.contourf(field,coords=['longitude','latitude'],
                        levels=np.linspace(vmin,vmax,num=n_levels),axes=axes,cmap='viridis',vmin=vmin,vmax=vmax)
    
    colors_mask=['darkred','orange','crimson','red','darkorange']
    
    for i_row,row in track.iterrows():        
        particle=row['particle']
        color=colors_mask[int(particle%len(colors_mask))]
        axes.plot(row['longitude'],row['latitude'],color=color,marker='x')
    
    
        if not text_mass_signal:        
            particle_string='     '+str(int(row['particle']))
        if text_mass_signal:        
            particle_string='     '+str(int(row['particle']))+'('+"{0:0.4}".format(row['mass'])+','+"{0:0.4}".format(row['signal'])+')'
    
    axes.text(row['longitude'],row['latitude'],particle_string,color=color,fontsize=6)

    Mask_i=None
    if Mask.ndim==2:
        Mask_i=mask_particle(Mask,particle,masked=False)
    elif Mask.ndim==3:
        Mask_i=mask_particle_surface(Mask,particle,masked=False,z_coord='model_level_number')
    else:
        raise ValueError('mask has shape that cannot be understood')

    iplt.contour(Mask_i,coords=['longitude','latitude'],
                     levels=[0,particle],colors=color,axes=axes)

    cbar=plt.colorbar(plot_field,orientation='horizontal')
    cbar.ax.set_xlabel(field.name()+ '('+field.units.symbol +')') 
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    return axes

