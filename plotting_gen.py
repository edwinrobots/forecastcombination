#Python script to hold functions generally useful for plotting

import numpy as np
import matplotlib.pyplot as plt

#Python function to return color map for contour plots that is blue for negative numbers and red for positive numbers.
#data_min and data_max are the minimum and maximum of the data respectively.
def get_cmap_rb_zero_centred(data_min,data_max):
    from matplotlib.colors import LinearSegmentedColormap

    data_range=float(data_max-data_min)
    if data_min<0 and data_max>0:
        zero_frac=abs(data_min)/data_range  #fraction along colorbar where colour should change from blue to red
    elif data_min>=0: zero_frac=0.
    elif data_max<=0: zero_frac=1.
    #print zero_frac
  
#    cdict = {'red':  ((0.0, 0.0, 0.0),
#                     (zero_frac/2, 0.0, 0.0),
#                     (zero_frac, 0.8, 1.0),
#                     ((1+zero_frac)/2, 1.0, 1.0),
#                     (1.0, 0.4, 1.0)),
#
#           'green': ((0.0, 0.0, 0.0),
#                     (zero_frac/2, 0.0, 0.0),
#                     (zero_frac, 0.9, 0.9),
#                     ((1+zero_frac)/2, 0.0, 0.0),
#                     (1.0, 0.0, 0.0)),
#
#           'blue':  ((0.0, 0.0, 0.4),
#                     (zero_frac/2, 1.0, 1.0),
#                     (zero_frac, 1.0, 0.8),
#                     ((1+zero_frac)/2, 0.0, 0.0),
#                     (1.0, 0.0, 0.0))
#            }

    #Settings that give a colour scale that is less prone to saturation near the ends.
    cdict = {'red':  ((0.0, 0.0, 0.0),
                     (zero_frac/3, 0.0, 0.0),
                     (zero_frac, 0.8, 1.0),
                     ((2+zero_frac)/3, 1.0, 1.0),
                     (1.0, 0.4, 1.0)),

           'green': ((0.0, 0.0, 0.0),
                     (zero_frac/3, 0.0, 0.0),
                     (zero_frac, 0.9, 0.9),
                     ((2+zero_frac)/3, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),

           'blue':  ((0.0, 0.0, 0.4),
                     (zero_frac/3, 1.0, 1.0),
                     (zero_frac, 1.0, 0.8),
                     ((2+zero_frac)/3, 0.0, 0.0),
                     (1.0, 0.0, 0.0))
            }
    

##Trying to get the maximum intensity of each colour to correspond to the relative magnitude of the maximum positive/negative values
#Don't think it is necessary to do this - just setting data_min and data_max to have the same magnitude when one is positive and the other negative seems to have the desired effect.
#    if data_min<0 and data_max>0:
#        data_abs_max=np.max([abs(data_min),data_max])
#        if abs(data_min)>data_max:
#            #red_frac=zero_frac+data_max/2/data_abs_max*(1-zero_frac)
#            red_frac=(1+zero_frac)/2.*abs(data_min)/data_max
#            blue_frac=zero_frac/2
#        else:
#            red_frac=(1+zero_frac)/2
#            #blue_frac=abs(data_min)/2*data_abs_max
#            blue_frac=zero_frac/2.*data_max/abs(data_min)
#  
#    print data_min, data_max, data_abs_max, red_frac, blue_frac, zero_frac
#  
#    cdict = {'red':  ((0.0, 0.0, 0.0),
#                      (blue_frac, 0.0, 0.0),
#                      (zero_frac, 0.8, 1.0),
#                      (np.min([1.0,red_frac]), 1.0, 1.0),
#                      (1.0, 0.4, 1.0)),

#           'green': ((0.0, 0.0, 0.0),
#                     (blue_frac, 0.0, 0.0),
#                     (zero_frac, 0.9, 0.9),
#                     (np.min([1.0,red_frac]), 0.0, 0.0),
#                     (1.0, 0.0, 0.0)),

#           'blue':  ((0.0, 0.0, 0.4),
#                     (blue_frac, 1.0, 1.0),
#                     (zero_frac, 1.0, 0.8),
#                     (np.min([1.0,red_frac]), 0.0, 0.0),
#                     (1.0, 0.0, 0.0))
#          }

    cmap=LinearSegmentedColormap('rb_centred',cdict)

    return cmap


#Function to return levels for a contour plot, including zero if the data includes +ve and -ve values, with nlevels between the max and min data values
def get_levels(data,nlevels):
    from python_gen import round_to_n

#    levmin=round_to_n(np.nanmin(data),2,round_down=1)
#    levmax=round_to_n(np.nanmax(data),2,round_up=1)
#    step = round_to_n((levmax - levmin) / (nlevels-1)
#    #levels = (np.arange(nlevels+2)-1) * step + levmin  #code used in my IDL function get_levels()
#    levels = np.arange(nlevels) * step + levmin

    levmin=round_to_n(np.nanmin(data),2,round_down=1)
    levmax=np.nanmax(data)
    step = round_to_n((levmax - levmin) / (nlevels-1), 2, round_up=1)
    levels = np.arange(nlevels) * step + levmin

    if levmax>0 and levmin<0:
        from python_gen import closest
        zero_ind=closest(levels,0)
        levels=levels-levels[zero_ind]
    
#    levels_min=round_to_n(np.min(levels),2)
#    levels=levels-levels_min  #subtract minimum level before rounding, so that only the level interval is being rounded and not the level values, which could be problematic when the values are much larger than the interval.
#    levels=round_to_n(levels,2)
#    levels=levels+levels_min
    

    return levels

#Function to return units as they should be shown on a plot
def get_unit_plot(unit):
    unit_plot=unit.replace('**','^')
    return unit_plot
    

#Function to adjust y-axis of a line plot so that the legend does not cover the lines
#ax is a matplotlib axis instance, leg is a matplotlib.pyplot.legend instance, and leg_loc is the keyword given to pyplot.legend to specify the location of the legend on the plot
#Have only written this for legends placed in the upper right so far
#Could make it so that axis range is rescaled if the plot is manually dragged?
def make_space_for_legend(ax,leg,leg_loc):

    ymin,ymax=ax.get_ybound()  #original data interval
    
    if leg_loc=='upper right' or leg_loc==1:
        for i in range(3):  #iterate setting the y-limits of the plot and redrawing the legend, and hopefully it will converge to a sensible legend position.
            plt.draw()  #do this first to make sure that the legend is on the plot to start
            #yrange=plt.ylim()[1]-plt.ylim()[0]
            #plt.ylim(plt.ylim()[0],ylim_orig[1]+leg.get_window_extent().transformed(leg.axes.transAxes.inverted()).height*yrange)
            yrange=ax.get_ybound()[1]-ax.get_ybound()[0]
            ax.set_ybound(ax.get_ybound()[0], ymax+yrange*0.05+leg.get_window_extent().transformed(leg.axes.transAxes.inverted()).height*yrange)


#Function to plot 2D data, where multiple arrays can be plotted as subplots in the same figure, each with potentially an array represented with line contours, filled contours, vectors and stippling. Currently this will only work if all plots are the same type (latlon etc.). Arrays on the same subplot need to be on the same axes in general. Returns a 2-element tuple: the figure object and the array of axis objects.
#xvals_list and yvals_list are either arrays containing the x- and y-coordinate values (if these are the same for all subplots) or lists containing the values for the separate subplots - set these if all data to be plotted on each subplot has the same axes. If the data have different axes, set x/yvals_contour_list, x/yvals_filled_list, x/yvals_vect_list and x/yvals_hatch_list separately.
#contour_list and filled_list are each either a single 2D array or a list of 2D arrays containing the data to be plotted as a line/filled contour plot respectively - each array will be plotted on a separate subplot, and should share the same axes. They may be specified together so the contours are plotted on top of the filled plot - in this case both lists should have the same length.
#levels_contour_list and levels_filled_list are either an array containing the contour levels to use for all of the contour and filled plots respectively, or a list containing arrays with the levels for each subplot, or 'equal' to specify that the levels should be chosen to be the same for each subplot, or None to use levels calculated appropriately for single plots. When specifying one set of levels to be used for all the plots, make it a list or array within a list e.g. [[-2,-1,0,1,2]].
#nlevels_contour and nlevels_filled are the number of levels to plot for the contour/filled plots, if levels_filled/levels_contour is not set.
#Set clip_levels_contour and clip_levels_filled to clip the maximum and minimum level to be no more than 3 standard deviations of the whole field from the mean - useful when a few points have very large or small values.
#contour_col is the colour of the line contours.
#contour_linewidths is the width of the line contours ("1" is the matplotlib default).
#clabel_format is the format code for line contour labels.
#unit_filled_list is a string specifying the unit to add to the colour bar on filled contour plots if this is the same for each subplot, or a list of strings corresponding to each array in the data lists.
#vect_list specifies data to use to overplot arrows: it is a list of 2-element lists, where each element is a 2D array of data specifying the x- and y-components of the data respectively. This doesn't seem to work well with stereographic projection. Currently this assumes that all the vectors are on the axes described by xvals_list[0] and yvals_list[0].
#vect_scale_list sets the data value represented by a reference arrow - this number should be of a similar magnitude to the data being plotted to give reasonable-looking arrows. Set as a number to give the same scale for all subplots, as a list of numbers to set different scales for each subplot, as 'equal' to get the code to calculate an appropriate scale for all subplots, or None to calculate an appropriate scale for each subplot separately.
#unit_vect_list gives units for vectors - set as a string to give the same unit for both components or as a 2-element list to give different units to the x- and y-components that is the same for each subplot, as a list of such strings or 2-element lists to give different units to each subplot. 
#Set no_vect_legend to not plot the size of the reference arrow alongside the arrow.
#Set no_vect_ref to not plot the reference arrow (and no legend will also be written).
#Set streamplot when vector data is given, to plot streamlines rather than arrows.
#hatch_list - set as a list of arrays with values greater than 1 at grid points where hatching should be applied on each subplot (e.g. for marking regions of statistical significance). hatch_type controls the pattern used for hatching (default is stippling).
#shade_list - set as a list of arrays with numerical values where shading (a semi-transparent filled black region) should be plotted, and NaNs where it should not be plotted.
#cmap_name sets the colour scheme to use - default is 'rb_zero_centred', for colour scale with red for positive values and blue for negative values.
#Set plot_type to 'latlon' to make the plot a Basemap object, with plotted outlines of continents - latitude is assumed to be the first dimension of the arrays in data_list. Setting this to 'latlon_stereo' will make a stereographic plot - also set stereo_lat_range as a 2-element list or list of 2-element lists indicating the ranges of latitudes to plot.
#Set wrap_x_list as True when making a lat-lon stereographic plot to wrap the longitude axis around on itself, to remove the whitespace between the start and end points of the data. To do this for some subplots and not others, set it as a list of True and None/False appropriately.
#ncolumns is the number of columns to plot (default 1). Alternatively, the number of rows can be set with nrows.
#title_list is a single title or a list of titles corresponding to each array in data_list
#xlabel and ylabel are the x- and y-axis titles.
#xticks_list and yticks_list are lists of x- and y-tick labels are either each an array containing the ticks to use for all of subplots, or a list containing arrays with the levels for each subplot.
#xticks_labels_list and yticks_labels_list are lists of labels to use corresponding to the axis ticks.
#shape_pts_list is either a list containing 2-element lists/arrays giving the x- and y-coords of points to use to draw a shape, or a list of such lists containing different sets of points for each subplot. shape_thick and shape_col control the thickness and colour of the lines used.
#figsize is a 2-element tuple, setting the width and height of the figure in inches.
#titlesize and labelsize are font sizes of title and axis/colourbar labels respectively.
#paper_plot causes the text size to be increased, to make the plot roughly suitable for use in a paper.
#poster_plot does the same as paper_plot, for now.
#map_col is the colour of continent outlines to include on lat-lon plots.

#TO DO: 
# enable plotting filled and line contour plots using data on different axes.
# when plotting streamfunction, calculate the scalar stream function and contour it rather than use the matplotlib streamplot 
# allow option for logarithmic axes e.g. for plotting against pressure.
# allow plotting of vectors with x- and y-components with very different magnitudes e.g. have option to apply scaling to components so they appear a similar size on the plots.
# add option to set the centred longitude in lat-lon plots with lons wrapping around the globe
# work out how to get the colour bar intensities to match for positive and negative values without having to make the largest negative value equal the largest positive value, which reduces the effective number of contour levels.
# When plotting vectors, stop data being interpolated across regions where it is not wanted.
# fix plotting of vectors when plotting a longitude range that crosses the dateline with stereographic projection - currently the code converts the longitude axis to range from -180 to 180 and then interpolates across the huge gap where there is no data. I tried only shifting the vector longitude axis and doing the transformation, but if the longitude axis ranges between 0-360 then the calculation fails for longitudes above 180 in the map projection. One option is to take data for all longitudes, do the vector transformation to map coords and then select only the longitudes wanted. Alternatively, it may be better to use cartopy instead of Basemap.
def plot_2d(xvals_list=None, yvals_list=None, 
            xvals_contour_list=None, yvals_contour_list=None,
            xvals_filled_list=None, yvals_filled_list=None,
            xvals_vect_list=None, yvals_vect_list=None,
            xvals_hatch_list=None, yvals_hatch_list=None,
            xvals_shade_list=None, yvals_shade_list=None,
            contour_list=None, levels_contour_list=None, nlevels_contour=None, clip_levels_contour=None, contour_col=None, contour_linewidths=None, clabel_format=None,
            filled_list=None, levels_filled_list=None, nlevels_filled=None, clip_levels_filled=None, unit_filled_list=None,
            vect_list=None, vect_scale_list=None, unit_vect_list=None, no_vect_legend=None, no_vect_ref=None, streamplot=None,
            hatch_list=None, hatch_type='.', shade_list=None, 
            cmap_name='rb_zero_centred', plot_type=None, 
            stereo_lat_range_list=None, wrap_x_list=None,
            title_list=None, ncolumns=1, nrows=None, xlabel=None, ylabel=None,
            xticks_list=None, yticks_list=None,
            xticks_labels_list=None, yticks_labels_list=None,
            shape_pts_list=None, shape_thick=1, shape_col='k', 
            shape_pts_list2=None, shape_thick2=1, shape_col2='k', 
            figsize=None, titlesize=None, labelsize=None,
            paper_plot=None, poster_plot=None,
            map_col='gray'):

    #from mpl_toolkits.basemap import shiftgrid  #may be useful for centering plots on a specified longitude    
    import sys
    from plotting_gen import plot_2d_make_lists
    from python_gen import round_to_n, wrap_around
    
    #Convert data arrays to lists if necessary and get no. of plots
#    if contour_list and type(contour_list)!=list:  contour_list=[contour_list]  #this code doesn't work when the arguments are arrays
#    if filled_list and type(filled_list)!=list:  filled_list=[filled_list]
#    if vect_list and type(vect_list)!=list:  vect_list=[vect_list]
#    if hatch_list and type(hatch_list)!=list:  hatch_list=[hatch_list]
#    if shade_list and type(shade_list)!=list:  shade_list=[shade_list]
    if type(contour_list) not in [list,type(None)]: contour_list=[contour_list]
    if type(filled_list) not in [list,type(None)]: filled_list=[filled_list]
    if type(vect_list) not in [list,type(None)]: vect_list=[vect_list]
    if type(hatch_list) not in [list,type(None)]: hatch_list=[hatch_list]
    if type(shade_list) not in [list,type(None)]: shade_list=[shade_list]

    if contour_list: nplots=len(contour_list)
    elif filled_list: nplots=len(filled_list)
    elif vect_list:  nplots=len(vect_list)
    else:
        print 'plot_2d: At least one of contour_list, filled_list and vect_list must be set.'
        sys.exit()
    
    #Check that lists of data arrays have the same length if more than 1 is present
    if contour_list and filled_list:
        if len(contour_list) != len(filled_list):
            print 'len(contour_list)=',len(contour_list), ' != len(filled_list)=',len(filled_list)
            sys.exit()
        elif vect_list and len(contour_list) != len(vect_list):
            print 'len(contour_list)=',len(contour_list), ' != len(vect_list)=',len(vect_list)
            sys.exit()
        elif hatch_list and len(contour_list) != len(hatch_list):
            print 'len(contour_list)=',len(contour_list), ' != len(hatch_list)=',len(hatch_list)
            sys.exit()
        elif shade_list and len(contour_list) != len(shade_list):
            print 'len(contour_list)=',len(contour_list), ' != len(shade_list)=',len(shade_list)
            sys.exit()
    if filled_list and vect_list:
        if len(filled_list) != len(vect_list):
            print 'len(filled_list)=',len(filled_list), ' != len(vect_list)=',len(vect_list)
            sys.exit()
        elif hatch_list and len(filled_list) != len(hatch_list):
            print 'len(filled_list)=',len(filled_list), ' != len(hatch_list)=',len(hatch_list)
            sys.exit()
        elif shade_list and len(filled_list) != len(shade_list):
            print 'len(filled_list)=',len(filled_list), ' != len(shade_list)=',len(shade_list)
            sys.exit()
    if vect_list and hatch_list:
        if len(vect_list) != len(hatch_list):
            print 'len(vect_list)=',len(vect_list), ' != len(hatch_list)=',len(hatch_list)
            sys.exit()
        elif len(vect_list) != len(shade_list):
            print 'len(vect_list)=',len(vect_list), ' != len(shade_list)=',len(shade_list)
            sys.exit()
    
    #Get lists of other quantities as appropriate
    if xvals_list is not None and yvals_list is not None:
        xvals_list=plot_2d_make_lists(xvals_list,nplots)
        yvals_list=plot_2d_make_lists(yvals_list,nplots)
        
        #Assign same axes to all plots if xvals_list and yvals_list are set
        xvals_contour_list=xvals_list
        xvals_filled_list=xvals_list
        xvals_vect_list=xvals_list
        xvals_hatch_list=xvals_list
        xvals_shade_list=xvals_list
        yvals_contour_list=yvals_list
        yvals_filled_list=yvals_list
        yvals_vect_list=yvals_list
        yvals_hatch_list=yvals_list
        yvals_shade_list=yvals_list
    elif xvals_list or yvals_list:
        print 'Cannot set xvals_list without setting yvals_list or vice versa'
        sys.exit()
    else:
        if xvals_contour_list is not None and yvals_contour_list is not None:
            xvals_contour_list=plot_2d_make_lists(xvals_contour_list,nplots)
            yvals_contour_list=plot_2d_make_lists(yvals_contour_list,nplots)
        elif xvals_contour_list or yvals_contour_list:
            print 'Cannot set xvals_contour_list without setting yvals_contour_list or vice versa'
            sys.exit()
    
        if xvals_filled_list is not None and yvals_filled_list is not None:
            xvals_filled_list=plot_2d_make_lists(xvals_filled_list,nplots)
            yvals_filled_list=plot_2d_make_lists(yvals_filled_list,nplots)
        elif xvals_filled_list or yvals_filled_list:
            print 'Cannot set xvals_filled_list without setting yvals_filled_list or vice versa'
            sys.exit()
    
        if xvals_vect_list is not None and yvals_vect_list is not None:
            xvals_vect_list=plot_2d_make_lists(xvals_vect_list,nplots)
            yvals_vect_list=plot_2d_make_lists(yvals_vect_list,nplots)
        elif xvals_vect_list or yvals_vect_list:
            print 'Cannot set xvals_filled_list without setting yvals_filled_list or vice versa'
            sys.exit()
    
        if xvals_hatch_list is not None and yvals_hatch_list is not None:
            xvals_hatch_list=plot_2d_make_lists(xvals_hatch_list,nplots)
            yvals_hatch_list=plot_2d_make_lists(yvals_hatch_list,nplots)
        elif xvals_hatch_list or yvals_hatch_list:
            print 'Cannot set xvals_hatch_list without setting yvals_hatch_list or vice versa'
            sys.exit()
    
        if xvals_shade_list is not None and yvals_shade_list is not None:
            xvals_shade_list=plot_2d_make_lists(xvals_shade_list,nplots)
            yvals_shade_list=plot_2d_make_lists(yvals_shade_list,nplots)
        elif xvals_shade_list or yvals_shade_list:
            print 'Cannot set xvals_shade_list without setting yvals_shade_list or vice versa'
            sys.exit()
    
    if type(levels_contour_list)!=type(None) and levels_contour_list!='equal':
        if type(levels_contour_list)==list and type(levels_contour_list[0]) not in [list,np.ndarray,type(None)]:
            levels_contour_list=[levels_contour_list]
        levels_contour_list=plot_2d_make_lists(levels_contour_list,nplots)
    
    if type(levels_filled_list)!=type(None) and levels_filled_list!='equal':
        if type(levels_filled_list)==list and type(levels_filled_list[0]) not in [list,np.ndarray,type(None)]:
            levels_filled_list=[levels_filled_list]
        levels_filled_list=plot_2d_make_lists(levels_filled_list,nplots)
    
    if unit_filled_list:
        unit_filled_list=plot_2d_make_lists(unit_filled_list,nplots)
    elif filled_list:
        unit_filled_list=[None]*nplots  #create a list so that scaling factors can be added as the units if necessary 
    
    if vect_scale_list and vect_scale_list!='equal':
        vect_scale_list=plot_2d_make_lists(vect_scale_list,nplots)
    
    if unit_vect_list:    
        if type(unit_vect_list)!=list:  
            unit_vect_list=[unit_vect_list]
        assert len(unit_vect_list) in [1,2,nplots], "No. of sets of units for filled plot should match no. of plots, len(unit_vect_list)="+str(len(unit_vect_list))+" nplots="+str(nplots)
        if len(unit_vect_list)==1:
            unit_vect_list=unit_vect_list*nplots #use the same unit for both components of the vector
        elif len(unit_vect_list)==2 and type(unit_vect_list[0])==str: 
            unit_vect_list=[unit_vect_list]*nplots
    
    if title_list:
        if type(title_list)!=list:  
            title_list=[title_list]
        assert len(title_list)==nplots, "No. of titles should match no. of plots, len(title_list)="+str(len(title_list))+" nplots="+str(nplots)
    
    if xticks_list:
        if type(xticks_list)==list and type(xticks_list[0]) not in [list,np.ndarray,type(None)]:
            xticks_list=[xticks_list]
        xticks_list=plot_2d_make_lists(xticks_list,nplots)

    if yticks_list:
        if type(yticks_list)==list and type(yticks_list[0]) not in [list,np.ndarray,type(None)]:
            yticks_list=[yticks_list]
        yticks_list=plot_2d_make_lists(yticks_list,nplots)
    
    if xticks_labels_list:
        if type(xticks_labels_list)==list and type(xticks_labels_list[0]) not in [list,np.ndarray,type(None)]:
            xticks_labels_list=[xticks_labels_list]
        xticks_labels_list=plot_2d_make_lists(xticks_labels_list,nplots)

    if yticks_labels_list:
        if type(yticks_labels_list)==list and type(yticks_labels_list[0]) not in [list,np.ndarray,type(None)]:
            yticks_labels_list=[yticks_labels_list]
        yticks_labels_list=plot_2d_make_lists(yticks_labels_list,nplots)
    
    if stereo_lat_range_list:
        assert len(stereo_lat_range_list) in [2,nplots], "stereo_lat_range_list should be a 2-element list, giving the latitude range for all plots, or a list of nplots 2-element lists, giving separate ranges for each plot, len(stereo_lat_range_list)="+str(len(stereo_lat_range_list))+" nplots="+str(nplots)
        if len(stereo_lat_range_list)==2:
            stereo_lat_range_list=stereo_lat_range_list*nplots
    
    wrap_x_list=plot_2d_make_lists(wrap_x_list, nplots)
    
    if shape_pts_list:
        if type(shape_pts_list)==list and type(shape_pts_list[0])!=type(None) and type(shape_pts_list[0][0]) not in [list,np.ndarray,type(None)]:
                shape_pts_list=[shape_pts_list]
        shape_pts_list=plot_2d_make_lists(shape_pts_list,nplots)
    if shape_pts_list2:
        if type(shape_pts_list2)==list and type(shape_pts_list2[0])!=type(None) and type(shape_pts_list2[0][0]) not in [list,np.ndarray,type(None)]:
                shape_pts_list2=[shape_pts_list2]
        shape_pts_list2=plot_2d_make_lists(shape_pts_list2,nplots)
    
    #Set levels and colormap to be used for filled plots when it will be the same for all plots.
    if nlevels_contour is None: nlevels_contour=9
    if nlevels_filled is None: nlevels_filled=9
    
    if filled_list and levels_filled_list=='equal':
        #Set levels to be the same for each subplot. Limit maximum/minimum level value to 4 standard deviations from the mean.
        data_max=np.max([np.max(abs(data)) for data in filled_list])
        data_min=np.min([np.min(data) for data in filled_list])
        if clip_levels_filled:
            data_max=np.min([data_max, np.max([abs(np.mean(data))+4*np.std(data) for data in filled_list]) ])
            data_min=np.max([data_min, np.min([-abs(np.mean(data))-4*np.std(data) for data in filled_list]) ])
           
        if data_min<0: data_min=-data_max  #to get colour bar with positive and negative shading of similar intensity for positive and negative values of similar magnitude
        levels_filled=get_levels([data_min,data_max],nlevels_filled)
        data_filled_plt_min=np.min(levels_filled)
        data_filled_plt_max=np.max(levels_filled)
        exec("cmap=get_cmap_"+cmap_name+"(data_filled_plt_min,data_filled_plt_max)")  #getting colour map
        
        levels_filled_list=[levels_filled]*nplots
    
    #Setting levels for line contour plots when levels_contour=='equal' - only when there is more than one plot (else let matplotlib choose the levels) or the data used to make filled and line contour plots is the same, in which case the levels are made to match those of levels_filled.
    if contour_list and levels_contour_list in [None,'equal']:
        if filled_list and np.all([np.array_equal(contour_list[ind],filled_list[ind]) for ind in range(len(contour_list))]):
            levels_contour=levels_filled
        elif levels_contour_list=='equal' and len(contour_list)>1:  
            data_max=np.max([np.max(abs(data)) for data in contour_list])
            data_min=np.min([np.min(data) for data in contour_list])
            if clip_levels_contour:
                data_max=np.min([data_max, np.max([abs(np.mean(data))+4*np.std(data) for data in contour_list]) ])
                data_min=np.max([data_min, np.min([-abs(np.mean(data))-4*np.std(data) for data in contour_list]) ])
            
            levels_contour=get_levels([data_min,data_max],nlevels_contour)
        
        else:
            levels_contour=None
        
        levels_contour_list=[levels_contour]*nplots
    
    #Prepare for plotting vectors
    if vect_list:
        vect_list_plot=[]
        
        if vect_scale_list and vect_scale_list=='equal':
            scale=[30*1.5*np.std([np.sqrt(vect_arr_pair[0]**2+vect_arr_pair[1]**2) for vect_arr_pair in vect_list])][0]

#        #For latlon plots, shift longitudes to be in the range -180<=x<180, as required by the transform_vector function (note 180 does not seem to be an acceptable longitude)
#        #This incorrectly results in data being interpolated between the eastmost and westmost points when xvals straddles the dateline - deal with this when plotting the vectors below instead.
#        if plot_type in ['latlon','latlon_stereo'] and np.max(xvals)>=180:
#            if contour_list: contour_list=[ np.concatenate((arr[:,xvals>=180],arr[:,xvals<180]), axis=1) for arr in contour_list ]
#            if filled_list: filled_list=[ np.concatenate((arr[:,xvals>=180],arr[:,xvals<180]), axis=1) for arr in filled_list ]
#            if hatch_list: hatch_list=[ np.concatenate((arr[:,xvals>=180],arr[:,xvals<180]), axis=1) for arr in hatch_list ]
#            vect_list=[ [np.concatenate((vect[0][:,xvals>=180],vect[0][:,xvals<180]), axis=1), np.concatenate((vect[1][:,xvals>=180],vect[1][:,xvals<180]), axis=1)] for vect in vect_list ]
#            xvals=np.concatenate((xvals[xvals>=180]-360,xvals[xvals<180]))

#            #Commands to do the equivalent manipulation using shiftgrid - might be more useful if I want to have an option to centre plots on a particular longitude
#            if contour_list: contour_list=[ shiftgrid(180,arr,xvals,start=False)[0] for arr in contour_list ]
#            if filled_list: filled_list=[ shiftgrid(180,arr,xvals,start=False)[0] for arr in filled_list ]
#            if hatch_list: hatch_list=[ shiftgrid(180,arr,xvals,start=False)[0] for arr in hatch_list ]
#            vect_list=[ [shiftgrid(180,vect[0],xvals,start=False)[0], shiftgrid(180,vect[1],xvals,start=False)[0]] for vect in vect_list ]
#            xvals=shiftgrid(180,np.meshgrid(xvals,xvals)[0],xvals,start=False)[1]  #use meshgrid here to make fake data array

    #Set up axes
    if nrows:
        ncolumns=(nplots-1)/nrows+1
    elif ncolumns:
        nrows=(nplots-1)/ncolumns+1
    else:
        print 'One of ncolumns and nrows must be set'
        sys.exit()
    
    if not figsize:
        figsize=(6*ncolumns,4*nrows)  #good figure size when making lat-lon plots I think
    assert type(figsize)==tuple and len(figsize)==2, "figsize should be a 2-element tuple"

    fig, axarr = plt.subplots(nrows, ncolumns, figsize=figsize)
    if nrows==1 and ncolumns==1:  axarr=np.array(axarr)[np.newaxis,np.newaxis]  #for single plots, convert axes into array of axes of size (1,1)
    elif nrows==1:  axarr=axarr[np.newaxis,:]  #for when nrows=1 but ncolumns>1
    elif ncolumns==1:  axarr=axarr[:,np.newaxis]



    #Now loop over the subplots and plot the data
    for ind in range(nplots):
        basemap_set=None

        plt.sca(axarr[ind/ncolumns, ind % ncolumns])  #selects a particular subplot on axes for climatology plots (moving along rows)

        xmin_list=[]  #lists for collecting min and max coord values for creating basemap and for working out where a reference arrow will be placed on vector plots.
        xmax_list=[]
        ymin_list=[]
        ymax_list=[]
        if xvals_contour_list is not None and yvals_contour_list is not None:
            if xvals_contour_list[ind] is not None and yvals_contour_list[ind] is not None:
                xvals_contour=xvals_contour_list[ind]
                yvals_contour=yvals_contour_list[ind]
                xmin_list.append(np.min(xvals_contour))
                xmax_list.append(np.max(xvals_contour))
                ymin_list.append(np.min(yvals_contour))
                ymax_list.append(np.max(yvals_contour))
        if xvals_filled_list is not None and yvals_filled_list is not None:
            if xvals_filled_list[ind] is not None and yvals_filled_list[ind] is not None:
                xvals_filled=xvals_filled_list[ind]
                yvals_filled=yvals_filled_list[ind]    
                xmin_list.append(np.min(xvals_filled))
                xmax_list.append(np.max(xvals_filled))
                ymin_list.append(np.min(yvals_filled))
                ymax_list.append(np.max(yvals_filled))
        if xvals_vect_list is not None and yvals_vect_list is not None:
            if xvals_vect_list[ind] is not None and yvals_vect_list[ind] is not None:
                xvals_vect=xvals_vect_list[ind]
                yvals_vect=yvals_vect_list[ind]       
                xmin_list.append(np.min(xvals_vect))
                xmax_list.append(np.max(xvals_vect))
                ymin_list.append(np.min(yvals_vect))
                ymax_list.append(np.max(yvals_vect))
        if xvals_hatch_list is not None and yvals_hatch_list is not None:
            if xvals_hatch_list[ind] is not None and yvals_hatch_list[ind] is not None:
                xvals_hatch=xvals_hatch_list[ind]
                yvals_hatch=yvals_hatch_list[ind]       
                xmin_list.append(np.min(xvals_hatch))
                xmax_list.append(np.max(xvals_hatch))
                ymin_list.append(np.min(yvals_hatch))
                ymax_list.append(np.max(yvals_hatch))
        if xvals_shade_list is not None and yvals_shade_list is not None:
            if xvals_shade_list[ind] is not None and yvals_shade_list[ind] is not None:
                xvals_shade=xvals_shade_list[ind]
                yvals_shade=yvals_shade_list[ind]       
                xmin_list.append(np.min(xvals_shade))
                xmax_list.append(np.max(xvals_shade))
                ymin_list.append(np.min(yvals_shade))
                ymax_list.append(np.max(yvals_shade))
    
        if len(xmin_list)>0:  #Only proceed with plotting if data and axes have been specified for this particular subplot.
            xmin=np.min(xmin_list)
            xmax=np.max(xmax_list)
            ymin=np.min(ymin_list)
            ymax=np.max(ymax_list)
                
            if plot_type and plot_type in ['latlon','latlon_stereo']:
                if plot_type=='latlon_stereo':
                    if wrap_x_list[ind]:
                        if contour_list and contour_list[ind] is not None:  
                            contour_list[ind]=wrap_around(contour_list[ind],axis=1)  #put data for first longitude at both ends of longitude axis in data array
                            xvals_contour=wrap_around(xvals_contour)
                        if filled_list and filled_list[ind] is not None:  
                            
                            filled_list[ind]=wrap_around(filled_list[ind],axis=1)
                            xvals_filled=wrap_around(xvals_filled)
                    stereo=1
                    stereo_lat_range=stereo_lat_range_list[ind]
                    if (stereo_lat_range[0]<=0 and stereo_lat_range[1]>=0) or (stereo_lat_range[0]>=0 and stereo_lat_range[1]<=0):
                        print 'stereo_lat_range should have values that are both in the same hemisphere, excluding zero: stereo_lat_range=',stereo_lat_range
                        import sys
                        sys.exit()
                    
                else:
                    stereo=None
                    stereo_lat_range=None
                
                #Create Basemap object
                x_dummy, y_dummy, m = set_basemap([xmin,xmax],[ymin,ymax],stereo=stereo,lat_range=stereo_lat_range,map_col=map_col)
                
                #converting to map projection coords
                if xvals_contour_list is not None and yvals_contour_list is not None:
                    if xvals_contour_list[ind] is not None and yvals_contour_list[ind] is not None:
                        xvals_contour_plot, yvals_contour_plot = np.meshgrid(xvals_contour, yvals_contour)
                        xvals_contour_plot, yvals_contour_plot = m(xvals_contour_plot, yvals_contour_plot)
                if xvals_filled_list is not None and yvals_filled_list is not None:
                    if xvals_filled_list[ind] is not None and yvals_filled_list[ind] is not None:
                        xvals_filled_plot, yvals_filled_plot = np.meshgrid(xvals_filled, yvals_filled)
                        xvals_filled_plot, yvals_filled_plot = m(xvals_filled_plot, yvals_filled_plot)
                if xvals_vect_list is not None and yvals_vect_list is not None:
                    if xvals_vect_list[ind] is not None and yvals_vect_list[ind] is not None:
                        xvals_vect_plot, yvals_vect_plot = np.meshgrid(xvals_vect, yvals_vect)
                        xvals_vect_plot, yvals_vect_plot = m(xvals_vect_plot, yvals_vect_plot)
                
                basemap_set=1
            else:
                if xvals_contour_list is not None and yvals_contour_list is not None:
                    if xvals_contour_list[ind] is not None and yvals_contour_list[ind] is not None:
                        xvals_contour_plot, yval_contours_plot = np.meshgrid(xvals_contour, yvals_contour)
                if xvals_filled_list is not None and yvals_filled_list is not None:
                    if xvals_filled_list[ind] is not None and yvals_filled_list[ind] is not None:
                        xvals_filled_plot, yvals_filled_plot = np.meshgrid(xvals_filled, yvals_filled)
                if xvals_vect_list is not None and yvals_vect_list is not None:
                    if xvals_vect_list[ind] is not None and yvals_vect_list[ind] is not None:
                        xvals_vect_plot, yvals_vect_plot = np.meshgrid(xvals_vect, yvals_vect)
    
            #If vectors are to be added, set up parameters for the reference arrow 
            if vect_list and vect_list[ind] is not None and (not no_vect_ref) and (not streamplot):
                if plot_type in [None,'latlon','latlon_stereo']:
                    ref_arrow_loc=[0.05,0.92] #(x,y) coords of reference arrow as fractions of the axis values
                    arrow_box_bot=ymin+(ymax-ymin)*(ref_arrow_loc[1]-0.05)
                    #arrow_box_right=xmin+(xmax-xmin)*(ref_arrow_loc[0]+0.5)
                    if no_vect_legend:
                        arrow_box_right=xmin+(xmax-xmin)*(ref_arrow_loc[0]+0.05)
                    else:
                        arrow_box_right=xmax  #setting the box for the arrow legend to stretch across the whole figure width
    
                    #For latlon plots, set data in region where the reference arrow and its legend will go to NaN - I can't see how to select the right region for stereo plots.
                    if plot_type=='latlon':
                        if contour_list and contour_list[ind] is not None:
                            inds=np.meshgrid(np.where(yvals_contour>arrow_box_bot)[0],np.where(xvals_contour<arrow_box_right)[0])
                            contour_list[ind][inds]=np.nan
                        if filled_list and filled_list[ind] is not None:
                            inds=np.meshgrid(np.where(yvals_filled>arrow_box_bot)[0],np.where(xvals_filled<arrow_box_right)[0])
                            filled_list[ind][inds]=np.nan
                        if hatch_list and hatch_list[ind] is not None:
                            inds=np.meshgrid(np.where(yvals_hatch>arrow_box_bot)[0],np.where(xvals_hatch<arrow_box_right)[0])
                            hatch_list[ind][inds]=np.nan
                        if shade_list and shade_list[ind] is not None:
                            inds=np.meshgrid(np.where(yvals_shade>arrow_box_bot)[0],np.where(xvals_shade<arrow_box_right)[0])
                            shade_list[ind][inds]=np.nan
                        
                    """An attempt to get this to work with stereo plots by finding indices to set to NaN in map coordinates - I don't know why it doesn't.
                    x_plot_min_list=[]  #lists for collecting min and max coord values for creating basemap and for working out where a reference arrow will be placed on vector plots.
                    x_plot_max_list=[]
                    y_plot_min_list=[]
                    y_plot_max_list=[]
                    if xvals_contour_list is not None and yvals_contour_list is not None:
                        if xvals_contour_list[ind] is not None and yvals_contour_list[ind] is not None:
                            x_plot_min_list.append(np.min(xvals_contour_plot))
                            x_plot_max_list.append(np.max(xvals_contour_plot[xvals_contour_plot<1.e30]))  #1.e30 seems to be the value representing unplotted data or something 
                            y_plot_min_list.append(np.min(yvals_contour_plot))
                            y_plot_max_list.append(np.max(yvals_contour_plot[yvals_contour_plot<1.e30]))
                    if xvals_filled_list is not None and yvals_filled_list is not None:
                        if xvals_filled_list[ind] is not None and yvals_filled_list[ind] is not None:
                            x_plot_min_list.append(np.min(xvals_filled_plot))
                            x_plot_max_list.append(np.max(xvals_filled_plot[xvals_filled_plot<1.e30]))
                            y_plot_min_list.append(np.min(yvals_filled_plot))
                            y_plot_max_list.append(np.max(yvals_filled_plot[yvals_filled_plot<1.e30]))
                    if xvals_vect_list is not None and yvals_vect_list is not None:
                        if xvals_vect_list[ind] is not None and yvals_vect_list[ind] is not None:
                            x_plot_min_list.append(np.min(xvals_vect_plot))
                            x_plot_max_list.append(np.max(xvals_vect_plot[xvals_vect_plot<1.e30]))
                            y_plot_min_list.append(np.min(yvals_vect_plot))
                            y_plot_max_list.append(np.max(yvals_vect_plot[yvals_vect_plot<1.e30]))
                
                    x_plot_min=np.min(x_plot_min_list)
                    x_plot_max=np.min(x_plot_max_list)
                    y_plot_min=np.min(y_plot_min_list)
                    y_plot_max=np.min(y_plot_max_list)
                
                    ref_arrow_loc=[0.05,0.92] #(x,y) coords of reference arrow as fractions of the axis values
                    arrow_box_bot=y_plot_min+(y_plot_max-y_plot_min)*(ref_arrow_loc[1]-0.05)
                    arrow_box_right=xmin+(x_plot_max-x_plot_min)*(ref_arrow_loc[0]+0.5)
    
                    if contour_list and contour_list[ind] is not None:
                        inds=np.meshgrid(np.where(yvals_contour_plot[:,0]>arrow_box_bot)[0],np.where(xvals_contour_plot[0,:]<arrow_box_right)[0])
                        contour_list[ind][inds]=np.nan
                    if filled_list and filled_list[ind] is not None:
                        inds=np.meshgrid(np.where(yvals_filled_plot[:,0]>arrow_box_bot)[0],np.where(xvals_filled_plot[0,:]<arrow_box_right)[0])
                        filled_list[ind][inds]=np.nan
                    if hatch_list and hatch_list[ind] is not None:
                        inds=np.meshgrid(np.where(yvals_hatch_plot[:,0]>arrow_box_bot)[0],np.where(xvals_hatch_plot[0,:]<arrow_box_right)[0])
                        hatch_list[ind][inds]=np.nan"""                           
    
            #Make line contour plots
            if contour_list and contour_list[ind] is not None:
                #if levels_contour_list!='equal': #It should always be a list now
                if levels_contour_list and levels_contour_list[ind] is not None:
                    levels_contour=levels_contour_list[ind]
                else:
                    levels_contour=None
    
                #Make contours thick and light grey if both vectors and filled contours are also being plotted, and black otherwise.
                if not contour_col:
                    if filled_list and filled_list[ind] is not None and vect_list and vect_list[ind] is not None:
                        col='0.9'
                    else:
                        col='k'
                else:
                    col=contour_col
                
                if not contour_linewidths:
                    if filled_list and filled_list[ind] is not None and vect_list and vect_list[ind] is not None:
                        linewidths=2
                    else:
                        linewidths=1
                else:
                    linewidths=contour_linewidths
                
                if basemap_set:
                    p=m.contour(xvals_contour_plot,yvals_contour_plot,contour_list[ind], colors=col, linewidths=linewidths, levels=levels_contour)
                else:
                    p=plt.contour(xvals_contour_plot,yvals_contour_plot,contour_list[ind], colors=col, linewidths=linewidths, levels=levels_contour)
                
                if not clabel_format:
                    if levels_contour is None:
                        clabel_format='%1.1f'
                    else:                
                        if np.max(np.abs([np.log10(abs(lev)) for lev in levels_contour if abs(lev)>0]))<=3:       #identify if log10(lev) goes above 3 or below -3   
                            clabel_format='%1.1f'
                        else:
                            clabel_format='%1.1e'
                plt.clabel(p, inline=1, fontsize=9, fmt=clabel_format)
    
            #Make filled contour plots
            if filled_list and filled_list[ind] is not None:
                #if levels_filled_list!='equal':  #It should always be a list now
                if levels_filled_list and levels_filled_list[ind] is not None:
                    levels_filled=levels_filled_list[ind]
                else:
                    data_max=np.nanmax(abs(filled_list[ind]))
                    data_min=np.nanmin(filled_list[ind])
                    if data_min<0: data_min=-data_max  #to get colour bar with positive and negative shading of similar intensity for positive and negative values of similar magnitude
                    levels_filled=get_levels([data_min,data_max],nlevels_filled)
                levels_filled=np.array(levels_filled)

                #If levels reach more than 1000, scale the data and levels to make the maximum of the data less than 10, and put the scaling into the units.
                if np.nanmax(np.abs(levels_filled))>=1000.:
                    expnt=int(np.floor(np.log10(np.nanmax(np.abs(levels_filled)))))
                elif np.nanmin(np.abs(levels_filled[np.nonzero(levels_filled)]))<=1/1000.:
                    expnt=int(np.floor(np.log10(np.nanmax(np.abs(levels_filled[np.nonzero(levels_filled)])))))
                else:
                    expnt=1
                    
                if np.abs(expnt)>=3:
                    scale=10**expnt
#                    if type(levels_filled)==list:
#                        #Scale the data and levels
#                        filled_list[ind]=filled_list[ind]/scale
#                        levels_filled=levels_filled/scale
#                    else:
#                        #If there is one set of levels for all plots, then when the levels are scaled, this code is not executed again - so scale all the data arrays here.
#                        filled_list=[arr/scale for arr in filled_list]
#                        levels_filled=levels_filled/scale
                    
                    filled_list[ind]=filled_list[ind]/scale
                    levels_filled=levels_filled/scale
                        
                    if unit_filled_list[ind] is not None:
                        unit_filled_list[ind]='10^'+str(expnt)+' '+str(unit_filled_list[ind])
                    else:
                        unit_filled_list[ind]='10^'+str(expnt)
                
                #Get the colour map
                data_filled_plt_min=np.nanmin(levels_filled)
                data_filled_plt_max=np.nanmax(levels_filled)
                exec("cmap=get_cmap_"+cmap_name+"(data_filled_plt_min,data_filled_plt_max)")  #getting colour map
                
                if basemap_set: 
                    p=m.contourf(xvals_filled_plot,yvals_filled_plot,filled_list[ind], cmap=cmap, vmin=data_filled_plt_min, vmax=data_filled_plt_max, levels=levels_filled)
                else: 
                    p=plt.contourf(xvals_filled_plot,yvals_filled_plot,filled_list[ind], cmap=cmap, vmin=data_filled_plt_min, vmax=data_filled_plt_max, levels=levels_filled)
                cbar=plt.colorbar()
    
                #If levels are not evenly spaced, force labelling of each level in the colourbar
                #level_intervals=np.array(levels_filled[1:])-np.array(levels_filled[:-1])
                #if len(set(level_intervals))>1:  #didn't work because set() gives more than one value when level_intervals are only different by floating point error
                first_level_int=levels_filled[1]-levels_filled[0]
                if np.max([abs(levels_filled[lev_ind+1]-levels_filled[lev_ind]-first_level_int) for lev_ind in range(len(levels_filled)-1)]) > 1e-7*np.max(abs(np.array(levels_filled))):
                    cbar.set_ticks(levels_filled)
                
                #cbar.formatter.set_powerlimits((-3, 3))  #force colourbar to put exponent at the top if very large or small values are being plotted  #Not necessary now that I scale the data and levels above
                
                if labelsize:
                    cbar_fontsize=labelsize
                elif paper_plot:
                    cbar_fontsize='medium'
                elif poster_plot:
                    cbar_fontsize='large'
                else:
                    cbar_fontsize='medium'
                cbar.ax.tick_params(labelsize=cbar_fontsize)
                
                cbar.update_ticks()
    
                if unit_filled_list and unit_filled_list[ind] is not None:
                    cbar.set_label(unit_filled_list[ind], fontsize=cbar_fontsize)
            
            #Plot vectors
            if vect_list and vect_list[ind] is not None and (not streamplot):
                
                #Get vectors on reduced grid (i.e. select only a subsample of grid points at which to plot the vectors) - do for all subplots at once, so that the arrow scale can be set for all the plots. Currently assumes that all arrays of vector data are on axes described by xvals_list[0] and yvals_list[0].
                    
                n_arrows=15  #no. of arrows in x- and y-directions
                
                if basemap_set and plot_type != 'latlon':  #this code does not work when the plotted region crosses the dateline, due to the transform_vector() function not being able to handle this situation. For regular lat-lon plots, the code I have written under the else statement works fine, so just use that in such cases.
                    
                    #For latlon plots, shift longitudes to be in the range -180<=x<180, as required by the transform_vector function (note 180 does not seem to be an acceptable longitude)
                    if np.max(xvals_vect)>=180:
                        vect_shifted=[np.concatenate((vect_list[ind][0][:,xvals_vect>=180],vect_list[ind][0][:,xvals_vect<180]), axis=1), np.concatenate((vect_list[ind][1][:,xvals_vect>=180],vect_list[ind][1][:,xvals_vect<180]), axis=1)]
                        xvals_shifted=np.concatenate((xvals_vect[xvals_vect>=180]-360,xvals_vect[xvals_vect<180]))
                    else:
                        vect_shifted=vect_list[ind]
                        xvals_shifted=xvals_vect
                    #print xvals_shifted
                    
                    #Get indices that will put axes into ascending order, as required by the transform_vector function
                    inds_x=np.argsort(xvals_shifted)
                    inds_y=np.argsort(yvals_vect)
                    
                    vect_xy_inds=np.meshgrid(inds_y,inds_x)
    
                    #Get vectors transformed into the coordinate space
                    vect_plot_x,vect_plot_y,xvals_vect_trans,yvals_vect_trans = m.transform_vector(vect_shifted[0][vect_xy_inds].T,vect_shifted[1][vect_xy_inds].T,xvals_shifted[inds_x],yvals_vect[inds_y],n_arrows,n_arrows,returnxy=True,masked=True)
                    #I could remove vectors interpolated across regions where data has not been asked for here. This would involve identifying the boundaries of the requested region in the transformed space (e.g. using m.transform_vector() again and specifying the boundary values) and excluding points for which xvals_vect and yvals_vect lie outside this region.
                    
                    vect_list_plot=[vect_plot_x,vect_plot_y]
                        
                else:
                    
                    #Limit no. of arrows in each direction and get indices of grid points at which to plot arrows
                    if len(xvals_vect_plot[0,:])>=2*n_arrows:
                        vect_x_inds=np.arange(n_arrows)*len(xvals_vect_plot[0,:])/n_arrows
                    else:
                        vect_x_inds=range(len(xvals_vect_plot[0,:]))
                        
                    if len(yvals_vect_plot[:,0])>=2*n_arrows:
                        vect_y_inds=np.arange(n_arrows)*len(yvals_vect_plot[:,0])/n_arrows
                    else:
                        vect_y_inds=range(len(yvals_vect_plot[:,0]))
    
                    vect_xy_inds=np.meshgrid(vect_y_inds,vect_x_inds)
                    
                    xvals_vect_trans=xvals_vect_plot[vect_xy_inds]
                    yvals_vect_trans=yvals_vect_plot[vect_xy_inds]
                    vect_list_plot=[vect_list[ind][0][vect_xy_inds],vect_list[ind][1][vect_xy_inds]]
    
                
                #Set the arrow scale for each subplot
                if vect_scale_list!='equal':
                    if vect_scale_list and vect_scale_list[ind] is not None:
                        scale0=30*vect_scale_list[ind]  #multiply by 30 to get scale in units of data per plot width
                    else:
                        scale0=30*1.5*np.std(np.sqrt(vect_list_plot[0]**2+vect_list_plot[1]**2))
                    
                if paper_plot:  #set up some things for paper plots
                    scale=scale0*2./3  #reduce scale to get larger reference arrow
                    if ncolumns>=3 or nrows>=3:
                        width=0.0075
                else:
                    scale=scale0
                    width=None
                
                #For "standard" or stereographic projections, leave room in the top left corner to plot a reference arrow, and add the coords of this arrow onto those of the arrows to be plotted.
                if plot_type in [None,'latlon','latlon_stereo'] and (not no_vect_ref):
                    vect_xy_inds2=np.meshgrid(range(len(yvals_vect_trans[:,0])),range(len(xvals_vect_trans[0,:])))
                    vect_xy_inds2=[arr[np.where( (yvals_vect_trans[vect_xy_inds2]<arrow_box_bot) | (xvals_vect_trans[vect_xy_inds2]>arrow_box_right) )] for arr in vect_xy_inds2]
    
                    #Calculate the reference arrow parameters for each subplot and add them to the arrays describing the vectors to be plotted.
                    ref_arrow_mag=np.zeros(2)
                    ref_arrow_mag[0]=round_to_n(scale0/30., 2)  #use rounding so rounded values can be written on plot to indicate the reference arrow size below
                    ref_arrow_mag[1]=ref_arrow_mag[0]
    
                    xvals_vect_trans=np.append(xvals_vect_trans[vect_xy_inds2],xvals_vect_trans.min()+np.ptp(xvals_vect_trans)*ref_arrow_loc[0])
                    yvals_vect_trans=np.append(yvals_vect_trans[vect_xy_inds2],yvals_vect_trans.min()+np.ptp(yvals_vect_trans)*ref_arrow_loc[1])
                    vect_list_plot=[np.append(vect_list_plot[0][vect_xy_inds2],ref_arrow_mag[0]), np.append(vect_list_plot[1][vect_xy_inds2],ref_arrow_mag[1])]                        
                    ref_arrow_added=1
                else:
                    ref_arrow_added=None
                
                #Testing making vectors grey if line contours are also being plotted, and black otherwise - this didn't work that well.
                col='k'
    #            if contour_list and contour_list[ind] is not None:
    #                col='0.75'
    #            else:
    #                col='k'
    
                #Finished setting up - now plot the vectors/streamfunction
                if basemap_set:  m.quiver(xvals_vect_trans,yvals_vect_trans,vect_list_plot[0],vect_list_plot[1], pivot='middle', scale=scale, width=width, color=col, zorder=2) #"zorder=2" stops vectors being plotted behind contours
                else:  plt.quiver(xvals_vect_trans,yvals_vect_trans,vect_list_plot[0],vect_list_plot[1], pivot='middle', scale=scale, width=width, color=col, zorder=2)
                
                #Output size of reference arrow on plot if applicable
                #The quiverkey() function may also be useful here.
                if ref_arrow_added and (not no_vect_legend) and (not no_vect_ref):
                    #text_xpos=xvals_vect_trans.min()+np.ptp(xvals_vect_trans)*(ref_arrow_loc[0]+0.05)
                    #text_ypos=yvals_vect_trans.min()+np.ptp(yvals_vect_trans)*ref_arrow_loc[1]
                    text_xpos=xmin+(xmax-xmin)*(ref_arrow_loc[0]+0.05)
                    text_ypos=ymin+(ymax-ymin)*ref_arrow_loc[1]
                    if unit_vect_list and unit_vect_list[ind] is not None:
                        unit_vect=unit_vect_list[ind]
                        if type(unit_vect)!=list:
                            ref_arrow_text='({:},{:})'.format(ref_arrow_mag[0],ref_arrow_mag[1])+' '+str(unit_vect)
                        else:
                            if unit_vect[0]==unit_vect[1]:
                                ref_arrow_text='({:},{:})'.format(ref_arrow_mag[0],ref_arrow_mag[1])+' '+str(unit_vect[0])
                            else:
                                ref_arrow_text='({:} '+str(unit_vect[0])+',{:} '+str(unit_vect[1])+')'.format(ref_arrow_mag[0],ref_arrow_mag[1])
                    else:
                        ref_arrow_text='({:},{:})'.format(ref_arrow_mag[0],ref_arrow_mag[1])
    
                    plt.text(text_xpos, text_ypos, ref_arrow_text, verticalalignment='center')
    
            elif vect_list and streamplot:
                if basemap_set:  m.streamplot(xvals_vect_plot[0,:],yvals_vect_plot[:,0],vect_list[ind][0],vect_list[ind][1])
                else: plt.streamplot(xvals_vect_plot[0,:],yvals_vect_plot[:,0],vect_list[ind][0],vect_list[ind][1])
            
    
            #Plot hatching/stippling
            if hatch_list and hatch_list[ind] is not None:
                
                if wrap_x_list[ind]:
                    hatch_list[ind]=wrap_around(hatch_list[ind],axis=1)
                
                plt.contourf(xvals_hatch,yvals_hatch,hatch_list[ind],[1.,np.inf],hatches=[hatch_type,None],colors='none')
            
            #Plotting shading as a partially transparent black area
            if shade_list and shade_list[ind] is not None:
                
                if wrap_x_list[ind]:
                    shade_list[ind]=wrap_around(shade_list[ind],axis=1)
                
                plt.contourf(xvals_shade,yvals_shade,shade_list[ind],colors='k',levels=[0.5,1.01],alpha=0.25,extend=None)
       
            if title_list and title_list[ind] is not None:
                if titlesize:
                    fontsize=titlesize
                else:
                    fontsize=16-2*(title_list[ind].count('\n')+1) #reduce fontsize for each newline in the title
                    if paper_plot:
                        fontsize=fontsize
                    elif poster_plot: 
                        fontsize=1.5*fontsize
                plt.title(title_list[ind], fontsize=fontsize, y=1.05)  #raise slightly above the normal position so it doesn't overlap with the color bar

            if (xticks_list and xticks_list[ind]) or (yticks_list and yticks_list[ind]) or xlabel or ylabel:
                if labelsize:
                    fontsize=labelsize
                    axarr[ind/ncolumns, ind % ncolumns].tick_params(labelsize=fontsize)
                elif paper_plot:
                    fontsize='medium'
                elif poster_plot:
                    fontsize='large'
                    axarr[ind/ncolumns, ind % ncolumns].tick_params(labelsize=fontsize)  #making tick labels bigger
                else:
                    fontsize='medium'

            if xticks_list and xticks_list[ind]:
                if xticks_labels_list and xticks_labels_list[ind]:
                    plt.xticks(xticks_list[ind], xticks_labels_list[ind], fontsize=fontsize)
                else:
                    plt.xticks(xticks_list[ind], fontsize=fontsize)
            if yticks_list and yticks_list[ind]:
                if yticks_labels_list and yticks_labels_list[ind]:
                    plt.yticks(yticks_list[ind], yticks_labels_list[ind], fontsize=fontsize)
                else:
                    plt.yticks(yticks_list[ind], fontsize=fontsize)
            
            if xlabel or ylabel:
                if xlabel:
                    plt.xlabel(xlabel, fontsize=fontsize)
                if ylabel:
                    plt.ylabel(ylabel, fontsize=fontsize)
            
            
            #Plotting shapes, if specified
            if shape_pts_list and shape_pts_list[ind] is not None:
                xpts=[lst[0] for lst in shape_pts_list[ind]]
                ypts=[lst[1] for lst in shape_pts_list[ind]]
                plt.plot(xpts,ypts,color=shape_col,linewidth=shape_thick)
            if shape_pts_list2 and shape_pts_list2[ind] is not None:
                xpts=[lst[0] for lst in shape_pts_list2[ind]]
                ypts=[lst[1] for lst in shape_pts_list2[ind]]
                plt.plot(xpts,ypts,color=shape_col2,linewidth=shape_thick2)

    #Finished looping over the subplots
        
    #Remove unneeded blank subplots
    if nplots<len(axarr.ravel()):
        for i in range(ncolumns - (nplots % ncolumns)):
            fig.delaxes(axarr[-1,ncolumns-(i+1)])
    
    plt.tight_layout()
    
#    #Fudges to make layout of figure look alright when there are many subplots
#    if plot_type!='latlon_stereo':
#        if nrows==4: fig.subplots_adjust(hspace=0.3)
#        elif nrows==5: 
#            fig.subplots_adjust(hspace=0.6)
#            fig.subplots_adjust(top=0.92)
#            
#        if ncolumns==3: 
#            if nrows==1:  fig.subplots_adjust(wspace=0.2)
#            else: fig.subplots_adjust(wspace=0.12)
#        elif ncolumns==4: fig.subplots_adjust(wspace=0.2)
    

    return fig,axarr


#Function for use in plot_2d() to make appropriate lists of input quantities
def plot_2d_make_lists(lst,nplots):
    if type(lst)!=list:  
        lst=[lst]
    assert len(lst)==1 or len(lst)==nplots, "Length of lists should match no. of plots, len(lst)="+str(len(lst))+" nplots="+str(nplots)
    if len(lst)==1: 
        lst=lst*nplots #repeat array, to be used for all subplots
    
    return lst


#Function to set axis ranges sensibly on plots
#ax is input axis object
#varx and vary arer the x- and y-values
def set_axis_ranges(ax,varx,vary):
    x_range=max(varx)-min(varx)
    ax.set_xlim(min(varx)-0.05*x_range,max(varx)+0.05*x_range)
    y_range=max(vary)-min(vary)
    ax.set_ylim(min(vary)-0.05*y_range,max(vary)+0.05*y_range)


#Function to set up basemap for plotting
#Arguments are the x-coords xvals; y-coords yvals; stereo, set to plot polar stereographic projection; lat_range, which sets the latitude range to set the plot edges when used with the stereo option; wrap_x, which causes the xvals array to be wrapped around when set (useful when making stereographic plots for getting rid of the whitespace where the ends of the longitude axis meet); and xlabel and ylabel, which specify the axis labels to use (only work when stereo is not set).
#map_col sets the colours of the continent outlines.
#Currently this overrides the default behaviour of fixing the aspect ratio to match that of the map projection for non-stereographic plots - change this by allowing the "fix_aspect" option to be set.
def set_basemap(xvals,yvals,stereo=None,lat_range=None,wrap_x=None,xlabel=None,ylabel=None,map_col='gray'):
    from mpl_toolkits.basemap import Basemap
    from python_gen import wrap_around

    if stereo: 
        lon_0=0.
        if np.min(lat_range) > 0:
            m = Basemap(projection='npstere', boundinglat=np.min(lat_range), lon_0=lon_0)
            parallels=np.arange(0,90,30)
        elif np.max(lat_range) < 0: 
            m = Basemap(projection='spstere', boundinglat=np.max(lat_range), lon_0=lon_0)
            parallels=np.arange(-60,1,30)  #gives [-60, -30, 0]
        m.drawmeridians(np.arange(0,360,30), labels=[True,False,False,True])  #labels = [left,right,top,bottom]
        m.drawparallels(parallels)
        
        #labelling the parallels
        ax=plt.gca()
        for parallel in parallels:  
            x,y=m(lon_0,parallel+0.02*(np.max(lat_range)-np.min(lat_range)))
            if parallel>0 and abs(y)<1e10:  #only plot label if the parallel is on the plot
                ax.text(x,y,str(parallel)+'N')
            elif parallel<0 and abs(y)<1e10:
                ax.text(x,y,str(-parallel)+'S')
                            
    else:
        m = Basemap(llcrnrlon=np.min(xvals),llcrnrlat=np.min(yvals),urcrnrlon=np.max(xvals),urcrnrlat=np.max(yvals), suppress_ticks=False, fix_aspect=False)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
    
    if wrap_x:
        xvals=wrap_around(xvals)  #wrap_around() is function in python_gen.py
    
    xvals, yvals = np.meshgrid(xvals, yvals)
    xvals, yvals = m(xvals, yvals)  #converting to map projection coords
    m.drawcoastlines(color=map_col)

    return xvals, yvals, m
    

