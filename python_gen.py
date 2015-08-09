#Script containing some useful python functions and subroutines

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import fnmatch
from scipy.integrate import trapz as int1D
from scipy.interpolate import interp1d
import set_physical_constants as constants
from Scientific.IO.NetCDF import NetCDFFile

#Python function to calculate geopotential height assuming hydrostatic balance, given pressure levels in Pa in array plevs, the temperature on each level in array T and the height z0 on level p0. 
#T can be multi-dimensional - in that case if the pressure axis is not axis 0 (the first axis), give the axis number. 
#z0 can be a single number, if it is the same along all non-pressure dimensions of T, or it can have the same dimensions as T except for a length-1 pressure dimension, with different values at each position along the other dimensions.

def calc_geo_hb(plevs,T,z0,p0,axis=0):

  logp=np.log(plevs)
  
  #Get index corresponding to specified z0
  z0_ind=np.where(plevs==p0)[0]
  if len(z0_ind)==0: 
    print "Can't find level p0=",p0
    print "plevs=",plevs
    sys.exit()
  #print z0_ind,plevs[z0_ind]
  
  #Get pressure axis as leading axis
  T=T.swapaxes(0,axis)
  if len(np.ravel(z0))>1: 
    z0=z0.swapaxes(0,axis)

  z=np.zeros(T.shape)
  
  if len(np.ravel(z0))>1:
    z[z0_ind,...]=z0
  else:
    z[z0_ind,...]=np.tile(z0, z[z0_ind,...].shape)
  
  R=constants.R
  g=constants.g
  for k in range(z0_ind+1,len(plevs)):
    #print k
    z[k,...]=z[z0_ind,...]-R/g*int1D(T[z0_ind:k+1,...],logp[z0_ind:k+1],axis=0)

  for k in range(0,z0_ind):
    #print k
    z[k,...]=z[z0_ind,...]+R/g*int1D(T[k:z0_ind+1,...],logp[k:z0_ind+1],axis=0)
  
  #Put axes back to their original positions
  z=z.swapaxes(0,axis)

  #Not sure these lines are necessary, but I put them here just in case there are situations when the input arguments could be modified in the calling script.
  T=T.swapaxes(0,axis)
  if len(np.ravel(z0))>1: 
    z0=z0.swapaxes(0,axis)

  return z


#Python function to convert Cartesian coords (x,y,z) to spherical coords - the inverse of lon_lat_to_cartesian(). z is taken to be along the Earth's rotation axis and increase northwards, and (1,0,0) corresponds to (0E, 0N) on a unit sphere. Returned lons are in the range -180 to 180 deg.
#rads specifies to return the result in radians - else it is returned in degrees.
def cartesian_to_lon_lat(x, y, z, rads=None):

    lon=np.arctan2(y,x)
    
#    if np.sqrt(x**2 + y**2)>0:    
#        lat=np.arctan(z/np.sqrt(x**2 + y**2))
#    elif z>0:  lat=np.pi/2.  #x=y=0 cases
#    elif z<0:   lat=-np.pi/2.

    lat=np.arctan2(z,np.sqrt(x**2 + y**2))

    R=np.sqrt(x**2+y**2+z**2)

    if not rads:
        lon = np.degrees(lon)
        lat = np.degrees(lat)

    return lon,lat,R


#Python function to mimic IDL's closest() and return the index of a numpy array where the array value is closest to "value".
def closest(array,value):
  ind=abs(array-value).argmin()
  return ind

#Python function to convert string representations of numbers that may contain fractions into floats, modified from one answer at http://stackoverflow.com/questions/1806278/convert-fraction-to-float
#e.g. convert_fracstr_to_flt('-1 2/3') gives -1.666666...
def convert_fracstr_to_flt(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
            
        if frac_str[0]=='-':
            return float(leading) - float(num) / float(denom)
        else:
            return float(leading) + float(num) / float(denom)


#Function to make a copy of a function, from http://stackoverflow.com/questions/6527633/how-can-i-make-a-deepcopy-of-a-function-in-python
def copy_func(f, name=None):
    import types
    return types.FunctionType(f.func_code, f.func_globals, name or f.func_name,f.func_defaults, f.func_closure)


#Class whose members are functions that can be added, such that (f+g)(x) = f(x) + g(x), or composed, such that (f*g)(x) = f(g(x)). Taken from http://stackoverflow.com/questions/4101244/how-to-add-functions .
class FunctionalFunction(object):
    from numbers import Number  #PW - to allow multiplication by ordinary numbers, following answer at http://stackoverflow.com/questions/4233628/override-mul-from-a-child-class-using-parent-implementation-leads-to-proble     
    
    def __init__(self, func):
            self.func = func

    def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    def __add__(self, other):
            def summed(*args, **kwargs):
                    return self(*args, **kwargs) + other(*args, **kwargs)
            return summed

    def __mul__(self, other):
            from numbers import Number
            
            if isinstance(other,Number):  #PW - to allow multiplication by ordinary numbers, following answer at http://stackoverflow.com/questions/4233628/override-mul-from-a-child-class-using-parent-implementation-leads-to-proble
                def multiplied(*args, **kwargs):
                        print 'Multiplying FunctionalFunction by Number'
                        return type(self)(self.func(*args, **kwargs) * other)
                return multiplied
                
            else:
                def composed(*args, **kwargs):
                        return self(other(*args, **kwargs))
                return composed

#Function to return the mean of a function, using integrate_quad_gen(), defined in this file. fn should be a function with a 'loc' attribute giving an argument value where substantial probability density can be found.
def get_fn_mean(fn):
    fn2=lambda x: x*fn(x)
    fn2.loc=fn.loc
    return integrate_quad_gen(fn2,-np.inf,np.inf)[0]

#Function to return the standard deviation of a function, using get_fn_var(), defined below
def get_fn_sd(fn):
    return np.sqrt(get_fn_var(fn))

#Function to return the variance of a function, using integrate_quad_gen(), defined in this file. fn should be a function with a 'loc' attribute giving an argument value where substantial probability density can be found.
def get_fn_var(fn):
    fn_mean=get_fn_mean(fn)
    fn2=lambda x: (x-fn_mean)**2 * fn(x)
    fn2.loc=fn.loc
    return integrate_quad_gen(fn2,-np.inf,np.inf)[0]



#Function to return a datetime object corresponding to the middle of a given month. "years" and "months" can be arrays.
def get_midmonth_datetime(years,months):
    import datetime as dt
    import calendar

    try:
        #for arrays, lists etc.
        a=iter(years)
        dates=np.tile(np.datetime64(dt.datetime(1,1,1)).astype(dt.datetime), years.shape)  #whilst it is convoluted to wrap the dates with np.datetime64(), this is the only way I found to get this to work.  
        date_iter=np.nditer(years, flags=['multi_index'])
        while not date_iter.finished:
            ind=date_iter.multi_index
            day=calendar.monthrange(year,month)[1]/2  #calendar.monthrange(...)[1] automatically gets last day of the month, dividing by 2 gets the approx mid-month day
            dates[ind]=dt.datetime(years[ind],months[ind],day)
            date_iter.iternext()
    except:
        day=calendar.monthrange(years,months)[1]/2
        dates=dt.datetime(years,months,day)
    
    return dates
    

#Function to return the nearest-neighbour data value, and the coordinates of the nearest neighbour, for data on any 2D or 3D grid. This supersedes get_nearest_neighbour_irreg() below. Based on http://earthpy.org/interpolation_between_grids_with_ckdtree.html
#points_in is an array with coordinates of the data points (with shape npoints_in x ndims).
#values is an array with the data at each point in points_in (with shape npoints_in x ndims).
#point_out is an array with coordinates of points where data will be interpolated to (with shape npoints_out x ndims) (unless only one point is given, in which case the first dimension is added automatically).
#spherical option makes the code interpret the first dimension of points as longitude and the second as latitude on a sphere. Else points are assumed to be on a Cartesian grid.
#rads option specifies that longitudes and latitudes are in radians. Else they are assumed to be in degrees.
def get_nearest_neighbour(points_in,values,points_out,spherical=None,rads=None):
    from scipy.spatial import cKDTree
    
    if points_out.ndim==1:
        points_out=points_out.reshape(1,len(points_out))
    
    if spherical:
        from python_gen import lon_lat_to_cartesian, cartesian_to_lon_lat
        points_in_save=points_in  #used in conversion of coords back to lat-lon values below
        points_in=np.array(lon_lat_to_cartesian(points_in[:,0],points_in[:,1],rads=rads)).T  #find positions of points on 3D Cartesian grid.
        points_out=np.array(lon_lat_to_cartesian(points_out[:,0],points_out[:,1],rads=rads)).T
    
    ndim=points_in.shape[1]
    x_in=points_in[:,0]
    y_in=points_in[:,1]
    if ndim==2:
        z_in=np.zeros((points_in.shape[0]))
    else:
        z_in=points_in[:,2]
        
    tree = cKDTree(zip(x_in, y_in, z_in))
    
    x_out=points_out[:,0]
    y_out=points_out[:,1]
    if ndim==2:
        z_out=np.zeros((points_out.shape[0]))
    else:
        z_out=points_out[:,2]
    
    distances, inds = tree.query(zip(x_out,y_out,z_out), k=1)  #"k=1" specifies to use the nearest neighbour point only
    
    values_out=values[inds]
    
    points_ret=points_in[inds,:]
    if spherical:
        points_ret=np.array(cartesian_to_lon_lat(points_ret[:,0],points_ret[:,1],points_ret[:,2], rads=rads)).T #transpose to get this into shape npoints x ndims
        if np.max(points_in_save[:,0])>180: points_ret[points_ret[:,0]<0,0]=points_ret[points_ret[:,0]<0,0]+360  #if input lons are in range 0-360, adjust output lons to also be, rather than in the range -180 to 180 as returned by cartesian_to_lon_lat.
        #print points_ret
    
    return values_out, points_ret, inds
    
#Function to return the nearest-neighbour data value, and the coordinates of the nearest neighbour, for data on an irregular grid. Assumes data is 2D spatially.
#data_x and data_y are 1D arrays with the x and y values of the points in data
#data is a 1D array containing the data on the grid
#(x,y) is the coordinate of the point at which data is required
#weights_x and weights_y are arrays of weights to place on the distances in the x- and y-directions respectively at each point - assumes symmetry wrt direction of vector linking points.
#spherical option makes the code interpret the x and y coords as longitude and latitude points on a sphere.
#radians option specifies that longitudes and latitudes are in radians.
def get_nearest_neighbour_irreg(data_x,data_y,data,x,y,weights_x=None,weights_y=None, spherical=None, radians=None):

    if spherical is None:
        if weights_x is None: weights_x=np.ones(len(data_x))
        if weights_y is None: weights_y=np.ones(len(data_y))
        distances=np.sqrt((weights_x*(data_x-x))**2 + (weights_y*(data_y-y))**2)

    else:
        if radians is None:
            lons=data_x*np.pi/180.
            lats=data_y*np.pi/180.
            x=x*np.pi/180.
            y=y*np.pi/180.
        else:
            lons=data_x
            lats=data_y
        
        #calculation of distances uses formula at http://en.wikipedia.org/wiki/Great-circle_distance#Formulas (04/06/13), taking sphere to have unit radius. Set values that are returned outside the range -1 to 1 due to floating point error to be within the range so that arccos does not give an error.
        cos_distances=np.sin(y)*np.sin(lats) + np.cos(y)*np.cos(lats)*np.cos(lons-x)
        cos_distances[cos_distances<-1]=-1
        cos_distances[cos_distances>1]=1
        distances=np.arccos(cos_distances)  
        
    ind=distances.argmin()
#    print ind
#    print x,y,lons[ind],lats[ind],distances[ind]
#    print lons[ind-2:ind+3],lats[ind-2:ind+3],distances[ind-2:ind+3]
#    stop
    
    data_out=data[ind]
    x_pt=data_x[ind]
    y_pt=data_y[ind]
    return data_out, x_pt, y_pt
    

#Function to get directories of data in different datasets
def get_savedir(job):
    
    data_dir_main='/home/jupiter/cpdn/watson/Projects/'
    
    if job in ['b18r','b18s','b19b','b19i','b19j','b19k','b19l','b19x','b19y','b19z','b1a0','b1a4','b1a7','b1a8','b1aa','b1ab','b1ak','b1ax','b1ay','b1az','IFS_T159_clim_sst_stoch']:            
        data_dir_main=data_dir_main+'Trop_WPAC_influence/Plots/'
    elif job not in ['era40','era-int','era_comb','gpcp']: 
        data_dir_main=data_dir_main+'IFS_AMIP/Plots/'
    
    return data_dir_main

#Heaviside step function. Note for x=loc, this returns 0.5.
def heaviside(x,loc=0):
    return 0.5 * (np.sign(x-loc) + 1)

#Function to use scipy.integrate.quad to compute an integral of function (of one variable) fn over interval (a,b), but which can handle integration limits going from minus infinity to infinity if fn has an attribute 'loc', giving a sensible location where fn has enough density that quad can do the separate integrals from -inf to loc and from loc to inf separately.
#Returns the value of the integral and an estimate of the error in a tuple, calculated assuming the errors on each part are independent when integrating from -inf to inf.
#Set tol to vary the acceptable tolerance for the error as a fraction of the integral.
#Set min_i_for_error to specify a threshold that the integral should be larger than for an error to be thrown if the estimated error on the integral is larger than tol.
def integrate_quad_gen(fn,a,b,tol=1e-6,min_i_for_error=0.):
    from scipy.integrate import quad

    if a==-np.inf and b==np.inf:
        try:
            fn.loc
        except:
            raise AttributeError('integrate_quad_gen: fn needs loc attribute when integrating from -inf to inf')
        i1,err1=quad(fn,a,fn.loc)            
        i2,err2=quad(fn,fn.loc,b)            
        i=i1+i2
        err=np.sqrt(err1**2+err2**2)
    else:
        i,err=quad(fn,a,b)
    
    if i>0 and err/i>tol and i>=min_i_for_error:
        raise ValueError('integrate_quad_gen: Error large, err/i = '+str(err/i)+', tol='+str(tol)+', err='+str(err)+', i='+str(i))
    
    return i,err


#Interpolation from hybrid model levels to specified pressure levels (plevs, in hPa), using linear interpolation. Assumes height dimension of input data arr_hyb array is the first. lnsp is the log of surface pressure in hPa. hyb_lev_name is a string specifying which model level structure is being used.
def interp_hybrid_to_p(arr_hyb, lnsp, hyb_lev_name, plevs):
    
    #Get the a and b coeffs of the levels
    if fnmatch.fnmatch(hyb_lev_name,'ifs*'):
        #ECMWF IFS levels
        lev_def_dir='/home/cirrus/pred/watson/Data/'
        if hyb_lev_name=='ifs_62': lev_def_filename=lev_def_dir+'IFS_model_level_def_62.txt'
        elif hyb_lev_name=='ifs_91': lev_def_filename=lev_def_dir+'IFS_model_level_def_91.txt'
        
        lev_def_file=open(lev_def_filename)
        lev_def_file_txt=lev_def_file.readlines()[6:]
        lev_def_file_txt[0]='    0         0.000000     0.000000          0.0000        0.0000\n' #add value to last column
        
        lev_def_arr=np.vstack([line.split() for line in lev_def_file_txt]).astype(np.float)
        lev_nos=lev_def_arr[:,0]
        a=lev_def_arr[:,1]/100. #convert to hPa
        b=lev_def_arr[:,2]
    
    a=np.rollaxis(np.tile(a,lnsp.shape+(1,)), -1)  #makes arrays of a and b coeffs the same shape as the data arrays. The final "+1," in the tuple argument to tile makes the last dimension equal to a for any values of indices of the other dimensions, and rollaxis with argument -1 brings shifts this final dimension to be the first dimension.
    b=np.rollaxis(np.tile(b,lnsp.shape+(1,)), -1)
    lnsp=np.tile(lnsp, (a.shape[0],)+(1,)*lnsp.ndim)
    
    hyb_levs_p_full=a+b*np.exp(lnsp)
    hyb_levs_p=(hyb_levs_p_full[:-1]+hyb_levs_p_full[1:])/2.
    shape=hyb_levs_p.shape
    
    #Get levels and data array into shape with 2 dimensions: levels and then all the data values
    hyb_levs_p_reshaped=hyb_levs_p.reshape(hyb_levs_p.shape[0],len(np.ravel(hyb_levs_p))/hyb_levs_p.shape[0])
    arr_hyb_reshaped=arr_hyb.reshape(hyb_levs_p.shape[0],len(np.ravel(hyb_levs_p))/hyb_levs_p.shape[0])
    arr_plevs=np.zeros((len(plevs),len(arr_hyb_reshaped[0,:])))
    
    for i in range(len(hyb_levs_p_reshaped[0,:])):
        f=interp1d(hyb_levs_p_reshaped[:,i],arr_hyb_reshaped[:,i],axis=0, bounds_error=False) #bounds_error argument stops the next line failing when max(plevs)>surface pressure, and instead assigns NaN to the result at these levels.
        arr_plevs[:,i]=f(plevs)
    
#    import matplotlib.pyplot as plt
#    plt.plot(arr_hyb_reshaped[:,0],hyb_levs_p_reshaped[:,0])
#    plt.plot(arr_plevs[:,0],plevs)
#    plt.show()
#    stop
    
    arr_plevs=arr_plevs.reshape((len(plevs),)+shape[1:])
    
    return arr_plevs

#Calculates Cartesian coordinates of a point on a sphere with radius R. The z-axis increases northwards.
#Modified from code at http://earthpy.org/interpolation_between_grids_with_ckdtree.html
#Added "rads" option - set to specify that coords are in radians. Else it is assumed they are in degrees.
def lon_lat_to_cartesian(lon, lat, R = 1, rads=None):

    if not rads:
        lon_r = np.radians(lon)
        lat_r = np.radians(lat)
    else:
        lon_r=lon
        lat_r=lat

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z
    
    
#Moving averages, modified from Jaime's answer at http://stackoverflow.com/questions/14313510/moving-average-function-on-numpy-scipy
#a - input array  n - no. of points over which to take moving average (an odd no.) axis - axis of array over which to do moving average
#set cyclic option when the ends of the input array should be treated as being joined. 
def moving_average(a,n,axis=0,cyclic=None):

    a=a.swapaxes(0,axis)
    
    if cyclic:
        if n%2==1:  #odd n
            a=np.concatenate([a[a.shape[0]-(n-1)/2:,...],a,a[:(n-1)/2,...]])
        else:  #even n
            a=np.concatenate([a[a.shape[0]-n/2:,...],a,a[:n/2-1,...]])  #arbitrarily choose to put (n/2) of the n-1 elements formed by averaging over the start and end of the array at the beginning and (n/2)-1 at the end.
        #print a
    
    #This code gets the sequence which is the average over the first n elements of a, then the 2nd to the n+1'th, then the 3rd to the n+2'th etc.
    ret = np.cumsum(a, axis=0)
    ret[n:,...] = ret[n:,...] - ret[:-n,...]
    ret=ret[n - 1:,...] / float(n)
    ret=ret.swapaxes(0,axis)
    
    return ret


#Python function to create a dimension axis in a NetCDF file with associated axis values
#ncfile is the Scientific.IO.NetCDF file object in which the dimension is being created
#dim_name is string giving the dimension name
#dim_values is a numpy array containing numeric values of the dimension
#units is optional string giving dimension units
#Note since NetCDF3 does not support 64-bit ints, this could fail if the dimension values are large integers.
def netcdf_create_dimension(ncfile,dim_name,dim_values,units=None):
  
    dtype=type(dim_values[0])
    if np.dtype(dtype).char=='l':
        dim_values=dim_values.astype('i')
        dtype=type(dim_values[0])

    #print dim_name,dtype,np.dtype(dtype).char,dim_values

    ncfile.createDimension(dim_name,len(dim_values))
    axis=ncfile.createVariable(dim_name,np.dtype(dtype).char,(dim_name,)) 
    axis[:]=dim_values
    if units != None: axis.units=units


#Function to do conservative regridding of data on a reduced Gaussian grid using the ncl ESMF_regrid function. It first writes the data to a netCDF file, which gets read into by ncl, which writes a NetCDF file with the result, which gets read in by python and returned.
#data is the data array
#lons and lats are 1D arrays containing the lon and lat coordinates for each datapoint respectively.
#outgrid is a string grid description that can be recognised by ncl e.g. "0.25x0.25" or "0.25deg" for 0.25 deg by 0.25 deg lat-lon grid, "G64" for N64 Gaussian grid
#spacedim is an optional argument specifying which dimension of data is the spatial dimension. If this is not set, the script identifies this dimension as the first that has the same length as lons and lats, which could be dangerous if another dimension happens to have the same length.
#See http://www.ncl.ucar.edu/Document/Functions/ESMF/ESMF_regrid.shtml for more documentation of the ncl ESMF_regrid function
def regrid_cons_rg(data,lons,lats,outgrid,spacedim=None):

    #Produce a NetCDF file containing the data
    ncfile=NetCDFFile('regrid_cons_rg.nc','w')
    
    ndims=len(data.shape)
    spacedim_set_flag=0  #to stop the function creating the spatial dimension twice if there is more than one dimension with the same length as xvals
    dim_names=['']*ndims

    if spacedim:
        dim_name='space'
        ncfile.createDimension(dim_name,data.shape[spacedim])
        spacedim_set_flag=1
        dim_names[spacedim]=dim_name
  
    for i in range(ndims):
        if i!=spacedim:
            if data.shape[i]==len(lons) and spacedim_set_flag==0:
                dim_name='space'
                spacedim_set_flag=1
            else:
                dim_name='dim_'+str(i)
                
            ncfile.createDimension(dim_name,data.shape[i])
            dim_names[i]=dim_name

    dtype=type(np.ravel(data)[0])
    ncdata=ncfile.createVariable('data',np.dtype(dtype).char,tuple(dim_names))
    ncdata[:]=data
    lon1d=ncfile.createVariable('lon1d',np.dtype(dtype).char,('space',))  #lons1d is the dimension name required by the ncl ESMF_regrid function
    lon1d[:]=lons
    lat1d=ncfile.createVariable('lat1d',np.dtype(dtype).char,('space',))  #lats1d is the dimension name required by the ncl ESMF_regrid function
    lat1d[:]=lats

#    #Set the coordinates as attributes, as required for the ncl ESMF_regrid function
#    setattr(ncdata,'lon1d',xvals)
#    setattr(ncdata,'lat1d',yvals)

    #Create corner information for the cells of input grid. This is particular to reduced Gaussian grids, which have the points arranged in lines of constant latitude.
    ncfile.createDimension('corner',4)
    
    corner_lons=np.zeros((len(lons),4))
    corner_lats=np.zeros((len(lons),4))
    lats_unique=list(sorted(set(lats)))  #gets sorted list of the unique latitudes
    for i in range(len(lats_unique)):
        
        lat=lats_unique[i]
        ind=np.where(lats==lat)[0]

        #Set longitudes of corners at this latitude
        lons_at_lat=lons[ind]
        dlon=lons_at_lat[1]-lons_at_lat[0]
        
        #Pick convention that corner 0 is bottom left, corner 1 is top left, corner 2 is top right and corner 3 is bottom right (where north is up and east is right).
        corner_lons[ind,0]=lons_at_lat-dlon/2.
        corner_lons[ind,1]=lons_at_lat-dlon/2.
        corner_lons[ind,2]=lons_at_lat+dlon/2.
        corner_lons[ind,3]=lons_at_lat+dlon/2.
        
        #Set latitudes of corners at this latitude
        if i==0:  #minimum lat
            dlat1=lats_unique[i+1]-lats_unique[i]  #the step to the next latitude
        
            corner_lats[ind,0]=-90.  #set southern corner latitudes to be -90 for the southmost row.
            corner_lats[ind,1]=lat+dlat1/2.  #set other corners to be halfway between this and the next row
            corner_lats[ind,2]=lat+dlat1/2.
            corner_lats[ind,3]=-90.

        elif i==(len(lats_unique)-1):  #max lat
            dlat2=lats_unique[i]-lats_unique[i-1]  #the step from the previous latitude
            corner_lats[ind,0]=lat-dlat2/2.
            corner_lats[ind,1]=90.  #set northern corner latitudes to be 90 for the northmost row.
            corner_lats[ind,2]=90.
            corner_lats[ind,3]=lat-dlat2/2.

        else:
            dlat1=lats_unique[i+1]-lats_unique[i]
            dlat2=lats_unique[i]-lats_unique[i-1]
            corner_lats[ind,0]=lat-dlat2/2.
            corner_lats[ind,1]=lat+dlat1/2.
            corner_lats[ind,2]=lat+dlat1/2.
            corner_lats[ind,3]=lat-dlat2/2.

    corner_lons_nc=ncfile.createVariable('corner_lons',np.dtype(dtype).char,('space','corner'))
    corner_lons_nc[:,:]=corner_lons
    corner_lats_nc=ncfile.createVariable('corner_lats',np.dtype(dtype).char,('space','corner'))
    corner_lats_nc[:,:]=corner_lats

    ncfile.close()
    
    stop


#Function to do conservative regridding of data, 
#INCOMPLETE
#Idea is to apply the assumption that the input data at each grid point represents the value of some function that is uniform over a cell, consisting of all the points for which that grid point is the closest, averaged over that cell. The regridded data at each grid point is the sum over all the cells that overlap with its own cell of the input data multiplied by the fraction of each cell that is inside the output cell. This has the effect of smearing out the input data where the output grid is finer, and of spatially averaging the input data where the output grid is coarser. 
#THIS ASSUMES THAT THE INPUT GRID IS NOT COMPLETELY IRREGULAR - THE X COORDINATES SHOULD BE ARRANGED IN ROWS WITH ENOUGH DATA SHARING EACH Y COORDINATE, SINCE THE CODE ASSUMES THAT POINTS AT A GIVEN Y ARE TO BE ASSOCIATED WITH A CELL IN THE SAME ROW. THE X COORDS ON EACH ROW SHOULD BE REGULARLY SPACED (BUT THE SPACING MAY DIFFER ON DIFFERENT ROWS). THIS SHOULD BE COMPATIBLE WITH REGULAR AND REDUCED GAUSSIAN GRIDS.
#data is the input data array (1D or 2D). The spatial dimension(s) are assumed to be the last dimensions.
#x_in and y_in are 1D arrays holding the coordinates of data - if data is 1D (e.g. for an unstructured grid) then x and y should have the same length as data.
#x_out and y_out are coordinates of the output grid. If irreg_out is not set, then x_out and y_out will be meshed to form a 2D grid and there will be x_out*y_out returned data values in a 2D array. Specify the irreg_out option if the output grid is irregular, in each case there will be only one output data value for each value of x_out/y_out, and the returned array will be 1D.
#Set norm_by_cell_area when the input data represents an accumulation over the domain area, so that the data magnitude should be made smaller if the output cells are smaller.
#Set minx,maxx,miny,maxy to specify bounds that the data should not be assumed to lie outside e.g. if y_in is latitude and includes -90, set miny=-90 so that the cells with latitude -90 are not assumed to cover y values less than -90.
#Set wrap_x/wrap_y to specify that the x/y coords are cyclic (e.g. for longitude), so they should be wrapped around. min and max values for the axes should also be set, so the code knows where the connecting point is.
def regrid_cons_simple(data,x_in,y_in,x_out,y_out,irreg_out=None,norm_by_cell_area=None,minx=None,maxx=None,miny=None,maxy=None,wrap_x=None,wrap_y=None):

#Steps:
#1. Identify corners of cells for each input grid point - actually it might be easier to use the cell edges.
#2. Normalise the input data according to the size of each cell if norm_by_cell_area is not set.
#3. For each output grid cell, identify which input grid points correspond to cells that overlap with the output cell i.e. which input cells have edges within the output cell.
#4. For each output grid cell, sum up the contributions data*fraction of cell for each input grid cell that overlaps with it.

    #Make data array 1D - do nothing if it is a list
    if type(data)==np.ndarray:  
        
        #Get coordinates of every point in data separately
        if data.ndim==2:
            x_in2=np.meshgrid(x_in,y_in)[0].ravel()
            y_in=np.meshgrid(x_in,y_in)[1].ravel()
            x_in=x_in2            
            
        data=data.ravel()
        
    
    #1. Identify corners of cells for each input grid point.
    corner_xs=np.zeros((len(data),4))
    corner_ys=np.zeros((len(data),4))
    ys_unique=list(sorted(set(y_in)))  #gets sorted list of the unique latitudes
    for i in range(len(ys_unique)):
        
        y=ys_unique[i]
        ind=np.where(y_in==y)[0]
    
        #Set x coords of corners at this y - assumes regular spacing
        xs_at_y=x_in[ind]
        dx=xs_at_y[1]-xs_at_y[0]
        np.testing.assert_array_equal(xs_at_y[1:]-xs_at_y[:-1], np.repeat(dx,len(ind)-1),'x-values do not have even spacing at y='+str(y))
        
        #Pick convention that corner 0 is bottom left, corner 1 is top left, corner 2 is top right and corner 3 is bottom right (where north is up and east is right).
        corner_xs[ind,0]=xs_at_y-dx/2.
        corner_xs[ind,1]=xs_at_y-dx/2.
        corner_xs[ind,2]=xs_at_y+dx/2.
        corner_xs[ind,3]=xs_at_y+dx/2.
        
        #THE SETTING OF MIN AND MAX X IS NOT WORKING AND I DON'T KNOW WHY
        if minx != None:
            for j in range(2):  #only need to adjust the top and bottom left corners
                if wrap_x:
                    #Set corners dx/2 past the axis endpoint
                    corner_xs[corner_xs[:,j]<minx,j]=maxx-dx/2
                else:
                    corner_xs[corner_xs[:,j]<minx,j]=minx
        if maxx != None:
            for j in range(2,4):  #only need to adjust the top and bottom right corners
                if wrap_x:
                    corner_xs[corner_xs[:,j]>maxx,j]=minx+dx/2
                else:
                    corner_xs[corner_xs[:,j]>maxx,j]=maxx
        
        #Set y coords of corners for data at this y
        if miny != None and i==0:  #minimum y
            dy1=ys_unique[i+1]-ys_unique[i]  #the step to the next y
        
            corner_ys[ind,0]=miny  #set southern corner y coords to be miny for the first row.
            corner_ys[ind,1]=y+dy1/2.  #set other corners to be halfway between this and the next row
            corner_ys[ind,2]=y+dy1/2.
            corner_ys[ind,3]=miny
    
        elif maxy != None and i==(len(ys_unique)-1):  #max y
            dy2=ys_unique[i]-ys_unique[i-1]
            corner_ys[ind,0]=y-dy2/2.
            corner_ys[ind,1]=maxy  #set northern corner y coords to be maxy for the last row.
            corner_ys[ind,2]=maxy
            corner_ys[ind,3]=y-dy2/2.
    
        else:
            dy1=ys_unique[i+1]-ys_unique[i]
            dy2=ys_unique[i]-ys_unique[i-1]
            corner_ys[ind,0]=y-dy2/2.
            corner_ys[ind,1]=y+dy1/2.
            corner_ys[ind,2]=y+dy1/2.
            corner_ys[ind,3]=y-dy2/2.
    
    plt.figure()
    plt.scatter(x_in,y_in,marker='.')
    plt.scatter(corner_xs[:,0],corner_ys[:,0],marker='x')    

    plt.figure()
    plt.scatter(x_in,y_in,marker='.')
    plt.scatter(corner_xs[:,1],corner_ys[:,1],marker='x')    

    plt.figure()
    plt.scatter(x_in,y_in,marker='.')
    plt.scatter(corner_xs[:,2],corner_ys[:,2],marker='x')    

    plt.figure()
    plt.scatter(x_in,y_in,marker='.')
    plt.scatter(corner_xs[:,3],corner_ys[:,3],marker='x')    
    
    plt.show()

#Function to remove zonal mean from array arr. axis is the dimension of the longitude axis.
def rem_zm(arr,axis=0):
    tile_tup=(1,)*axis+(arr.shape[axis],)+(1,)*(arr.ndim-axis-1)    
    arr=arr-np.tile(np.mean(arr,axis=axis)[...,np.newaxis], tile_tup)
    return arr

#Function to round numbers to a given number n of significant figures
#Modified from Roy Hyunjin Han's answer at http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
#Supersedes round_to_precision() below
def round_to_n(x,n,round_up=None,round_down=None):

    if round_up and round_down:
        print 'round_to_n: Cannot set both round_up and round_down'
        sys.exit()

    x=np.array(x)
    x_shape=x.shape
    x=np.ravel(x)
    ret=np.zeros(len(x))
    for ind,i in enumerate(x):
        if not np.isfinite(i):  #to allow infinite or NaN values to be handled
            ret[ind]=i
        elif i==0:
            ret[ind]=0
        else:
            tens=int(np.floor(np.log10(abs(i)))) - (n - 1)
            ret[ind]=round(i, -tens)

            if round_up and ret[ind]<i:
                ret[ind]+=math.pow(10,tens)
            if round_down and ret[ind]>i:
                ret[ind]-=math.pow(10,tens)
    
    ret=ret.reshape(x_shape)
    return ret

#Function to round numbers to a given number p of significant figures
#Modified from code at http://randlet.com/blog/python-significant-figures-format/
#Added noexp argument to specify not to use exponential notation
def round_to_precision(x,p,noexp=None):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """
    
    if np.isfinite(x):  #PW - to allow infinite or NaN values to be handled
        x = float(x)

        if x == 0.:
          return "0." + "0"*(p-1)
        out = []

        if x < 0:
          out.append("-")
          x = -x

        e = int(math.log10(x))
        tens = math.pow(10, e - p + 1)
        n = math.floor(x/tens)

        if n < math.pow(10, p - 1):
          e = e -1
          tens = math.pow(10, e - p+1)
          n = math.floor(x / tens)

        if abs((n + 1.) * tens - x) <= abs(n * tens -x):
          n = n + 1

        if n >= math.pow(10,p):
          n = n / 10.
          e = e + 1

        m = "%.*g" % (p, n)

        if e < -2 or e >= p and noexp==None:  #PW - added " and noexp==None"
          out.append(m[0])
          if p > 1:
              out.append(".")
              out.extend(m[1:p])
          out.append('e')
          if e > 0:
              out.append("+")
          out.append(str(e))
        elif e < -2 or e >= p and noexp:          #PW
          out.append(m)
          out.extend(["0"]*(e+1-len(m)))
        elif e == (p -1):
          out.append(m)
        elif e >= 0:
          out.append(m[:e+1])
          if e+1 < len(m):
              out.append(".")
              out.extend(m[e+1:])
        else:
          out.append("0.")
          out.extend(["0"]*-(e+1))
          out.append(m)


    elif not np.isfinite(x):  #PW
        out=str(x)

    return "".join(out)


#Function to wrap one dimension of an array around on itself - useful when making stereographic plots to stop whitespace appearing where the ends of the longitude axis are. axis specifies the axis of arr to be wrapped around. 
def wrap_around(arr,axis=0):
    arr=np.rollaxis(arr,axis)
    arr=np.concatenate((arr, arr[0,...].reshape((1,)+arr.shape[1:])))
    arr=np.rollaxis(arr,0,axis+1)
    
    return arr
