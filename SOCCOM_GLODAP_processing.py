import xarray as xr
import dask.array
import numpy as np


def interpolated_SOCCOM_floats(variable, float_dir, float_str, float_end='SOOCNQC.nc'):
    """
    Aggregates all SOCCOM float data from files float_dir/float_str+float_end(e.g. '/work/Ruth.Moorman/SOCCOM-floats/LR_SOCCOM_20200308/0037SOOCNQC.nc') into a single xarray datastructure with dimensions N_PROF and depth.
    depth_bin dimension is determined interpolating SOCCOM float data onto a 2001 level 1m grid [0,1,...,2000] using 4 interpolation methods: linear, nearest neighbour, cubic spline and quadratic spline (so later uncertainty associated with the interpolation method may be quantified)
    
    float_str will be an array of strings designating files for different floats e.g ['11090', '12545', '12733', '12784'...]
    variable is a string that is set to one of the following:
    
    Temperature
    Salinity
    Sigma_theta
    Oxygen
    Nitrate
    Chl_a
    POC
    TALK
    DIC
    
    will return 4 DataArray structures associated with the 4 interpolation methods.
    return tracer_linear, tracer_nearest_neighbour, tracer_cubic, tracer_quadratic
    
    (change function if using CANYON or MLR versions of TALK, DIC, pCO2)
    Note, not all floats have TALK/ DIC/ pCO2 values, may need different float_str for these.
    Ruth Moorman, Princeton University, Mar 2020
    """
    
    interp_depths = np.linspace(2000,0,2001)
    
    ################################### initialise tracer array with first float ###################################
    float_file = xr.open_dataset(float_dir+float_str[0]+float_end)
    float_lons = float_file.Lon.values
    float_lats = float_file.Lat.values
    float_times = float_file.JULD.values
    float_no = float_str.astype('float64')
    float_name = np.zeros(len(float_lons))+float_no[0]

    if variable == 'Temperature':
        tracer = float_file.Temperature
        units = tracer.units
        QC_mask = float_file.Temperature_QFA.where(float_file.Temperature_QFA ==0)+1 # code 0 is QC approved data
        tracer = tracer * QC_mask
    elif variable == 'Salinity':
        tracer = float_file.Salinity
        units = tracer.units
        QC_mask = float_file.Salinity_QFA.where(float_file.Salinity_QFA ==0)+1 
        tracer = tracer * QC_mask
    elif variable == 'Sigma_theta':
        tracer = float_file.Sigma_theta
        units = tracer.units
        QC_mask = float_file.Sigma_theta_QFA.where(float_file.Sigma_theta_QFA ==0)+1 
        tracer = tracer * QC_mask
    elif variable == 'Oxygen':
        tracer = float_file.Oxygen
        units = tracer.units
        QC_mask = float_file.Oxygen_QFA.where(float_file.Oxygen_QFA ==0)+1 
        tracer = tracer * QC_mask
    elif variable == 'Nitrate':
        tracer = float_file.Nitrate
        units = tracer.units
        QC_mask = float_file.Nitrate_QFA.where(float_file.Nitrate_QFA ==0)+1 
        tracer = tracer * QC_mask
        tracer = tracer.where(tracer >0) # there were negative concentrations, unsure how this wasn't picked up in the QC flag
    elif variable == 'Chl_a':
        tracer = float_file.Chl_a
        units = tracer.units
        QC_mask = float_file.Chl_a_QFA.where(float_file.Chl_a_QFA ==0)+1 
        tracer = tracer * QC_mask
    elif variable == 'POC':
        tracer = float_file.POC
        units = tracer.units
        QC_mask = float_file.POC_QFA.where(float_file.POC_QFA ==0)+1 
        tracer = tracer * QC_mask
    elif variable == 'TALK':
        tracer = float_file.TALK_LIAR
        units = tracer.units
        QC_mask = float_file.TALK_LIAR_QFA.where(float_file.TALK_LIAR_QFA ==0)+1 
        tracer = tracer * QC_mask
    elif variable == 'DIC':
        tracer = float_file.DIC_LIAR
        units = tracer.units
        QC_mask = float_file.DIC_LIAR_QFA.where(float_file.DIC_LIAR_QFA ==0)+1 
        tracer = tracer * QC_mask
    else:
        return print('Use valid variable option, help(gridded_SOCCOM_floats) for options')
    
    
    
    soccom_tracer_linear = xr.DataArray(np.empty((len(float_file.N_PROF), len(interp_depths))),coords = [float_file.N_PROF,interp_depths],dims = ['N_PROF','depth'])
    soccom_tracer_nearest = xr.DataArray(np.empty((len(float_file.N_PROF), len(interp_depths))),coords = [float_file.N_PROF,interp_depths],dims = ['N_PROF','depth'])
    soccom_tracer_cubic = xr.DataArray(np.empty((len(float_file.N_PROF), len(interp_depths))),coords = [float_file.N_PROF,interp_depths],dims = ['N_PROF','depth'])
    soccom_tracer_quad = xr.DataArray(np.empty((len(float_file.N_PROF), len(interp_depths))),coords = [float_file.N_PROF,interp_depths],dims = ['N_PROF','depth'])

    soccom_tracer_linear[:,:] = np.nan
    soccom_tracer_nearest[:,:] = np.nan
    soccom_tracer_cubic[:,:] = np.nan
    soccom_tracer_quad[:,:] = np.nan
    
    for i in range(len(float_file.N_PROF)):
        tracer_i = tracer.isel(N_PROF =i)
        tracer_i.coords['N_LEVELS'] = float_file.Depth.isel(N_PROF =i).values
        tracer_i = tracer_i.rename({'N_LEVELS':'depth'})
        tracer_i=tracer_i.where(np.abs(tracer_i)>0,drop = True ) # remove nans, they muck up interpolation
        tracer_i = tracer_i.where(np.abs(tracer_i.depth)>0,drop = True ) #nans aren't removed by uniqueness filter below
        _, unique_depth_index = np.unique(tracer_i.depth, return_index=True) # remove repeat depth measurements, again, they muck up interpolation (they don't seem to be common, I'm not concerned by removing)
        tracer_i = tracer_i.isel(depth = unique_depth_index)
        if len(tracer_i.depth) > 10:
            soccom_tracer_linear[i,:] = tracer_i.interp(depth = interp_depths, method = 'linear')
            soccom_tracer_cubic[i,:] = tracer_i.interp(depth = interp_depths, method = 'cubic')
            soccom_tracer_nearest[i,:] = tracer_i.interp(depth = interp_depths, method = 'nearest')
            soccom_tracer_quad[i,:] = tracer_i.interp(depth = interp_depths, method = 'quadratic')
        else: 
            fill = np.empty((len(interp_depths)))
            fill[:] = np.nan
            soccom_tracer_linear[i,:] = fill
            soccom_tracer_cubic[i,:] = fill
            soccom_tracer_nearest[i,:] = fill
            soccom_tracer_quad[i,:] = fill
                
    count = len(float_file.N_PROF)
    
    ############################################ gather data from all floats #########################################
    for j in range(1,len(float_str)):
        float_file = xr.open_dataset(float_dir+float_str[j]+float_end)

        ## record lat, lon, name, time
        float_lons = np.append(float_lons, float_file.Lon.values)
        float_lats = np.append(float_lats, float_file.Lat.values)
        float_times = np.append(float_times, float_file.JULD.values)
        float_name = np.append(float_name, np.zeros(len(float_file.Lon.values))+float_no[j])

        if variable == 'Temperature':
            tracer = float_file.Temperature
            units = tracer.units
            QC_mask = float_file.Temperature_QFA.where(float_file.Temperature_QFA ==0)+1 # code 0 is QC approved data
            tracer = tracer * QC_mask
        elif variable == 'Salinity':
            tracer = float_file.Salinity
            units = tracer.units
            QC_mask = float_file.Salinity_QFA.where(float_file.Salinity_QFA ==0)+1 
            tracer = tracer * QC_mask
        elif variable == 'Sigma_theta':
            tracer = float_file.Sigma_theta
            units = tracer.units
            QC_mask = float_file.Sigma_theta_QFA.where(float_file.Sigma_theta_QFA ==0)+1 
            tracer = tracer * QC_mask
        elif variable == 'Oxygen':
            tracer = float_file.Oxygen
            units = tracer.units
            QC_mask = float_file.Oxygen_QFA.where(float_file.Oxygen_QFA ==0)+1 
            tracer = tracer * QC_mask
        elif variable == 'Nitrate':
            tracer = float_file.Nitrate
            units = tracer.units
            QC_mask = float_file.Nitrate_QFA.where(float_file.Nitrate_QFA ==0)+1 
            tracer = tracer * QC_mask
            tracer = tracer.where(tracer >0) # there were negative concentrations, unsure how this wasn't picked up in the QC flag 
        elif variable == 'Chl_a':
            tracer = float_file.Chl_a
            units = tracer.units
            QC_mask = float_file.Chl_a_QFA.where(float_file.Chl_a_QFA ==0)+1 
            tracer = tracer * QC_mask
        elif variable == 'POC':
            tracer = float_file.POC
            units = tracer.units
            QC_mask = float_file.POC_QFA.where(float_file.POC_QFA ==0)+1 
            tracer = tracer * QC_mask
        elif variable == 'TALK':
            tracer = float_file.TALK_LIAR
            units = tracer.units
            QC_mask = float_file.TALK_LIAR_QFA.where(float_file.TALK_LIAR_QFA ==0)+1 
            tracer = tracer * QC_mask
        elif variable == 'DIC':
            tracer = float_file.DIC_LIAR
            units = tracer.units
            QC_mask = float_file.DIC_LIAR_QFA.where(float_file.DIC_LIAR_QFA ==0)+1 
            tracer = tracer * QC_mask
        else:
            return print('Use valid variable option, help(gridded_SOCCOM_floats) for options')
        
        
        tracer_linear = xr.DataArray(np.empty((len(float_file.N_PROF), len(interp_depths))),coords = [float_file.N_PROF,interp_depths],dims = ['N_PROF','depth'])
        tracer_nearest = xr.DataArray(np.empty((len(float_file.N_PROF), len(interp_depths))),coords = [float_file.N_PROF,interp_depths],dims = ['N_PROF','depth'])
        tracer_cubic = xr.DataArray(np.empty((len(float_file.N_PROF), len(interp_depths))),coords = [float_file.N_PROF,interp_depths],dims = ['N_PROF','depth'])
        tracer_quad = xr.DataArray(np.empty((len(float_file.N_PROF), len(interp_depths))),coords = [float_file.N_PROF,interp_depths],dims = ['N_PROF','depth'])
        
        tracer_linear[:,:] = np.nan
        tracer_nearest[:,:] = np.nan
        tracer_cubic[:,:] = np.nan
        tracer_quad[:,:] = np.nan
        
        for i in range(len(float_file.N_PROF)):
            tracer_i = tracer.isel(N_PROF =i)
            tracer_i.coords['N_LEVELS'] = float_file.Depth.isel(N_PROF =i).values
            tracer_i = tracer_i.rename({'N_LEVELS':'depth'})
            tracer_i=tracer_i.where(np.abs(tracer_i)>0,drop = True ) # remove nans, they muck up interpolation
            tracer_i = tracer_i.where(np.abs(tracer_i.depth)>0,drop = True ) #nans aren't removed by uniqueness filter below and non uniqueness interferes with the interpolation protocol
            _, unique_depth_index = np.unique(tracer_i.depth, return_index=True) # remove repeat depth measurements, again, they muck up interpolation (they don't seem to be common, I'm not concerned by removing)
            tracer_i = tracer_i.isel(depth = unique_depth_index)
            if len(tracer_i.depth) > 10:
                tracer_linear[i,:] = tracer_i.interp(depth = interp_depths, method = 'linear')
                tracer_cubic[i,:] = tracer_i.interp(depth = interp_depths, method = 'cubic')
                tracer_nearest[i,:] = tracer_i.interp(depth = interp_depths, method = 'nearest')
                tracer_quad[i,:] = tracer_i.interp(depth = interp_depths, method = 'quadratic')
            else: ## there were a few profiles (4 to be precise) that had a tiny number of measurements. These data are questionable and hard to interpolate so I just threw them out
                fill = np.empty((len(interp_depths)))
                fill[:] = np.nan
                tracer_linear[i,:] = fill
                tracer_cubic[i,:] = fill
                tracer_nearest[i,:] = fill
                tracer_quad[i,:] = fill
            
        tracer_linear.coords['N_PROF'] = tracer_linear.N_PROF.values + count
        tracer_cubic.coords['N_PROF'] = tracer_cubic.N_PROF.values + count
        tracer_nearest.coords['N_PROF'] = tracer_nearest.N_PROF.values + count
        tracer_quad.coords['N_PROF'] = tracer_quad.N_PROF.values + count   
        count = count + len(float_file.N_PROF)
        
        soccom_tracer_linear = xr.concat((soccom_tracer_linear, tracer_linear), dim = 'N_PROF')
        soccom_tracer_nearest = xr.concat((soccom_tracer_nearest, tracer_nearest), dim = 'N_PROF')
        soccom_tracer_cubic = xr.concat((soccom_tracer_cubic, tracer_cubic), dim = 'N_PROF')
        soccom_tracer_quad = xr.concat((soccom_tracer_quad, tracer_quad), dim = 'N_PROF')
        
        
        
    soccom_lons = xr.DataArray(float_lons, coords = [soccom_tracer_linear.N_PROF], dims = 'N_PROF' )
    soccom_lats = xr.DataArray(float_lats, coords = [soccom_tracer_linear.N_PROF], dims = 'N_PROF' )
    soccom_times = xr.DataArray(float_times, coords = [soccom_tracer_linear.N_PROF], dims = 'N_PROF' )
    soccom_name = xr.DataArray(float_name, coords = [soccom_tracer_linear.N_PROF], dims = 'N_PROF' )
    
    soccom_tracer_linear = xr.DataArray(soccom_tracer_linear, coords = {'N_PROF':soccom_tracer_linear.N_PROF.values, 'depth':soccom_tracer_linear.depth.values, 'lat':soccom_lats, 'lon': soccom_lons, 'time': soccom_times, 'float_name': soccom_name})
    soccom_tracer_linear.attrs['units']= units
    
    soccom_tracer_nearest = xr.DataArray(soccom_tracer_nearest, coords = {'N_PROF':soccom_tracer_nearest.N_PROF.values, 'depth':soccom_tracer_nearest.depth.values, 'lat':soccom_lats, 'lon': soccom_lons, 'time': soccom_times, 'float_name': soccom_name})
    soccom_tracer_nearest.attrs['units']= units
    
    soccom_tracer_cubic = xr.DataArray(soccom_tracer_cubic, coords = {'N_PROF':soccom_tracer_cubic.N_PROF.values, 'depth':soccom_tracer_cubic.depth.values, 'lat':soccom_lats, 'lon': soccom_lons, 'time': soccom_times, 'float_name': soccom_name})
    soccom_tracer_cubic.attrs['units']= units
    
    soccom_tracer_quad = xr.DataArray(soccom_tracer_quad, coords = {'N_PROF':soccom_tracer_quad.N_PROF.values, 'depth':soccom_tracer_quad.depth.values, 'lat':soccom_lats, 'lon': soccom_lons, 'time': soccom_times, 'float_name': soccom_name})
    soccom_tracer_quad.attrs['units']= units
    
    return soccom_tracer_linear, soccom_tracer_nearest, soccom_tracer_cubic, soccom_tracer_quad

def depth_binned_profiles_Bronselaer(variable, float_dir, prof):
    """
    Depth averages interpolated SOCCOM or GLODAP profiles into bins used by Ben Bronselaer in 2020 nature geoscience paper.
    Note interpolated values must be found in float_dir+'ALL_'+prof+'_TEMPERATURE-vertically-interpolated.nc' or similar.
    return SOCCOM_linear_depth_binned, SOCCOM_nearest_depth_binned, SOCCOM_cubic_depth_binned, SOCCOM_quad_depth_binned
    Ruth Moorman, Princeton University, Mar 2020
    """
    if variable == 'Temperature':
        soccom = xr.open_dataset(float_dir+'ALL_'+prof+'_TEMPERATURE-vertically-interpolated.nc')
    elif variable == 'Salinity':
        soccom = xr.open_dataset(float_dir+'ALL_'+prof+'_SALINITY-vertically-interpolated.nc')
    elif variable == 'Sigma_theta':
        soccom = xr.open_dataset(float_dir+'ALL_'+prof+'_SIGMA-THETA-vertically-interpolated.nc')
    elif variable == 'Oxygen':
        soccom = xr.open_dataset(float_dir+'ALL_'+prof+'_OXYGEN-vertically-interpolated.nc')
    elif variable == 'Nitrate':
        soccom = xr.open_dataset(float_dir+'ALL_'+prof+'_NITRATE-vertically-interpolated.nc')
    elif variable == 'Chl_a':
        soccom = xr.open_dataset(float_dir+'ALL_'+prof+'_CHLA-vertically-interpolated.nc')
    elif variable == 'POC':
        soccom = xr.open_dataset(float_dir+'ALL_'+prof+'_POC-vertically-interpolated.nc')
    elif variable == 'TALK':
        soccom = xr.open_dataset(float_dir+'ALL_'+prof+'_TALK-vertically-interpolated.nc')
    elif variable == 'DIC':
        soccom = xr.open_dataset(float_dir+'ALL_'+prof+'_DIC-vertically-interpolated.nc')
    else:
        return print('Use valid variable option, help(gridded_SOCCOM_floats) for options')

    SOCCOM_linear = soccom.linear
    SOCCOM_nearest = soccom.nearest
    SOCCOM_cubic = soccom.cubic
    SOCCOM_quad = soccom.quad
    depth_bin_center = np.array([4.,7.,12.,19.,26.,35.,47.,65.,91.,126.,179.,249.,346.,468.,614.,772.,947.,1123.,1298.,1474.,1649.,1825.,2000.])
    depth_bin_bounds = np.array([2.5,5.5,8.5,15.5,22.5,29.5,40.5,53.5,76.5,105.5,146.5,211.5,286.5,405.5,530.5,697.5,846.5,1047.5,1198.5,1397.5,1550.5,1747.5,1902.5,2097.5,])

    ## groupby depth bins and mean
    SOCCOM_linear_depth_binned = SOCCOM_linear.groupby_bins('depth', depth_bin_bounds).mean()
    SOCCOM_nearest_depth_binned = SOCCOM_nearest.groupby_bins('depth', depth_bin_bounds).mean()
    SOCCOM_cubic_depth_binned = SOCCOM_cubic.groupby_bins('depth', depth_bin_bounds).mean()
    SOCCOM_quad_depth_binned = SOCCOM_quad.groupby_bins('depth', depth_bin_bounds).mean()

    ## replace depth bin tuple coordinate with bin center values (but retain bin bounds as a separate dimension)
    SOCCOM_linear_depth_binned.coords['depth_bin_bounds'] = SOCCOM_linear_depth_binned.depth_bins
    SOCCOM_nearest_depth_binned.coords['depth_bin_bounds'] = SOCCOM_nearest_depth_binned.depth_bins
    SOCCOM_cubic_depth_binned.coords['depth_bin_bounds'] = SOCCOM_cubic_depth_binned.depth_bins
    SOCCOM_quad_depth_binned.coords['depth_bin_bounds'] = SOCCOM_quad_depth_binned.depth_bins
    SOCCOM_linear_depth_binned.coords['depth_bins'] = depth_bin_center
    SOCCOM_nearest_depth_binned.coords['depth_bins'] = depth_bin_center
    SOCCOM_cubic_depth_binned.coords['depth_bins'] = depth_bin_center
    SOCCOM_quad_depth_binned.coords['depth_bins'] = depth_bin_center
    
    return SOCCOM_linear_depth_binned, SOCCOM_nearest_depth_binned, SOCCOM_cubic_depth_binned, SOCCOM_quad_depth_binned

def SOCCOM_horizontal_binning_Bronselaer(soccom_linear_depthbin, soccom_nearest_depthbin, soccom_cubic_depthbin, soccom_quad_depthbin):
    """
    Creates a climatology of horizontally gridded (as per Bronselaer et al 2020) mean tracer values from a depth 
    binned profile product (linear, nearest, cubic and quadratic interpolation).
    return SOCCOM_linear_grid, SOCCOM_nearest_grid, SOCCOM_cubic_grid, SOCCOM_quad_grid, SOCCOM_profile_count
    Ruth Moorman, Princeton University, Mar 2020
    """
    
    # pull out the indices of profiles taken in each month
    SOCCOM_Jan_index = soccom_linear_depthbin.groupby('time.month').groups[1]
    SOCCOM_Feb_index = soccom_linear_depthbin.groupby('time.month').groups[2]
    SOCCOM_Mar_index = soccom_linear_depthbin.groupby('time.month').groups[3]
    SOCCOM_Apr_index = soccom_linear_depthbin.groupby('time.month').groups[4]
    SOCCOM_May_index = soccom_linear_depthbin.groupby('time.month').groups[5]
    SOCCOM_Jun_index = soccom_linear_depthbin.groupby('time.month').groups[6]
    SOCCOM_Jul_index = soccom_linear_depthbin.groupby('time.month').groups[7]
    SOCCOM_Aug_index = soccom_linear_depthbin.groupby('time.month').groups[8]
    SOCCOM_Sep_index = soccom_linear_depthbin.groupby('time.month').groups[9]
    SOCCOM_Oct_index = soccom_linear_depthbin.groupby('time.month').groups[10]
    SOCCOM_Nov_index = soccom_linear_depthbin.groupby('time.month').groups[11]
    SOCCOM_Dec_index = soccom_linear_depthbin.groupby('time.month').groups[12]
    
    # initialise empty arrays
    lon_centers = np.arange(4,360,8)
    lon_bounds = np.arange(0,361,8)
    lat_centers = np.arange(-88.5,-30,3)
    lat_bounds = np.arange(-90,-29,3)
    depth_bin_center = np.array([4.,7.,12.,19.,26.,35.,47.,65.,91.,126.,179.,249.,346.,468.,614.,772.,947.,1123.,1298.,1474.,1649.,1825.,2000.])
    depth_bin_bounds = np.array([2.5,5.5,8.5,15.5,22.5,29.5,40.5,53.5,76.5,105.5,146.5,211.5,286.5,405.5,530.5,697.5,846.5,1047.5,1198.5,1397.5,1550.5,1747.5,1902.5,2097.5,])

    SOCCOM_linear_grid = xr.DataArray(np.empty((12,20,45,23)),coords = [np.arange(1,13),lat_centers, lon_centers, depth_bin_center],dims = ['month','latitude','longitude','depth'])
    SOCCOM_nearest_grid = xr.DataArray(np.empty((12,20,45,23)),coords = [np.arange(1,13),lat_centers, lon_centers, depth_bin_center],dims = ['month','latitude','longitude','depth'])
    SOCCOM_cubic_grid = xr.DataArray(np.empty((12,20,45,23)),coords = [np.arange(1,13),lat_centers, lon_centers, depth_bin_center],dims = ['month','latitude','longitude','depth'])
    SOCCOM_quad_grid = xr.DataArray(np.empty((12,20,45,23)),coords = [np.arange(1,13),lat_centers, lon_centers, depth_bin_center],dims = ['month','latitude','longitude','depth'])
    SOCCOM_profile_count = xr.DataArray(np.empty((12,20,45)),coords = [np.arange(1,13), lat_centers, lon_centers],dims = ['month','latitude','longitude'])
    SOCCOM_linear_grid[:,:,:,:] = np.nan
    SOCCOM_nearest_grid[:,:,:,:] = np.nan
    SOCCOM_cubic_grid[:,:,:,:] = np.nan
    SOCCOM_quad_grid[:,:,:,:] = np.nan
    SOCCOM_profile_count[:,:,:] = np.nan
    
    for j in range(12):
        if j == 0:
            month_index = SOCCOM_Jan_index
        elif j == 1:
            month_index = SOCCOM_Feb_index
        elif j == 2:
            month_index = SOCCOM_Mar_index
        elif j == 3:
            month_index = SOCCOM_Apr_index
        elif j == 4:
            month_index = SOCCOM_May_index    
        elif j == 5:
            month_index = SOCCOM_Jun_index
        elif j == 6:
            month_index = SOCCOM_Jul_index
        elif j == 7:
            month_index = SOCCOM_Aug_index
        elif j == 8:
            month_index = SOCCOM_Sep_index
        elif j == 9:
            month_index = SOCCOM_Oct_index    
        elif j == 10:
            month_index = SOCCOM_Nov_index
        elif j == 11:
            month_index = SOCCOM_Dec_index

        linear_month = soccom_linear_depthbin[month_index]
        nearest_month = soccom_nearest_depthbin[month_index]
        cubic_month = soccom_cubic_depthbin[month_index]
        quad_month = soccom_quad_depthbin[month_index]
    
        for i in range(20): 
            lat_S = lat_bounds[i]
            lat_N = lat_bounds[i+1]

            for m in range(45):
                lon_E = lon_bounds[m]
                lon_W = lon_bounds[m+1]  

                # grab out all floats in this bin (linear interpolated)
                linear = linear_month.where(linear_month.lon>lon_E, drop = True)
                linear = linear.where(linear.lon<lon_W, drop = True)
                linear = linear.where(linear.lat<lat_N, drop = True)
                linear = linear.where(linear.lat>lat_S, drop = True)
                linear_mean = linear.mean(dim = 'N_PROF')

                # grab out all floats in this bin (linear interpolated)
                nearest = nearest_month.where(nearest_month.lon>lon_E, drop = True)
                nearest = nearest.where(nearest.lon<lon_W, drop = True)
                nearest = nearest.where(nearest.lat<lat_N, drop = True)
                nearest = nearest.where(nearest.lat>lat_S, drop = True)
                nearest_mean = nearest.mean(dim = 'N_PROF')

                # grab out all floats in this bin (linear interpolated)
                cubic = cubic_month.where(cubic_month.lon>lon_E, drop = True)
                cubic = cubic.where(cubic.lon<lon_W, drop = True)
                cubic = cubic.where(cubic.lat<lat_N, drop = True)
                cubic = cubic.where(cubic.lat>lat_S, drop = True)
                cubic_mean = cubic.mean(dim = 'N_PROF')

                # grab out all floats in this bin (linear interpolated)
                quad = quad_month.where(quad_month.lon>lon_E, drop = True)
                quad = quad.where(quad.lon<lon_W, drop = True)
                quad = quad.where(quad.lat<lat_N, drop = True)
                quad = quad.where(quad.lat>lat_S, drop = True)
                quad_mean = quad.mean(dim = 'N_PROF')


                if len(linear.N_PROF) != 0:
                    SOCCOM_linear_grid[j,i,m,:] = linear_mean
                    SOCCOM_nearest_grid[j,i,m,:] = nearest_mean
                    SOCCOM_cubic_grid[j,i,m,:] = cubic_mean
                    SOCCOM_quad_grid[j,i,m,:] = quad_mean
                    SOCCOM_profile_count[j,i,m] = len(linear.N_PROF)
        print(j)        
    return SOCCOM_linear_grid, SOCCOM_nearest_grid, SOCCOM_cubic_grid, SOCCOM_quad_grid, SOCCOM_profile_count

def interpolate_GLODAP_profiles(variable):
    """
    Aggregates all GLODAP profile data from GLODAPv2.2019_Southern_Ocean_30S.nc into a single xarray datastructure with dimensions N_PROF and depth.
    depth_bin dimension is determined interpolating SOCCOM float data onto a 2001 level 1m grid [0,1,...,2000] using 4 interpolation methods: linear, nearest neighbour, cubic spline and quadratic spline (so later uncertainty associated with the interpolation method may be quantified)
    
    variable is a string that is set to one of the following:
    
    Temperature
    Salinity
    Sigma_theta
    Oxygen
    Nitrate
    DIC
    
    will return 4 DataArray structures associated with the 4 interpolation methods.
    return tracer_linear, tracer_nearest_neighbour, tracer_cubic, tracer_quadratic
    
    Ruth Moorman, Princeton University, Mar 2020
    """
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc')
    so_glodap = so_glodap.Southern_Ocean_GLODAP
    
    cruise = so_glodap.cruise.values
    station = so_glodap.station.values
    time = so_glodap.station.time.astype('float').values
    profile_identifier = np.stack((cruise, station, time))
    profile_index = np.unique(profile_identifier, axis = 1, return_index = True)[1]
    profile_index = np.sort(profile_index)
    print('Number of GLODAP profiles in the Southern Ocean south of 30S (all time):',len(profile_index))
    profile_index_extended = np.append(profile_index, 278326) #add index of last sample
    
    if variable == 'Temperature':
        so_glodap_var = so_glodap.sel(VARIABLE = 'temperature')
    elif variable == 'Salinity':
        so_glodap_var = so_glodap.sel(VARIABLE = 'salinity')
    elif variable == 'Sigma_theta':
        so_glodap_var = so_glodap.sel(VARIABLE = 'sigma0')
    elif variable == 'Oxygen':
        so_glodap_var = so_glodap.sel(VARIABLE = 'oxygen')
    elif variable == 'Nitrate':
        so_glodap_var = so_glodap.sel(VARIABLE = 'nitrate')
    elif variable == 'DIC':
        so_glodap_var = so_glodap.sel(VARIABLE = 'tco2')
    else:
        return print('Use valid variable option, help(interpolate_GLODAP_profiles) for options')
    

    ## create an empty xarray with dimensions N_PROF, depth
    interp_depth = np.linspace(2000,0,2001)
    glodap_var_linear = xr.DataArray(np.empty((len(profile_index),len(interp_depth))), coords = [np.arange(len(profile_index)), interp_depth], dims = ['N_PROF', 'depth'])
    glodap_var_cubic = xr.DataArray(np.empty((len(profile_index),len(interp_depth))), coords = [np.arange(len(profile_index)), interp_depth], dims = ['N_PROF', 'depth'])
    glodap_var_quad = xr.DataArray(np.empty((len(profile_index),len(interp_depth))), coords = [np.arange(len(profile_index)), interp_depth], dims = ['N_PROF', 'depth'])
    glodap_var_nearest = xr.DataArray(np.empty((len(profile_index),len(interp_depth))), coords = [np.arange(len(profile_index)), interp_depth], dims = ['N_PROF', 'depth'])
    glodap_var_linear[:,:] = np.nan
    glodap_var_cubic[:,:] = np.nan
    glodap_var_quad[:,:] = np.nan
    glodap_var_nearest[:,:] = np.nan
    ## alsowant to reatin the following data
    glodap_time = xr.DataArray(np.empty((len(profile_index)),dtype = 'datetime64'), coords = [np.arange(len(profile_index))], dims = 'N_PROF')
    glodap_lon = xr.DataArray(np.empty((len(profile_index))), coords = [np.arange(len(profile_index))], dims = 'N_PROF')
    glodap_lat = xr.DataArray(np.empty((len(profile_index))), coords = [np.arange(len(profile_index))], dims = 'N_PROF')
    glodap_bottomdepth = xr.DataArray(np.empty((len(profile_index))), coords = [np.arange(len(profile_index))], dims = 'N_PROF')
    glodap_maxsampledepth = xr.DataArray(np.empty((len(profile_index))), coords = [np.arange(len(profile_index))], dims = 'N_PROF')
    glodap_time[:] = np.nan
    glodap_lon[:] = np.nan
    glodap_lat[:] = np.nan
    glodap_bottomdepth[:] = np.nan
    glodap_maxsampledepth[:] = np.nan


    count=0
    dropcount=0
    for i in range(len(profile_index)):
        ## extract desirable info
        prof_var = so_glodap_var[profile_index[i]:profile_index_extended[i+1]].values
        prof_depth = so_glodap.depth[profile_index[i]:profile_index_extended[i+1]].values

        prof_xarray = xr.DataArray(prof_var, coords = [prof_depth], dims = 'depth')
        if len(np.unique(prof_depth)) != len(prof_depth):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)

        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True ) # remove nans (necessary for interpolation)

        if len(prof_xarray.depth) > 3:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
            prof_var_cubic = prof_xarray.interp(depth = interp_depth, method = 'cubic')
            prof_var_nearest = prof_xarray.interp(depth = interp_depth, method = 'nearest')
            prof_var_quad = prof_xarray.interp(depth = interp_depth, method = 'quadratic')
            glodap_time[i] = so_glodap.time[profile_index[i]]
            glodap_lat[i] = so_glodap.latitude[profile_index[i]]
            glodap_lon[i] = so_glodap.longitude[profile_index[i]]
            glodap_bottomdepth[i] = so_glodap.bottomdepth[profile_index[i]]
            glodap_maxsampledepth[i] = so_glodap.maxsampledepth[profile_index[i]]
            count=count+1
        else:
            dropcount = dropcount+1
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
            prof_var_cubic = fill
            prof_var_nearest = fill
            prof_var_quad = fill
            glodap_time[i] = np.nan
            glodap_lat[i] = np.nan
            glodap_lon[i] = np.nan
            glodap_bottomdepth[i] = np.nan
            glodap_maxsampledepth[i] = np.nan

        glodap_var_linear[i,:] = prof_var_linear
        glodap_var_cubic[i,:] = prof_var_cubic
        glodap_var_quad[i,:] = prof_var_quad
        glodap_var_nearest[i,:] = prof_var_nearest

    print('dropped %i profiles (too few data for spline interpolation, <3 points)'%dropcount)
    print('retained %i profiles'%count)

    # the SOCCOM data have a different longitude dimension (0,360) and opposed to (-180,180), here I adjust GLODAP to match
    empty = np.empty((len(profile_index),3))
    empty[:,:] = np.nan
    empty[:,0] = 360 + glodap_lon.where(glodap_lon<0).values
    empty[:,1] = glodap_lon.where(glodap_lon>0).values
    empty[:,2] = 999 + glodap_lon.where(glodap_lon==0).values
    a = np.nansum(empty,1)
    b = np.where(a == 0, np.nan, a)
    adjusted_lon = np.where(b == 999, 0, b)
    adjusted_lon = xr.DataArray(adjusted_lon, coords = [np.arange(len(profile_index))], dims = 'N_PROF')
    glodap_var_linear = xr.DataArray(glodap_var_linear, coords = {'N_PROF':np.arange(len(profile_index)), 'depth':interp_depth, 'longitude':adjusted_lon, 'latitude':glodap_lat, 'time':glodap_time, 'bottomdepth':glodap_bottomdepth, 'maxsampledepth':glodap_maxsampledepth})
    glodap_var_nearest = xr.DataArray(glodap_var_nearest, coords = {'N_PROF':np.arange(len(profile_index)), 'depth':interp_depth, 'longitude':adjusted_lon, 'latitude':glodap_lat, 'time':glodap_time, 'bottomdepth':glodap_bottomdepth, 'maxsampledepth':glodap_maxsampledepth})
    glodap_var_cubic = xr.DataArray(glodap_var_cubic, coords = {'N_PROF':np.arange(len(profile_index)), 'depth':interp_depth, 'longitude':adjusted_lon, 'latitude':glodap_lat, 'time':glodap_time, 'bottomdepth':glodap_bottomdepth, 'maxsampledepth':glodap_maxsampledepth})
    glodap_var_quad = xr.DataArray(glodap_var_quad, coords = {'N_PROF':np.arange(len(profile_index)), 'depth':interp_depth, 'longitude':adjusted_lon, 'latitude':glodap_lat, 'time':glodap_time, 'bottomdepth':glodap_bottomdepth, 'maxsampledepth':glodap_maxsampledepth})
    
    return glodap_var_linear, glodap_var_nearest, glodap_var_cubic, glodap_var_quad

def GLODAP_horizontal_binning_Bronselaer(GLODAP_linear_depth_binned, GLODAP_nearest_depth_binned, GLODAP_cubic_depth_binned, GLODAP_quad_depth_binned):
    """
    Creates a gridded product (as per Bronselaer et al 2020) of mean tracer values from a depth 
    binned profile product (linear, nearest, cubic and quadratic interpolation).
    Product will be summer biased.
    return a dataset (ds) of all computed outputs (includes metadata)
    Ruth Moorman, Princeton University, Mar 2020
    """
    
    lon_centers = np.arange(4,360,8)
    lon_bounds = np.arange(0,361,8)
    lat_centers = np.arange(-88.5,-30,3)
    lat_bounds = np.arange(-90,-29,3)
    depth_bin_center = np.array([4.,7.,12.,19.,26.,35.,47.,65.,91.,126.,179.,249.,346.,468.,614.,772.,947.,1123.,1298.,1474.,1649.,1825.,2000.])
    depth_bin_bounds = np.array([2.5,5.5,8.5,15.5,22.5,29.5,40.5,53.5,76.5,105.5,146.5,211.5,286.5,405.5,530.5,697.5,846.5,1047.5,1198.5,1397.5,1550.5,1747.5,1902.5,2097.5,])

    ## initialise empty
    GLODAP_linear_grid = xr.DataArray(np.empty((20,45,23)),coords = [lat_centers, lon_centers, depth_bin_center],dims = ['latitude','longitude','depth'])
    GLODAP_nearest_grid = xr.DataArray(np.empty((20,45,23)),coords = [lat_centers, lon_centers, depth_bin_center],dims = ['latitude','longitude','depth'])
    GLODAP_cubic_grid = xr.DataArray(np.empty((20,45,23)),coords = [lat_centers, lon_centers, depth_bin_center],dims = ['latitude','longitude','depth'])
    GLODAP_quad_grid = xr.DataArray(np.empty((20,45,23)),coords = [lat_centers, lon_centers, depth_bin_center],dims = ['latitude','longitude','depth'])
    GLODAP_linear_grid[:,:,:] = np.nan
    GLODAP_nearest_grid[:,:,:] = np.nan
    GLODAP_cubic_grid[:,:,:] = np.nan
    GLODAP_quad_grid[:,:,:] = np.nan
    GLODAP_full_count = xr.DataArray(np.empty((20,45)),coords = [lat_centers, lon_centers],dims = ['latitude','longitude'])
    GLODAP_DJF_count = xr.DataArray(np.empty((20,45)),coords = [lat_centers, lon_centers],dims = ['latitude','longitude'])
    GLODAP_JJA_count = xr.DataArray(np.empty((20,45)),coords = [lat_centers, lon_centers],dims = ['latitude','longitude'])
    GLODAP_MAM_count = xr.DataArray(np.empty((20,45)),coords = [lat_centers, lon_centers],dims = ['latitude','longitude'])
    GLODAP_SON_count = xr.DataArray(np.empty((20,45)),coords = [lat_centers, lon_centers],dims = ['latitude','longitude'])
    GLODAP_full_count[:,:] = np.nan
    GLODAP_DJF_count[:,:] = np.nan
    GLODAP_JJA_count[:,:] = np.nan
    GLODAP_MAM_count[:,:] = np.nan
    GLODAP_SON_count[:,:] = np.nan
    
    for i in range(20): # no data for first few lat indices
        lat_S = lat_bounds[i]
        lat_N = lat_bounds[i+1]

        for m in range(45):
            lon_E = lon_bounds[m]
            lon_W = lon_bounds[m+1]  

            # grab out all floats in this bin (linear interpolated)
            linear = GLODAP_linear_depth_binned.where(GLODAP_linear_depth_binned.longitude>lon_E, drop = True)
            linear = linear.where(linear.longitude<lon_W, drop = True)
            linear = linear.where(linear.latitude<lat_N, drop = True)
            linear = linear.where(linear.latitude>lat_S, drop = True)
            linear_mean = linear.mean(dim = 'N_PROF')

            # grab out all floats in this bin (linear interpolated)
            nearest = GLODAP_nearest_depth_binned.where(GLODAP_nearest_depth_binned.longitude>lon_E, drop = True)
            nearest = nearest.where(nearest.longitude<lon_W, drop = True)
            nearest = nearest.where(nearest.latitude<lat_N, drop = True)
            nearest = nearest.where(nearest.latitude>lat_S, drop = True)
            nearest_mean = nearest.mean(dim = 'N_PROF')

            # grab out all floats in this bin (linear interpolated)
            cubic = GLODAP_cubic_depth_binned.where(GLODAP_cubic_depth_binned.longitude>lon_E, drop = True)
            cubic = cubic.where(cubic.longitude<lon_W, drop = True)
            cubic = cubic.where(cubic.latitude<lat_N, drop = True)
            cubic = cubic.where(cubic.latitude>lat_S, drop = True)
            cubic_mean = cubic.mean(dim = 'N_PROF')

            # grab out all floats in this bin (linear interpolated)
            quad = GLODAP_quad_depth_binned.where(GLODAP_quad_depth_binned.longitude>lon_E, drop = True)
            quad = quad.where(quad.longitude<lon_W, drop = True)
            quad = quad.where(quad.latitude<lat_N, drop = True)
            quad = quad.where(quad.latitude>lat_S, drop = True)
            quad_mean = quad.mean(dim = 'N_PROF')


            if len(linear.N_PROF) != 0:
                GLODAP_linear_grid[i,m,:] = linear_mean
                GLODAP_nearest_grid[i,m,:] = nearest_mean
                GLODAP_cubic_grid[i,m,:] = cubic_mean
                GLODAP_quad_grid[i,m,:] = quad_mean

                GLODAP_full_count[i,m] = len(linear.N_PROF)
                season_count = linear.groupby('time.season').count().max(dim = 'depth_bins')
    #             print(season_count)
                season = season_count.season
                if 'DJF' in season:
                    GLODAP_DJF_count[i,m] = season_count.sel(season = 'DJF').values
                if 'JJA' in season:
                    GLODAP_JJA_count[i,m] = season_count.sel(season = 'JJA').values
                if 'MAM' in season:
                    GLODAP_MAM_count[i,m] = season_count.sel(season = 'MAM').values
                if 'SON' in season:
                    GLODAP_SON_count[i,m] = season_count.sel(season = 'SON').values
    ds = xr.Dataset({'linear':GLODAP_linear_grid, 'nearest': GLODAP_nearest_grid, 'cubic':GLODAP_cubic_grid, 'quad': GLODAP_quad_grid, 'number_all':GLODAP_full_count, 'number_DJF':GLODAP_DJF_count, 'number_SON':GLODAP_SON_count, 'number_JJA':GLODAP_JJA_count, 'number_MAM':GLODAP_MAM_count})
    return ds