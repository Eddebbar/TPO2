#!/usr/bin/env python
# coding: utf-8

# # Oxygen Budget Analysis 

# This notebook calculates the O$_2$ budget from POP output using xarray and xgcm. Dissolved oxygen in the ocean's interior is simulated following:
# 
# $$
# \frac{\partial{O_2}}{\partial{t}}= \underbrace{- \frac{\partial{U.O_2}}{\partial{x}} -\frac{\partial{V.O_2}}{\partial{y}}}_\text{Lateral Advection}
# - \overbrace{\frac{\partial{W.O_2}}{\partial{z}}}^\text{Vertical Advection}
# + \underbrace{D(O_2)}_\text{Lateral Mixing}
# +\overbrace{\frac{\partial{}}{\partial{z}}k.\frac{\partial{O_2}}{\partial{z}}}^\text{Vertical Mixing}
# + \underbrace{ J(O_2)  }_\text{Sources - Sinks}
# $$
# 

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter("ignore") # Silence warnings

import xarray as xr
import numpy as np
from tqdm import tqdm
import xgcm 
import pop_tools

from utils import *


# In[70]:


C=CLSTR(1,59,300)


# In[83]:


C


# 
# 
# - #### Compute Budget Terms using xarray/roll only

# In[84]:


def pop_budget_roll(ds):
    '''function scales and derives budget terms from POP ouputs using simple xarray.roll operations '''
    di=xr.Dataset()  
    
    # Advective terms
    di['UE_O2'] = -((ds.UE_O2*ds.VOL) - (ds.UE_O2*ds.VOL).roll(nlon=1, roll_coords=True))
    di.UE_O2.attrs['units'] = 'nmol/s'
    di['VN_O2'] = -((ds.VN_O2*ds.VOL) - (ds.VN_O2*ds.VOL).roll(nlat=1, roll_coords=True))
    di.VN_O2.attrs['units'] = 'nmol/s'
    di['WT_O2'] = - (ds.WT_O2*(ds.VOL.drop('z_t').rename({"z_t":"z_w_top"}).assign_coords(z_w_top=ds.z_w_top))
                     - (ds.WT_O2*(ds.VOL.drop('z_t').rename({"z_t":"z_w_top"}).assign_coords(z_w_top=ds.z_w_top))).shift(z_w_top=-1).fillna(0)
                    ).drop('z_w_top').rename({"z_w_top":"z_t"}).assign_coords(z_t=ds.z_t)
    di.WT_O2.attrs['units'] = 'nmol/s'
    di['DIV'] = di.UE_O2+di.VN_O2+di.WT_O2
    di.DIV.attrs['units'] = 'nmol/s'

    # Vertical Mixing:
    di['DIA_IMPVF_O2']=  ((ds.DIA_IMPVF_O2*ds.TAREA).shift(z_w_bot=1).fillna(0) - ds.DIA_IMPVF_O2*ds.TAREA).drop('z_w_bot').rename({"z_w_bot":"z_t"})
    di['DIA_IMPVF_O2']= di['DIA_IMPVF_O2'].load()
    # Air-sea flux added in surface diffusive flux in upper most cell
    di['DIA_IMPVF_O2'][:,0,:,:]=(ds.STF_O2*ds.TAREA - ds.DIA_IMPVF_O2.isel(z_w_bot=0)*ds.TAREA)
    di.DIA_IMPVF_O2.attrs['units'] = 'nmol/s'
    di['KPP_SRC_O2']=ds.KPP_SRC_O2*ds.VOL
    di.KPP_SRC_O2.attrs['units'] = 'nmol/s'
    di['VDIF']=di['DIA_IMPVF_O2']+di['KPP_SRC_O2']

    # Lateral Diffusion (not available for 0004 hindcast 1960-1990 runs)
    if 'HDIFE_O2' in ds.variables:
        di['HDIFE_O2'] = ((ds.HDIFE_O2*ds.VOL) - (ds.HDIFE_O2*ds.VOL).roll(nlon=1, roll_coords=True))
        di.HDIFE_O2.attrs['units'] = 'nmol/s'
        di['HDIFN_O2'] = ((ds.HDIFN_O2*ds.VOL) - (ds.HDIFN_O2*ds.VOL).roll(nlat=1, roll_coords=True))
        di.HDIFN_O2.attrs['units'] = 'nmol/s'
        di['HDIFB_O2'] = ((ds.HDIFB_O2.drop('z_w_bot').rename({"z_w_bot":"z_t"})*ds.VOL).shift(z_t=1).fillna(0) - ds.HDIFB_O2.drop('z_w_bot').rename({"z_w_bot":"z_t"})*ds.VOL)
        di["HDIFB_O2"][:, 0, :, :] = -ds.HDIFB_O2.isel(z_w_bot=0)
        di.HDIFB_O2.attrs['units'] = 'nmol/s'
        di['HDIF'] =di['HDIFE_O2']+di['HDIFN_O2']+di['HDIFB_O2']
        di.HDIF.attrs['units'] = 'nmol/s'

    # Sources - SInk
    if 'J_O2' in ds.variables:
        di['J_O2'] =ds.J_O2*ds.VOL 
    else:
        di['J_O2'] =(ds.O2_PRODUCTION-ds.O2_CONSUMPTION)*ds.VOL
    di.J_O2.attrs['units'] = 'nmol/s'
    
    # Tendency
    if 'TEND_O2' in ds.variables:
        di['TEND_O2'] =ds.TEND_O2*ds.VOL 
    else:
        di['TEND_O2']=di.DIV+di.VDIF+di.HDIF+di.J_O2   
    di.TEND_O2.attrs['units'] = 'nmol/s'

    return di


# In[85]:


def VOLM(ds):        
    VOL = (ds.dz * ds.DXT* ds.DYT).compute()
    ds['VOL']=VOL.compute()
    ds.VOL.attrs['long_name'] = 'volume of T cells'
    ds.VOL.attrs['units'] = 'centimeter^3'
    ds.VOL.attrs['grid_loc'] = '3111'
    return ds


# In[86]:


month_str=1
month_end=13
yr_str=1
yr_end=6

for j in tqdm(np.arange(yr_str,yr_end)):
    for i in tqdm(np.arange(month_str,month_end)):
        print('loading')
        path=f'/glade/scratch/yeddebba/Mesoscale/LR/hist/g.e11.G.T62_g16.eco_x1_analog.001.pop.h.000'+str(j)+'-'+str(i).zfill(2)+'*.nc'
        ds = xr.open_mfdataset(path, parallel=True, coords="minimal", data_vars="minimal", compat='override') 
        keep_vars = ['time_bound', 'time', 
             'TAREA', 'REGION_MASK','dz',
             'z_t','TLAT', 'TLONG','DXT','DYT','KMT',
            'UE_O2','VN_O2','WT_O2',
            'DIA_IMPVF_O2','KPP_SRC_O2','J_O2',
             'HDIFE_O2','HDIFN_O2','HDIFB_O2','STF_O2']
        ds = ds[keep_vars].squeeze()
        print('Calculating Cell Volume')
        ds=VOLM(ds)
        print('Calculating Budget terms')
        di=pop_budget_roll(ds)
        print('Saving')
        di.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/O2_budget_000'+str(j)+'_'+str(i).zfill(2)+'.nc')
        print('Month '+str(i).zfill(2)+' is done')


# In[ ]:


def adv_lr(ds):
    ''' function scales and derives eddy vs mean advective budget terms from POP ouputs using xarray.roll operations '''
    di=xr.Dataset()
    
    # mean Advective terms
    U_O2=(0.5 * (ds.UVEL + ds.UVEL.roll(nlat=1))*ds.O2)
    V_O2=(0.5 * (ds.VVEL + ds.VVEL.roll(nlon=1))*ds.O2)
    W_O2=(ds.WVEL*ds.O2.drop('z_t').rename({"z_t":"z_w_top"}).assign_coords(z_w_top=ds.z_w_top))

    di['U_O2'] = -((U_O2*ds.VOL) - (U_O2*ds.VOL).roll(nlon=1, roll_coords=True))/ds.DXT
    di.U_O2.attrs['units'] = 'nmol/s'
    di['V_O2'] = -((V_O2*ds.VOL) - (V_O2*ds.VOL).roll(nlat=1, roll_coords=True))/ds.DYT
    di.V_O2.attrs['units'] = 'nmol/s'
    di['W_O2'] = - (W_O2*(ds.VOL.drop('z_t').rename({"z_t":"z_w_top"}).assign_coords(z_w_top=ds.z_w_top))
                     - (W_O2*(ds.VOL.drop('z_t').rename({"z_t":"z_w_top"}).assign_coords(z_w_top=ds.z_w_top))).shift(z_w_top=-1).fillna(0)
                    ).drop('z_w_top').rename({"z_w_top":"z_t"}).assign_coords(z_t=ds.z_t)/ds.dz
    di.W_O2.attrs['units'] = 'nmol/s'
    di['DIVm'] = di.U_O2+di.V_O2+di.W_O2
    di.DIVm.attrs['units'] = 'nmol/s'
    
    return di


# In[ ]:


dsh=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/O2_budget_000?_??.nc')
dsh=volume_LR(dsh)
dsh

dg = xr.open_mfdataset('/glade/work/yeddebba/grids/POP_gx1v6.nc')

dss=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/3D/STF_O2/STF_O2.nc', parallel=True,  data_vars="minimal", coords="minimal", compat='override').groupby('time.month').mean(dim='time')
dss['STF_O2']= (dss.STF_O2*dg.TAREA)
dss

dshm=dsh.groupby('time.month').mean(dim='time')
dshm

dg = xr.open_mfdataset('/glade/work/yeddebba/grids/POP_gx1v6.nc').isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))

dv=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/LR/3D/UVEL/UVEL.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/3D/VVEL/VVEL.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/3D/WVEL/WVEL.nc'}).isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))
dc = xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/3D/O2/O2.nc').isel(z_t=np.arange(0,41))
dh=xr.Dataset()
dh=xr.merge([dc,dv,dg],compat='override')
dh=volume_LR(dh)
dh

dg = xr.open_mfdataset('/glade/work/yeddebba/grids/POP_gx1v6.nc')#.isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))

dv=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/3D/[UVW]VEL/[UVW]VEL?.nc')#.isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))
dc = xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/3D/O2/O2?.nc')#.isel(z_t=np.arange(0,41))
dh=xr.Dataset()
dh=xr.merge([dc,dv,dg],compat='override')
dh=volume_LR(dh)
dh


# In[ ]:


dhm=dh.groupby('time.month').mean(dim='time')
dhm

dhma=adv_lr(dhm)
dhma


# In[ ]:


glade/scratch/yeddebba/Mesoscale/LR/dshm.TEND_O2.to_netcdf('final_budget/TEND_Mon_Mean.nc')()

dshm.DIV.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/DIV_Mon_Mean.nc')

dshm.VDIF.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/VDIF_Mon_Mean.nc')

dshm.HDIF.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/HDIF_Mon_Mean.nc')

dshm.HDIFB_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/HDIFB_Mon_Mean.nc')

dshm.J_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/J_O2_Mon_Mean.nc')

dshm.KPP_SRC_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/KPP_Mon_Mean.nc')

dshm.DIA_IMPVF_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/DIA_Mon_Mean.nc')

dshm.UE_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/UE_Mon_Mean.nc')

dshm.VN_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/VN_Mon_Mean.nc')

dshm.WT_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/WT_Mon_Mean.nc')

dss.STF_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/STF_O2_Mon_Mean.nc')

dhma.U_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/U_O2_Mon_Mean.nc')
dhma.V_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/V_O2_Mon_Mean.nc')
dhma.W_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/W_O2_Mon_Mean.nc')
ds=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/?_O2_Mon_Mean.nc')
ds['DIVm'] = ds.U_O2+ds.V_O2+ds.W_O2
ds.DIVm.attrs['units'] = 'nmol/s'
ds.DIVm.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/DIVm_Mon_Mean.nc')

#                       '/glade/scratch/yeddebba/Mesoscale/HR/CLM/O2_Mon_Mean.nc',
#                       '/glade/scratch/yeddebba/Mesoscale/HR/CLM/UVEL_Mon_Mean.nc',
dv=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/3D/UVEL/UVEL[12345].nc')#.isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))

dvh=dv.groupby('time.month').mean(dim='time')
dvh

dvh.UVEL.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/UVEL_Mon_Mean.nc')

dvh.VVEL.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/VVEL_Mon_Mean.nc')

dvh.WVEL.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/WVEL_Mon_Mean.nc')

dch.O2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/O2_Mean.nc')


# In[ ]:


dq=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/LR/final_budget/TEND_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/DIV_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/VDIF_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/HDIF_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/HDIFB_Mon_Mean.nc',                      
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/J_O2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/UE_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/VN_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/WT_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/STF_O2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/DIVm_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/U_O2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/V_O2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/W_O2_Mon_Mean.nc',
                     })
dq['Up_O2p']=dq.UE_O2-dq.U_O2
dq['Vp_O2p']=dq.VN_O2-dq.V_O2
dq['Wp_O2p']=dq.WT_O2-dq.W_O2

dq['DIVp']=dq.DIV-dq.DIVm

dq=xr.merge([dq,dg],compat='override')

dq=volume_LR(dq)
dq

dsh=dq.mean('month').squeeze()
dsh

dsh.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/O2_Budget_Mean.nc')

dqe=xr.Dataset()
dqe['Up_O2p']=dq.UE_O2-dq.U_O2
dqe['Vp_O2p']=dq.VN_O2-dq.V_O2
dqe['Wp_O2p']=dq.WT_O2-dq.W_O2
dqe['DIVp']=dq.DIV-dq.DIVm
dqe.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/O2_Budget_Eddy_Mon_Mean.nc')
dshe=dqe.mean('month').squeeze()
dshe
dshe.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/O2_Budget_Eddy_Mean.nc')

dq.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/Budegt_Mon_Mean.nc')


# In[ ]:






















# ### Calculates mean advective terms from 5-day mean U and O2 outputs 

# In[2]:


# Functions for calculating advective terms from output U and O2  

def adv(ds):
    ''' function scales and derives full d(u.O2)  advective budget terms from POP ouputs using xarray.roll operations '''
    di=xr.Dataset()
    
    # mean Advective terms
    U_O2=(0.5 * (ds.UVEL + ds.UVEL.roll(nlat=1))*ds.O2)
    V_O2=(0.5 * (ds.VVEL + ds.VVEL.roll(nlon=1))*ds.O2)
    W_O2=ds.WVEL*ds.O2.drop('z_t').rename({"z_t":"z_w_top"}).assign_coords(z_w_top=ds.z_w_top)
     
    di['U_O2'] = -((U_O2*ds.VOL.isel(month=0)) - (U_O2*ds.VOL.isel(month=0)).roll(nlon=1, roll_coords=True))/ds.DXT.isel(month=0)
    di.U_O2.attrs['units'] = 'nmol/s'
    di['V_O2'] = -((V_O2*ds.VOL.isel(month=0)) - (V_O2*ds.VOL.isel(month=0)).roll(nlat=1, roll_coords=True))/ds.DYT.isel(month=0)
    di.V_O2.attrs['units'] = 'nmol/s'
    di['W_O2'] = - (W_O2*(ds.VOL.isel(month=0).drop('z_t').rename({"z_t":"z_w_top"}).assign_coords(z_w_top=ds.z_w_top))
                     - (W_O2*(ds.VOL.isel(month=0).drop('z_t').rename({"z_t":"z_w_top"}).assign_coords(z_w_top=ds.z_w_top))).shift(z_w_top=-1).fillna(0)
                    ).drop('z_w_top').rename({"z_w_top":"z_t"}).assign_coords(z_t=ds.z_t)/ds.dz
    di.W_O2.attrs['units'] = 'nmol/s'
    di['DIVm'] = di.U_O2+di.V_O2+di.W_O2
    di.DIVm.attrs['units'] = 'nmol/s'
    
    return di

def adv_u(ds):
    ''' function scales and derives (u.dO2) advective budget terms for monthly means from POP ouputs using xarray.roll operations '''
    di=xr.Dataset()
    dO2dx = ((ds.O2*ds.VOL.isel(month=0)) - (ds.O2*ds.VOL.isel(month=0)).roll(nlon=1, roll_coords=True))/ds.DXT.isel(month=0)
    di['U_dO2'] = -ds.UVEL*dO2dx
    di.U_dO2.attrs['units'] = 'nmol/s'

    dO2dy = ((ds.O2*ds.VOL.isel(month=0)) - (ds.O2*ds.VOL.isel(month=0)).roll(nlat=1, roll_coords=True))/ds.DYT.isel(month=0)
    di['V_dO2'] = -ds.VVEL*dO2dy
    di.V_dO2.attrs['units'] = 'nmol/s'

    dO2dz = ((ds.O2*ds.VOL.isel(month=0))-(ds.O2*ds.VOL.isel(month=0)).shift(z_t=-1).fillna(0)
            )/ds.dz
    di['W_dO2'] = -(ds.WVEL*dO2dz).drop('z_w_top').rename({"z_w_top":"z_t"}).assign_coords(z_t=ds.z_t)
    di.W_dO2.attrs['units'] = 'nmol/s'

    di['DIVd'] = di.U_dO2+di.V_dO2+di.W_dO2
    di.DIVd.attrs['units'] = 'nmol/s'
    
    return di

def adv_ut(ds):
    ''' function scales and derives (u.dO2) advective budget terms for 5 day mean POP ouputs using xarray.roll operations '''
    di=xr.Dataset()
    dO2dx = ((ds.O2*ds.VOL) - (ds.O2*ds.VOL).roll(nlon=1, roll_coords=True))/ds.DXT
    di['U_dO2'] = -ds.UVEL*dO2dx
    di.U_dO2.attrs['units'] = 'nmol/s'

    dO2dy = ((ds.O2*ds.VOL) - (ds.O2*ds.VOL).roll(nlat=1, roll_coords=True))/ds.DYT
    di['V_dO2'] = -ds.VVEL*dO2dy
    di.V_dO2.attrs['units'] = 'nmol/s'

    dO2dz = (((ds.O2*ds.VOL)-(ds.O2*ds.VOL).shift(z_t=-1).fillna(0)
            )/ds.dz).drop('z_t').rename({"z_t":"z_w_top"}).assign_coords(z_w_top=ds.z_w_top)
    di['W_dO2'] = -(ds.WVEL*dO2dz).drop('z_w_top').rename({"z_w_top":"z_t"}).assign_coords(z_t=ds.z_t)
    di.W_dO2.attrs['units'] = 'nmol/s'

    di['DIVd'] = di.U_dO2+di.V_dO2+di.W_dO2
    di.DIVd.attrs['units'] = 'nmol/s'
    
    return di


# #### Calculate monthly climatological mean budget terms:
# 
# $$
# \overline{\frac{\partial{O_2}}{\partial{t}}}= \underbrace{- \overline{\frac{\partial{U.O_2}}{\partial{x}}} -\overline{\frac{\partial{V.O_2}}{\partial{y}}}}_\text{Lateral Advection}
# - \overbrace{\overline{\frac{\partial{W.O_2}}{\partial{z}}}}^\text{Vertical Advection}
# + \underbrace{\overline{A_h.\nabla^2{O_2}}}_\text{Lateral Mixing}
# +\overbrace{\overline{\frac{\partial{}}{\partial{z}}k.\frac{\partial{O_2}}{\partial{z}}}}^\text{Vertical Mixing}
# + \underbrace{\overline{ J(O_2)}  }_\text{Sources - Sinks}
# $$
# 
# 

# where the mean $\overline{X}$ refers to the monthly climatological mean

# In[11]:


dsh=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/O2_budget_000?_??.nc')
dss=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/budgets/STF_O2.000?-??.nc', parallel=True,  data_vars="minimal", coords="minimal", compat='override').groupby('time.month').mean(dim='time')
dss['STF_O2']= (dss.STF_O2*dh.TAREA)

dshm=dsh.groupby('time.month').mean(dim='time')

dshm.TEND_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/TEND_Mon_Mean.nc')
dshm.DIV.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIV_Mon_Mean.nc')
dshm.UE_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/UE_Mon_Mean.nc')
dshm.VN_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/VN_Mon_Mean.nc')
dshm.WT_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/WT_Mon_Mean.nc')
dshm.VDIF.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/VDIF_Mon_Mean.nc')
dshm.HDIF.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/HDIF_Mon_Mean.nc')
dshm.J_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/J_O2_Mon_Mean.nc')
dss.STF_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/STF_O2_Mon_Mean.nc')
dshm.KPP_SRC_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/KPP_Mon_Mean.nc')
dshm.DIA_IMPVF_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIA_Mon_Mean.nc')


# #### Calculate monthly climatological mean full advective terms:
# 
# $$
# -\nabla.{\overline{U}\overline{O2}}= -\frac{\partial{\overline{U}.\overline{O_2}}}{\partial{x}}-\frac{\partial{\overline{V}.\overline{O_2}}}{\partial{y}}-\frac{\partial{\overline{W}.\overline{O_2}}}{\partial{z}}
# $$
# 
# 

# In[101]:


dg = xr.open_mfdataset('/glade/work/yeddebba/grids/POP_gx1v6.nc').isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))

dv=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/LR/3D/UVEL/UVEL.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/3D/VVEL/VVEL.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/3D/WVEL/WVEL.nc'}).isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))
dc = xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/3D/O2/O2.nc').isel(z_t=np.arange(0,41))
dh=xr.Dataset()
dh=xr.merge([dc,dv,dg],compat='override')
dh=volume_LR(dh)
dh

dg = xr.open_mfdataset('/glade/work/yeddebba/grids/POP_gx1v6.nc')#.isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))

dv=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/3D/[UVW]VEL/[UVW]VEL?.nc')#.isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))
dc = xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/LR/3D/O2/O2?.nc')#.isel(z_t=np.arange(0,41))
dh=xr.Dataset()
dh=xr.merge([dc,dv,dg],compat='override')
dh=volume_LR(dh)
dh

dhm=dh.groupby('time.month').mean(dim='time')
dhma=adv(dhm)

dhm.UVEL.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/UVEL_Mon_Mean.nc')
dhm.VVEL.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/VVEL_Mon_Mean.nc')
dhm.WVEL.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/WVEL_Mon_Mean.nc')
dhm.O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/O2_Mon_Mean.nc')

dhma.DIVm.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/DIVm_Mon_Mean.nc')
dhma.U_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/U_O2_Mon_Mean.nc')
dhma.V_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/V_O2_Mon_Mean.nc')
dhma.W_O2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/W_O2_Mon_Mean.nc')


# ### Calculate Longterm Mean

# In[43]:


dq=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/LR/final_budget/TEND_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/DIV_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/VDIF_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/HDIF_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/HDIFB_Mon_Mean.nc',                      
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/J_O2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/UE_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/VN_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/WT_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/STF_O2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/DIVm_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/U_O2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/V_O2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/LR/final_budget/W_O2_Mon_Mean.nc',
                     })
dq['Up_O2p']=dq.UE_O2-dq.U_O2
dq['Vp_O2p']=dq.VN_O2-dq.V_O2
dq['Wp_O2p']=dq.WT_O2-dq.W_O2

dq['DIVp']=dq.DIV-dq.DIVm

dq=xr.merge([dq,dg],compat='override')

dq=volume_LR(dq)
dq

dsh=dq.mean('month').squeeze()
dsh

dsh.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/O2_Budget_Mean.nc')

dqe=xr.Dataset()
dqe['Up_O2p']=dq.UE_O2-dq.U_O2
dqe['Vp_O2p']=dq.VN_O2-dq.V_O2
dqe['Wp_O2p']=dq.WT_O2-dq.W_O2
dqe['DIVp']=dq.DIV-dq.DIVm
dqe.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/O2_Budget_Eddy_Mon_Mean.nc')
dshe=dqe.mean('month').squeeze()
dshe
dshe.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/final_budget/O2_Budget_Eddy_Mean.nc')

dq.to_netcdf('/glade/scratch/yeddebba/Mesoscale/LR/CLM/Budegt_Mon_Mean.nc')


# #### Calculate Eddy term using eddy decompostion as residual of mean total full advective terms - mean total full advective terms:
# 
# $$
# - \nabla.{\overline{U O_2}}=-\nabla.{\overline{U}\space \overline{O_2}}-\nabla.{\overline{U' O_2'}}
# $$
# so that 
# 
# $$
# -\nabla.{\overline{U' O_2'}}=- \nabla.{\overline{U O_2}}+\nabla.{\overline{U}\space \overline{O_2}}
# $$
# 
# 
# 

# In[ ]:


dqe=xr.Dataset()
dqe['Up_O2p']=dq.UE_O2-dq.U_O2
dqe['Vp_O2p']=dq.VN_O2-dq.V_O2
dqe['Wp_O2p']=dq.WT_O2-dq.W_O2
dqe['DIVp']=dq.DIV-dq.DIVm
dqe.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/O2_Budget_Eddy_Mon_Mean.nc')

dqe.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/O2_Budget_Eddy_Mean.nc')


# ### With mass conservation
# $$
# \nabla{U}= 0
# $$

# ### The mean total advection becomes: 
# $$
# -\nabla{.\overline{U O2}}= -\overline{U.\nabla{O2}} 
# $$

# ### Loop through model output and scale/calculate advective (u.dO2dx, v.dO2dy, w.dO2dz) budget terms

# $$
# u.\nabla{O_2}= \underbrace{- U.\frac{\partial{O_2}}{\partial{x}} -V.\frac{\partial{O_2}}{\partial{y}}}_\text{Lateral Advection}
# - \overbrace{W.\frac{\partial{O_2}}{\partial{z}}}^\text{Vertical Advection}
# $$

# In[ ]:


month_str=1
month_end=13
yr_str=1
yr_end=6

for j in tqdm(np.arange(yr_str,yr_end)):
    for i in tqdm(np.arange(month_str,month_end)):
        print('loading ...')
        path=f'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/O2.000'+str(j)+'-'+str(i).zfill(2)+'.nc'
        dc = xr.open_mfdataset(path, parallel=True, coords="minimal", data_vars="minimal", compat='override')
        path=f'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/?VEL.000'+str(j)+'-'+str(i).zfill(2)+'.nc'
        dv = xr.open_mfdataset(path, parallel=True, coords="minimal", data_vars="minimal", compat='override') 
        dg = xr.open_dataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/POP_GRID_F.nc')
        dh=xr.merge([dc,dv,dg])
        dh=dh.isel(z_w=np.arange(0,41),z_t=np.arange(0,41),z_w_top=np.arange(0,41)) 
        print('Calculating Cell Volume')
        dh=volume(dh)
        print('Calculating Adv Budget terms')
        di=adv_ut(dh)
        print('Saving')
#         di.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_budget_000'+str(j)+'_'+str(i).zfill(2)+'.nc')
        print('Saving u_do2dx')    
        di.U_dO2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_budget_000'+str(j)+'_'+str(i).zfill(2)+'.nc')
        print('Saving v_do2dy')            
        di.V_dO2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_budget_000'+str(j)+'_'+str(i).zfill(2)+'.nc')
        print('Saving w_do2dz')                    
        di.W_dO2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_budget_000'+str(j)+'_'+str(i).zfill(2)+'.nc')
        
        print('Saving DIVd')                    
        dx=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_budget_000'+str(j)+'_'+str(i).zfill(2)+'.nc',
                      '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_budget_000'+str(j)+'_'+str(i).zfill(2)+'.nc',
                      '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_budget_000'+str(j)+'_'+str(i).zfill(2)+'.nc'})
        di['DIVd'] = dx.U_dO2+dx.V_dO2+dx.W_dO2
        di.DIVd.attrs['units'] = 'nmol/s'
        di.DIVd.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVd_budget_000'+str(j)+'_'+str(i).zfill(2)+'.nc')
        print('Month '+str(i).zfill(2)+' is done')



# ### And the eddy flux is calculated as a residual in this reynolds decompsotion:
# $$
# \overline{-U.\nabla{O_2}} = -\overline{U} . \nabla{\overline{O_2}}-\overline{U'.\nabla{O_2'}} 
# $$

# ### So We calculate the mean total advective terms: 
# 
# $$
# \overline{-U.\nabla{O_2}} = -\overline{U.\frac{\partial{O_2}}{\partial{x}}} - \overline{V.\frac{\partial{O_2}}{\partial{y}}}-\overline{W.\frac{\partial{O_2}}{\partial{z}}} 
# $$
# 
# where the mean, $\overline{X}$, refers to the monthly climatological mean

# In[23]:


# Load all 5 day mean advective terms u.dO2 and calculate monthly means and longterm means 
dsh=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_budget_000?_??.nc')
dshm=dsh.groupby('time.month').mean(dim='time')
dshm.U_dO2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2m_Mon_Mean.nc')
dshm.U_dO2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2m_Mean.nc')

dsh=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_budget_000?_??.nc')
dshm=dsh.groupby('time.month').mean(dim='time')
dshm.V_dO2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2m_Mon_Mean.nc')
dshm.V_dO2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2m_Mean.nc')

dsh=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_budget_000?_??.nc')
dshm=dsh.groupby('time.month').mean(dim='time')
dshm.W_dO2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2m_Mon_Mean.nc')
dshm.W_dO2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2m_Mean.nc')

dsh=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVd_budget_000?_??.nc')
dshm=dsh.groupby('time.month').mean(dim='time')
dshm.DIVd.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVdm_Mon_Mean.nc')
dshm.DIVd.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVdm_Mean.nc')


# ### and calculate the mean advective terms: 
# 
# $$
# -\overline{U}.\nabla{\overline{O_2}} = -\overline{U}.\frac{\partial{\overline{O_2}}}{\partial{x}} -\overline{V}.\frac{\partial{\overline{O_2}}}{\partial{y}} -\overline{W}.\frac{\partial{\overline{O_2}}}{\partial{z}}
# $$
# 
# where the mean, $\overline{X}$, refers to the monthly climatological mean

# In[ ]:


dv=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/?VEL.000?-??.nc')
dc = xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/O2.000?-??.nc')
dg = xr.open_dataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/POP_GRID_F.nc')
dh=xr.Dataset()
dh=xr.merge([dc,dv,dg],compat='override')
dh=dh.isel(z_w=np.arange(0,41),z_t=np.arange(0,41), z_w_top=np.arange(0,41))
dh=volume(dh)

ds=dh.groupby('time.month').mean(dim='time')
dsa=adv_u(ds)

dsa.U_dO2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_Mon_Mean.nc')
dsa.V_dO2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_Mon_Mean.nc')
dsa.W_dO2.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_Mon_Mean.nc')


# In[138]:


dr=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_Mon_Mean.nc',})
dr.U_dO2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_Mean.nc')

dr=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_Mon_Mean.nc',})
dr.V_dO2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_Mean.nc')

dr=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_Mon_Mean.nc',})
dr.W_dO2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_Mean.nc')


# In[158]:


dss1=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_Mon_Mean.nc',})
dss2=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_Mon_Mean.nc',})
dss3=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_Mon_Mean.nc',})

dy=xr.Dataset()
dy['DIVdh'] = (dss1.U_dO2+dss2.V_dO2)
dy.DIVdh.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVdh_Mon_Mean.nc')

dss4=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVdh_Mon_Mean.nc',})
dy['DIVd'] =dss4.DIVdh +dss3.W_dO2
dy.DIVd.attrs['units'] = 'nmol/s'
dy.DIVd.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVd_Mon_Mean.nc')


# In[70]:


dr=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_Mon_Mean.nc',
                        '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_Mon_Mean.nc',
                        '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVd_Mon_Mean.nc'})

dr.U_dO2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_Mean.nc')
dr.V_dO2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_Mean.nc') 
dr.W_dO2.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_Mean.nc') 
dr.DIVd.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVd_Mean.nc') 


# ### And the eddy flux is calculated as a residual in this reynolds decompsotion:
# $$
# -\overline{U'.\nabla{O_2'}} = - \overline{U.\nabla{O_2}} + \overline{U} . \nabla{\overline{O_2}}
# $$

# In[71]:


dud=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2_Mon_Mean.nc',
                        '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2_Mon_Mean.nc',
                        '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVd_Mon_Mean.nc'})
dudm=xr.open_mfdataset({'/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/U_dO2m_Mon_Mean.nc',
                        '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/V_dO2m_Mon_Mean.nc',
                        '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/W_dO2m_Mon_Mean.nc',
                      '/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVdm_Mon_Mean.nc'})

dqe=xr.Dataset()
dqe['Up_dO2p']=dudm.U_dO2-dud.U_dO2
dqe['Vp_dO2p']=dudm.V_dO2-dud.V_dO2
dqe['Wp_dO2p']=dudm.W_dO2-dud.W_dO2
dqe['DIVdp']=dudm.DIVd-dud.DIVd

dqe.to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVde_Mon_Mean.nc')
dqe=xr.open_mfdataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVde_Mon_Mean.nc')
dqe.mean('month').to_netcdf('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/final_budget/DIVde_Mean.nc')

