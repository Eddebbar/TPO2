#! /usr/bin/env python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import holoviews as hv
import datashader
from holoviews.operation.datashader import regrid, shade, datashade
import pop_tools
import sys, os
#------------------------------------------------------------------------
#-- Dask Cluster
#------------------------------------------------------------------------
import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client

def CLSTR(N=1,T=15,M=109, P=10):
    nnodes = N; ntime="00:"+str(T)+":00"; nmemory= str(M)+"GB"; nprocesses= P
    cluster = PBSCluster(
        cores=nnodes, # The number of cores you want
        memory=nmemory, # Amount of memory
        processes=nprocesses, # How many processes
        walltime=ntime, # Amount of wall time
        queue='casper', # The type of queue to utilize (/glade/u/apps/dav/opt/usr/bin/execcasper)
        local_directory='/glade/scratch/yeddebba/tmp', # Use your local directory
        resource_spec='select=1:ncpus='+str(P)+':mem='+nmemory, # Specify resources
        project='USIO0015', # Input your project ID here
        interface='ib0', # Interface to use
    )
#     cluster.scale(nnodes*nprocesses)
    cluster.scale(nnodes)
    dask.config.set({'distributed.dashboard.link':'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'})
    client = Client(cluster)
    
    return client
#------------------------------------------------------------------------
#-- plotting functions
#------------------------------------------------------------------------

def savefig(fig, fname):
    fig.savefig(os.path.join(fname+'.png'),dpi=300, bbox_inches="tight")  # comment out to disable saving
#     fig.savefig(os.path.join(fname+'.pdf'),dpi=300, bbox_inches="tight")  # comment out to disable saving
    return

def plot_cmap(ax,lon,lat, var,vmn, vmx, stp, clr,units,title,coor,fs,fsx,lon_lab,lat_lab):
    ax.set_extent(coor,crs=cartopy.crs.PlateCarree())
    pc= ax.contourf(lon,lat,var,np.arange(vmn,vmx,stp),cmap=clr,transform=ccrs.PlateCarree(), extend='both') #, 
    land = ax.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '10m',linewidth=0.1, edgecolor='black', facecolor='grey'))
    cb = plt.colorbar(pc, ax=ax, orientation='vertical', extend='both',pad=0.02)#, aspect=20)  
    cb.ax.set_title(units,fontsize=fs-1)
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=fs-2)
    ax.set_aspect('auto')
    ax.set_title(title,loc='center',fontsize=fs)
    gl=ax.gridlines(draw_labels=True)
    gl.xlabel_style = {'size': fsx, 'color': 'black'}; gl.ylabel_style = {'size': fsx, 'color': 'black'}; 
    gl.xlabels_top = False; gl.ylabels_right= False; gl.xlines = False; gl.ylines = False
    gl.xlocator = mticker.FixedLocator(lon_lab); gl.ylocator = mticker.FixedLocator(lat_lab)
    gl.xformatter = LONGITUDE_FORMATTER ; gl.yformatter = LATITUDE_FORMATTER
    return ax

def plot_cmap_ncb(ax,lon,lat, var,vmn, vmx, stp, clr,units,title,coor,fs,fsx):
    ax.set_extent(coor,crs=cartopy.crs.PlateCarree())
    pc= ax.contourf(lon,lat,var,np.arange(vmn,vmx,stp),cmap=clr,transform=ccrs.PlateCarree(), extend='both') #, 
    land = ax.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '10m',linewidth=0.1, edgecolor='black', facecolor='grey'))
    ax.set_aspect('auto')
    ax.set_title(title,loc='center',fontsize=fs)
    return ax,pc

def plot_depth_section(ax,x,z, var,cntf,lev,clr,units,title,coords,fs,fsx,lw,alp,cb_on,profile):
    if cntf==True: 
        cs= ax.contourf(x,z,var,levels=lev,cmap=clr,extend='both')
        if cb_on==True:
            cb = plt.colorbar(cs, ax=ax, orientation='vertical', extend='both',pad=0.02)#, aspect=20)  
            cb.ax.set_title(units,fontsize=fs-1)
            cb.ax.tick_params(labelsize=fs-2)
            cb.ax.minorticks_off()
    if cntf==False: cs= ax.contour(x,z,var,colors=clr,linewidths=lw,levels=lev,alpha=alp)
    ax.set_title(title,loc='center',fontsize=fs)
    ax.set_ylim(coords[0]); ax.set_xlim(coords[1]); ax.minorticks_on(); 
    ax.set_ylabel('Depth (m)',fontsize=fsx+2); 
    ax.set_aspect('auto')
    if profile=='lon': ax.set_xlabel('Latitude ($^{o}$N)',fontsize=fsx+2)
    if profile=='lat': ax.set_xlabel('Longitude ($^{o}$E)',fontsize=fsx+2)
    ax.set_title(title,pad=0.01,fontsize=fs+2, loc='center');
    ax.tick_params(axis='both', labelsize=fsx)  
    return ax 

def plot_quiver(lon, lat, u, v,scl, lng, alp, ax):
    pcc=ax.quiver(lon, lat, u, v, color='black', scale_units ='x' , scale=lng, alpha=alp)
    ax.quiverkey(pcc, X=0.05, Y=1.03, U=scl, 
#                  label=str(scl/100)+' m.s$^{-1}$', 
                 label='U|'+str(scl/100)+' m.s$^{-1}$', 
#                  label='U|'+str(scl/100)+' m.s$^{-1}$ W|25m.d$^{-1}$', 
                 labelpos='E',fontproperties={'size': 7}) #
    return ax

def plot_hovmollar(ax,x,y,var,cntf,lev,clr,units,title,coords,fs,fsx,lw,alp,cb_on,profile):
    if cntf==True: 
        cs= ax.contourf(x,y,var,levels=lev,cmap=clr,extend='both')
        if cb_on==True:
            cb = plt.colorbar(cs,orientation='vertical', pad=0.02, aspect=20,extend='both')  
            cb.ax.set_title(units,fontsize=fs)
            cb.ax.tick_params(labelsize=fs)
    if cntf==False: cs= ax.contour(x,y,var,colors=clr,linewidths=lw,levels=lev,alpha=alp)
    ax.set_title(title,loc='center',fontsize=fs)
    ax.set_xlim(coords[0]); ax.minorticks_on(); #ax.set_ylim(coords[0]); 
    ax.set_ylabel('Time',fontsize=fs); 
    if profile=='lon': ax.set_xlabel('Latitude $^{o}$N',fontsize=fs)
    if profile=='lat': ax.set_xlabel('Longitude $^{o}$E',fontsize=fs)
    ax.set_title(title,pad=0.01,fontsize=fs, loc='center');
    ax.tick_params(axis='both', labelsize=fs)  
    return ax

def hview(v):
    hv.extension('bokeh', width=100)
    hv_ds = hv.Dataset(v.rename('v'))
    im = hv_ds.to(hv.Image, kdims=[v.dims[-1],v.dims[-2]], dynamic=True)
    
    return im
#=====================================
#======== Coords/Index functions ======
#=====================================

def hypoxic_depth(ds,hyp_level):
    '''returns depth (in m) of hypoxic conditions as defined by user
    '''
    dh=ds.z_t.where(ds.O2<=hyp_level)
    hd=dh.min(dim='z_t')*1e-2
    return hd

def find_indices(xgrid, ygrid, xpoint, ypoint):
    '''returns indices of lat/lon as defined by user --- Function written by Riley Brady
    '''
    dx = xgrid - xpoint
    dy = ygrid - ypoint
    reduced_grid = abs(dx) + abs(dy)
    min_ix = np.nanargmin(reduced_grid)
    i, j = np.unravel_index(min_ix, reduced_grid.shape)
    
    return i, j

def masker(ds,N,S,E,W):
    ''' Masking function, given a dataset, and coordinates
    '''
    dsm= ds.where((ds.TLAT>=S) & (ds.TLAT<=N) & (ds.TLONG>=360-W) & (ds.TLONG<=360-E))
    return dsm

def add_grid(ds):
    ''' Add grid variables to a dataset 
    '''
    dg0 = xr.open_dataset('/glade/scratch/yeddebba/Mesoscale/HR/TPAC/POP_GRID_F.nc')
    ds=xr.merge([ds,dg0])
    return ds

def volume(ds):
    VOL = (ds.DZT * ds.DXT* ds.DYT).compute()
    KMT = ds.KMT.compute()

    for j in range(len(KMT.nlat)):
        for i in range(len(KMT.nlon)):
            k = KMT.values[j,i].astype(int)
            VOL.values[k:,j,i] = 0.

    ds['VOL']=VOL
    ds.VOL.attrs['long_name'] = 'volume of T cells'
    ds.VOL.attrs['units'] = 'centimeter^3'
    ds.VOL.attrs['grid_loc'] = '3111'
    
    return ds
#=====================================================================================
#======== Horizontal Operators on the POP grid  - modified from POP-tools/XGCM =======
#=====================================================================================

metrics = {
    ("X",): ["DXU", "DXT"],  # X distances
    ("Y",): ["DYU", "DYT"],  # Y distances
    ("Z",): ["DZU", "DZT"],  # Z distances
    ("X", "Y"): ["UAREA", "TAREA"],  # areas
}

def relative_vorticity(u,v,grid,dsx):
    rv= (grid.diff(grid.interp(v * dsx.DYU, 'Y', boundary='fill'), 'X', boundary='extend') - grid.diff(grid.interp(u * dsx.DXU, 'X', boundary='fill'), 'Y', boundary='extend') ) / dsx.TAREA
    
    return rv

def okubo_weiss(u,v,grid,dsx):
    ss= (grid.diff(grid.interp(u * dsx.DYU, 'Y', boundary='fill'), 'X', boundary='extend') - grid.diff(grid.interp(v * dsx.DXU, 'X', boundary='fill'), 'Y', boundary='extend') ) / dsx.TAREA
    sn= (grid.diff(grid.interp(v * dsx.DYU, 'Y', boundary='fill'), 'X', boundary='extend') + grid.diff(grid.interp(u * dsx.DXU, 'X', boundary='fill'), 'Y', boundary='extend') ) / dsx.TAREA
    rv= (grid.diff(grid.interp(v * dsx.DYU, 'Y', boundary='fill'), 'X', boundary='extend') - grid.diff(grid.interp(u * dsx.DXU, 'X', boundary='fill'), 'Y', boundary='extend') ) / dsx.TAREA
    ow= ss**2 + sn**2 - rv**2
    return ow 

def divergence(U,V,grid, dsx):
    dy = grid.get_metric(U, "Y")
    dx = grid.get_metric(V, "X")
    dz = grid.get_metric(U, "Z")
    U_T = (grid.interp(U * dy * dz, axis="Y", boundary="extend")) 
    V_T = (grid.interp(V * dx * dz, axis="X", boundary="extend")) 
    div = (grid.diff(U_T, axis="X", boundary="extend") + grid.diff(V_T, axis="Y", boundary="extend") )
    vol = grid.get_metric(div, "XYZ")
    div = div / vol
    return div


def hgradient(T, grid):
    """
    gradient of scalar xgcm_metrics
    for pop
    """
    T_y = grid.interp(T, axis="Y", boundary="extend")  # 0.5*(F[i,j+1]+F[i,j])
    T_x = grid.interp(T, axis="X", boundary="extend")  # 0.5*(F[i+1,j]+F[i,j])
    dTdx = grid.derivative(T_y, axis="X", boundary="extend")
    dTdy = grid.derivative(T_x, axis="Y", boundary="extend")
    return dTdx, dTdy


def vgradient(T, grid):
    """
    vertical gradient of scalar xgcm_metrics
    for pop
    """
    T_z = grid.interp(T, axis="Z", boundary="extend")  # 0.5*(F[i,j+1]+F[i,j])
    dTdz = grid.derivative(T_z, axis="Z", boundary="extend")
    return dTdz   

#=====================================================================================
#======== XGCM tools =================================================================
#=====================================================================================

def gridx(ds):
    ''' Reshapes POP dataset to XGCM grid and calculates volume and area for integrals '''
    grid, dsx = pop_tools.to_xgcm_grid_dataset(ds, periodic=False, metrics=metrics)
    vol=dsx.DXT*dsx.DYT*dsx.DZT*cm3_m3
    vol.attrs['units']='m^3'
    dsx['vol']=vol

    area=dsx.DXT*dsx.DYT*cm2_m2
    area.attrs['units']='m^2' 
    dsx['area']=area
    
    return dsx, grid

def to_index(ds):
    '''Helper function by Anna-Lena Deppenmeier for handling variables with different POP grid dimensions transformed in xgcm'''
    ds = ds.copy()
    for dim in ds.dims:
        if dim in ["nlon_t", "nlat_t", "nlon_u", "nlat_u"]:
            ds = ds.drop(dim).rename({dim: dim[:-2]})
    return ds



#=====================================================================================
#======== Constants =================================================================
#=====================================================================================
cm_m=1e-2
m_cm=1e2
cm2_m2=1e-4
m2_cm2=1e4
cm3_m3=1e-6
m3_cm3=1e6
cm_s_m_s=1e-2
cm_s_m_d=(1e-2)*(60*60*24)

mmol_mol=1e-3
nmol_mol=1e-9
mmol_Tmol=1e-3*1e-12

ml_l_mmol_m3=44.4
umol_kg_mmol_m3=1*1.027

s_day=1/(60*60*24)
day_s=60*60*24
s_yr=1/(60*60*24*365)
yr_s=60*60*24*365
s_5d=5*60*60*24

#=====================================================================================
#======== Fonts, etc =================================================================
#=====================================================================================

fs=14; 
alp=0.5