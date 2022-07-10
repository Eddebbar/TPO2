#!/usr/bin/env python
# coding: utf-8

# # {{variable_id}}

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import xarray as xr
from dask.distributed import Client
import catalog
import util

xr.set_options(keep_attrs=True)


# ## Parameters

# In[2]:


# Monthly files
# /glade/campaign/cesm/development/bgcwg/projects/hi-res_JRA/cases/g.e22.G1850ECO_JRA_HR.TL319_t13.004/output/ocn/proc/tseries/month_1/g.e22.G1850ECO_JRA_HR.TL319_t13.004.pop.h.O2.003401-003412.nc

# Daily files
# /glade/campaign/cesm/development/bgcwg/projects/hi-res_JRA/cases/g.e22.G1850ECO_JRA_HR.TL319_t13.004/output/ocn/proc/tseries/day_1/g.e22.G1850ECO_JRA_HR.TL319_t13.004.pop.h.nday1.SST.00340101-00341231.nc


# In[3]:


archive = '/glade/campaign/cesm/development/bgcwg/projects/hi-res_JRA/cases/g.e22.G1850ECO_JRA_HR.TL319_t13.004/output/ocn/proc/tseries/'
freq= 'day_1'  #freq= 'month_1'
case = 'g.e22.G1850ECO_JRA_HR.TL319_t13.004'
component = 'pop'
stream = 'h.nday1' #stream ='h'
years='00340101-00341231'
variable_id = 'SST'

cluster_scheduler_address = None


# In[4]:


assert component in ['pop']
assert stream in ['h', 'h.nday1']


# ## Connect to cluster

# if cluster_scheduler_address is None:
#     cluster, client = util.get_ClusterClient()
#     cluster.scale(5)
# else:
#     client = Client(cluster_scheduler_address)
# client

# ## Load the data

# In[7]:


dset = catalog.to_dataset_dict(
    case=case,
    component=component,
    stream=stream,
    freq=freq,
    variable_id=variable_id,
    years=years,
)
dset.keys()


# <!-- ## Compute long-term mean and plot -->

# In[8]:


ds = dset[f'{case}.{component}.{stream}.{variable_id}.{years}']
ds


# In[1]:


ds[variable_id].mean('time').plot()


# In[ ]:




