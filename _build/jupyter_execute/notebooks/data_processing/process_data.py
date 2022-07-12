#!/usr/bin/env python
# coding: utf-8

# # Data Processing Script

# ## All model output files can be foud on glade/cheyenne: 
# 
# CESM JRA55 0.1º daily runs: '/glade/campaign/cesm/development/bgcwg/projects/hi-res_JRA/cases/g.e22.G1850ECO_JRA_HR.TL319_t13.004/output/ocn/proc/tseries/day_1/' 
# CESM JRA55 0.1º monthly runs: '/glade/campaign/cesm/development/bgcwg/projects/hi-res_JRA/cases/g.e22.G1850ECO_JRA_HR.TL319_t13.004/output/ocn/proc/tseries/monthly_1/'     
# CESM CORE 0.1º 5yr Runs:  '/glade/campaign/cesm/development/bgcwg/projects/hi-res_CESM1_CORE/g.e11.G.T62_t12.eco.006/ocn/hist/'
# B-TPOSE:  '/glade/work/yeddebba/BTPOSE/www.ecco.ucsd.edu/DATA/TROPAC/bgc/'
# TPOSE: '/glade/work/yeddebba/TPOSE/'
# CMIP6: 
#     'https://cmip6-preprocessing.readthedocs.io/en/latest/tutorial.html#combo'  
#     'https://github.com/jbusecke/cmip6_preprocessing'   
#     'https://github.com/pangeo-gallery/example-gallery'

# ## Tasks:
# 1. CESM Model outputs slices 
# 2. TPOSE/MITgcm load model outputs from ecco repository
# 3. Calcualte climatologies etc.
