#!/usr/bin/env python
# coding: utf-8

# In[1]:

import glob
import numpy as np
import time
import os
from scipy import interpolate
from scipy.optimize import nnls
import json
import uuid
import multiprocessing as mp

from PyAstronomy import pyasl
import corner


import matplotlib.pyplot as plt


import sys
import importlib

from spectres.spectral_resampling_numba import spectres_numba  as spectres

from utils import *
#from  matplotlib import colormaps as cmaps
#cm = cmaps['viridis']


old_version=False
continuum_penalty=False

select_conti_like=False
sum_sigma=True
radial_version=True
run_folder='run1'

use_ultranest=True
ext_model=None
sur_powerlaw=False
abs_powerlaw=False

#slice_sampler=True

if __name__ == "__main__":
    input_file=sys.argv[1]

if len(input_file)!=0:
    print(' ')
    print('----------------')
    print('----------------')
    print('Input taken!')
    print(input_file)
    ex=os.path.isfile(input_file)
    if ex:
        print('File found')
    else:
        print('File not found')
        exit()
    print('----------------')
    print('----------------')
    print(' ')
else:
    print(' ')
    print('----------------')
    print('----------------')
    print('No Input found!')
    print('----------------')
    print('----------------')
    print(' ')
    
    exit()

if '.py' in input_file:
    input_wo_end=input_file.split('.')[0]
    mdl = importlib.import_module(input_wo_end)

    # is there an __all__?  if so respect it
    if "__all__" in mdl.__dict__:
        names = mdl.__dict__["__all__"]
    else:
        # otherwise we import all names that don't begin with _
        names = [x for x in mdl.__dict__ if not x.startswith("_")]

    # now drag them in
    globals().update({k: getattr(mdl, k) for k in names})
else:
    #if '.' in input_file:
        #input_wo_end=input_file.split('.')[0]
    
    unique_filename = str(uuid.uuid4())
    os.system(f'cp {input_file} temporary_{unique_filename}.py')
    mdl = importlib.import_module('temporary_'+unique_filename)
    os.system(f'rm temporary_{unique_filename}.py')
    # is there an __all__?  if so respect it
    if "__all__" in mdl.__dict__:
        names = mdl.__dict__["__all__"]
    else:
        # otherwise we import all names that don't begin with _
        names = [x for x in mdl.__dict__ if not x.startswith("_")]

    # now drag them in
    globals().update({k: getattr(mdl, k) for k in names})


use_extinction=False
if 'E(B-V)' in prior_dict or 'E(B-V)' in fixed_dict:
    if 'Rv' in prior_dict or 'Rv' in fixed_dict:
        use_extinction=True            
        print('Found the extinction parameters')
        try:
            ext_model
            print('Extinction model loaded from input file')
        except NameError:
            from dust_extinction.parameter_averages import G23 as ext_model
            print("G23 is the default extinction model")    

try:
    absorp_species_list
except NameError:
    absorp_species_list=[]
    print("absorp_species_list not set in input file")
    print("and set to empty")
try:
    dust_species_list
except NameError:
    dust_species_list=[]
    print("dust_species_list not set in input file")
    print("and set to empty")

if use_ultranest:
    print('UltraNest')
    import ultranest
    from ultranest.stepsampler import SliceSampler
    from ultranest.integrator import ReactiveNestedSampler
    from ultranest.calibrator import ReactiveNestedCalibrator
else:
    print('MultiNest')
    from pymultinest.solve import solve, run

    
try:
    slice_sampler
    print('slice_sampler')
    print(slice_sampler)
except NameError:
    slice_sampler=True
    print("slice_sampler not set in input file")
    print('Default is false')    
try:
    slab_folder
    print('slab_folder')
    print(slab_folder)
except NameError:
    slab_folder='./LineData/'
    print("slab_folder not set in input file")
    print('./LineData')
    
    
    
try:
    log_coldens
    print('Log coldens')
    print(log_coldens)
except NameError:
    log_coldens=False
    print("log_coldens not set in input file")
    print('Default is false')

try:
    coldens_restriction
    print('coldens_restriction')
    print(coldens_restriction)
except NameError:
    coldens_restriction=False
    print("coldens_restriction not set in input file")
    print('Default is False')

try:
    limit_integrated_flux
    print('limit_integrated_flux')
    print(limit_integrated_flux)
except NameError:
    limit_integrated_flux=False
    print('limit_integrated_flux False by default')    

if limit_integrated_flux:
    save_mol_flux=True
try:
    save_mol_flux
    print('save_mol_flux')
    print(save_mol_flux)
except NameError:
    save_mol_flux=False
    print('save_mol_flux set to:')
    print(save_mol_flux)

try:
    length_ultra
    print('length_ultra')
    print(length_ultra)
except NameError:
    if length_ultra:
        length_ultra=2
    
    else:
        save_mol_flux=False
    print('save_mol_flux set to:')
    print(save_mol_flux)

try:
    fit_water_ratios
    print('fit_water_ratios')
    print(fit_water_ratios)
except NameError:
    fit_water_ratios=False

    print('fit_water_ratios set to:')
    print(fit_water_ratios)

try:
    fit_gas_only
    print('fit_gas_only')
    print(fit_gas_only)
except NameError:
    if fit_water_ratios:
        fit_gas_only=True
    else:
        fit_gas_only=False

    print('fit_gas_only set to:')
    print(fit_gas_only)
try:
    weights_obs
    print('Observation is weighted')
    weighted=True
except NameError:
    weighted=False


    print('weighted set to:')
    print(weighted)


debug=False

sigma_dict={}

# # here starts the part where you have to adjust things
# 

# put in the observation that you want to fit
# - lam_obs: array with the wavelength points in micron
# - flux_obs: array with the corresponding fluxes in Jy
# - sig_obs: array with the corresponding uncertainties of the observations

# In[23]:

if not fit_gas_only:
    max_flux_obs=np.max(flux_obs)




# ### setting up the folder where you want to run things
# there are two option to run multinest solve or run.
# 
# I'm not sure what the difference is, but if you leave running=False the solve option is used.
# 
# - subfold is the folder in which the files are saved
# - if you already run things with the same data, it might be nice to save them in the same folder but with a different name. This is done with run_number, that will append the new number to the file names

# In[12]:


running=False

fold_string=bayesian_folder
if subfold!='':
    fold_string=fold_string+subfold
if __name__ == "__main__":

        if not os.path.exists(fold_string):
            os.system(f'mkdir {fold_string}')
        else:
            print(f'Folder {fold_string} exists')


    # run MultiNest
prefix = fold_string+'test_'+str(run_number)






init_abundance={}
for entry in dust_species_list:
    init_abundance[entry]=None
init_abundance_absorp={}
for entry in absorp_species_list:
    init_abundance_absorp[entry]=None
# are there parameters that you want to fix to values instead of fitting them
# 


# here you can set the priors for the parameters that you want to fit
# 
# if you want to have non-uniform priors I can implement that put this is currently not done

# In[15]:



for key in slab_prior_dict:
    for key1 in slab_prior_dict[key]:
        new_key=key+':'+key1
        prior_dict[new_key]=slab_prior_dict[key][key1]
print(prior_dict)


# In[18]:


'''
prior of the scaling factors
'''

if sample_all:
    prior_dict_dust=init_abundance.copy()
    for key in prior_dict_dust:
        prior_dict_dust[key]=prior_scaling_dust



    prior_dict_dust_abs=init_abundance_absorp.copy()
    for key in prior_dict_dust_abs:
        prior_dict_dust_abs[key]=prior_scaling_abs


if 'tmax_s' in prior_dict or 'temp_s' in prior_dict:
    use_dust_emis=True
    if 'tmax_s' in prior_dict:
        sur_powerlaw=True
    else:
        sur_powerlaw=False
else:
    use_dust_emis=False
if 'tmax_abs' in prior_dict or 'temp_abs' in prior_dict:
    use_dust_absorp=True
    if 'tmax_abs' in prior_dict:
        abs_powerlaw=True
    else:
        abs_powerlaw=False
else:
    use_dust_absorp=False
use_mol_powerlaw=False
if 'q_emis' in prior_dict or 'q_emis' in fixed_dict:
    use_mol_powerlaw=True
    
# setting up the dictonaries and headers that will be used

# In[19]:
init_dict=return_init_dict(use_bb_star=use_bb_star,rin_powerlaw=rin_powerlaw,fit_gas_only=fit_gas_only,
                           prior_dict=prior_dict,fixed_dict=fixed_dict,use_extinction=use_extinction,use_dust_emis=use_dust_emis,use_dust_absorp=use_dust_absorp,sur_powerlaw=sur_powerlaw,abs_powerlaw=abs_powerlaw,mol_powerlaw=use_mol_powerlaw)



if 'log_sigma_obs' in prior_dict:
    fit_obs_err=True
    fit_abs_err=False
elif 'log_sigma_obs_abs' in prior_dict:
    fit_obs_err=True
    fit_abs_err=True
elif 'sigma_obs_abs' in prior_dict:
    fit_obs_err=True
    fit_abs_err=True
else:
    fit_obs_err=False
    fit_abs_err=False

if 'log_sigma_conti' in prior_dict:
    fit_conti_err=True
else:
    fit_conti_err=False
header,header_para,header_abund,header_slab,header_absorp,header_sigma=create_header(var_dict=init_dict,
                                                              abundance_dict=init_abundance,
                                                              slab_dict=slab_prior_dict,
                                                              fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err,fit_abs_err=fit_abs_err,
                                                              fixed_dict=fixed_dict,prior_dict=prior_dict,abundance_dict_absorption=init_abundance_absorp)
                                                              
upper_lim=[]
lower_lim=[]
complete_header=[]
for key in header_para:
    upper_lim.append(prior_dict[key][1])
    lower_lim.append(prior_dict[key][0])
    complete_header.append(key)
for key in header_slab:
    upper_lim.append(prior_dict[key][1])
    lower_lim.append(prior_dict[key][0])
    complete_header.append(key)
if sample_all:
    for key in prior_dict_dust:
        upper_lim.append(prior_dict_dust[key][1])
        lower_lim.append(prior_dict_dust[key][0])
        complete_header.append(key)
    for key in scale_prior:
        upper_lim.append(scale_prior[key][1])
        lower_lim.append(scale_prior[key][0])
        complete_header.append(key)

if fit_obs_err:
    if 'log_sigma_obs' in prior_dict:
        upper_lim.append(prior_dict['log_sigma_obs'][1])
        lower_lim.append(prior_dict['log_sigma_obs'][0])
        complete_header.append('log_sigma_obs')
        
    elif 'sigma_obs' in prior_dict:
        upper_lim.append(prior_dict['sigma_obs'][1])
        lower_lim.append(prior_dict['sigma_obs'][0])
        complete_header.append('sigma_obs')
    elif 'log_sigma_obs_abs' in prior_dict:
        upper_lim.append(prior_dict['log_sigma_obs_abs'][1])
        lower_lim.append(prior_dict['log_sigma_obs_abs'][0])
        complete_header.append('log_sigma_obs')
        
    elif 'sigma_obs_abs' in prior_dict:
        upper_lim.append(prior_dict['sigma_obs_abs'][1])
        lower_lim.append(prior_dict['sigma_obs_abs'][0])
        complete_header.append('sigma_obs')
            
if fit_conti_err:
    if 'log_sigma_conti' in prior_dict:
        upper_lim.append(prior_dict['log_sigma_conti'][1])
        lower_lim.append(prior_dict['log_sigma_conti'][0])
        complete_header.append('log_sigma_conti')
    elif 'sigma_conti' in prior_dict:
        upper_lim.append(prior_dict['sigma_conti'][1])
        lower_lim.append(prior_dict['sigma_conti'][0])
        complete_header.append('sigma_conti')
    
upper_lim=np.array(upper_lim)
lower_lim=np.array(lower_lim)


print('Upper lim', upper_lim)
print('Lower lim', lower_lim)


def prior_fast(cube):
    new_cube=(cube)*(upper_lim-lower_lim)+lower_lim
    return new_cube
def prior_run_fast(cube,ndim,nparams):
    
    cube=(cube)*(upper_lim-lower_lim)+lower_lim
    #return new_cube



def loglike_ratios(cube,debug=False,timeit=False):
    if timeit:
        time_1=time()
    if sample_all:
        var_dict,abundance_dict,slab_dict,sigma_dict=cube_to_dicts(cube,header_para=header_para,header_abund=header_abund,header_all=header,scale_prior=scale_prior,fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)

    else:
        var_dict,slab_dict,sigma_dict=cube_to_dict(cube,header=list(header_para)+list(header_slab)+list(header_sigma),fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err,log_coldens=log_coldens)

        abundance_dict={}
  
    if debug:
        print(var_dict)
        if sample_all:
            print(abundance_dict)
        print(slab_dict)
        if fit_conti_err or fit_obs_err:
            print(sigma_dict)
    if fixed_paras:
        for key in fixed_dict:
            if debug:
                print(f'Fixed {key}..')
            if key in header_abund:
                abundance_dict[key]=fixed_dict[key]
                if debug:
                    print('..added to abundance_dict')
            elif key in init_dict or key=='distance':
                var_dict[key]=fixed_dict[key]
                if debug:
                    print('..added to var_dict')
            elif key =='sigma_obs':
                sigma_dict['sigma_obs']=fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')
            elif key =='log_sigma_obs':
                sigma_dict['sigma_obs']=10**fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')
            elif key =='sigma_obs_abs':
                sigma_dict['sigma_obs_abs']=fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')
            elif key =='log_sigma_obs_abs':
                sigma_dict['sigma_obs_abs']=10**fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')

            elif ':' in key:
                idx=key.find(':')
                if key[:idx] not in slab_dict:
                    slab_dict[key[:idx]]={}
                if debug:
                    print('..added to slab_dict')

                slab_dict[key[:idx]][key[idx+1:]]=fixed_dict[key]
            else:
                print(f'{key} is in fixed_dict but not used for the retrieval. Please check that.')
     
                
    var_dict['bb_star']=use_bb_star
    
    #checking if the physics works out
    penalty=float(-10**100.0)
    sum_penalty=float(-10**100.0)
    trigger_penalty=False
    for key in slab_dict:
        if 'tmin' in slab_dict[key]: 
            if slab_dict[key]['tmin']>=slab_dict[key]['tmax']:
                trigger_penalty=True
                sum_penalty+=penalty*(slab_dict[key]['tmin']-slab_dict[key]['tmax'])
                if debug: print(f'Penalty {key} temp')
        if coldens_restriction:
            if 'ColDens_tmin' in slab_dict[key]: 
                if slab_dict[key]['ColDens_tmin']>slab_dict[key]['ColDens_tmax']:
                    trigger_penalty=True
                    sum_penalty+=penalty*(slab_dict[key]['ColDens_tmin']-slab_dict[key]['ColDens_tmax'])
                    if debug: print(f'Penalty {key} coldens')
    if trigger_penalty:
        if debug:
            print('Triggered penalty')
        return sum_penalty
    if timeit:
        time_2=time()    

    #here we are inserting the run of the model (with fixed radii) and the calculation of the ratios
    '''
    '''

    con_model.run_model(variables=var_dict,dust_species=abundance_dict,slab_dict=slab_dict,output_all=False,timeit=False)
    ratio_calc_dict,_=con_model.water_line_diagnostics(debug=False,plot=False,ratios=True,printing=False)
    ratio_calc=np.array(list(ratio_calc_dict.values()))

    if len(ratio_calc)!=len(ratio_obs):
        return penalty
    
    if timeit:
        time_3=time()

    if 'sigma_obs' in sigma_dict:
        sigma=sigma_dict['sigma_obs']*flux_obs   
    elif 'sigma_obs_abs' in sigma_dict:
        sigma=sigma_dict['sigma_obs_abs']   
    else:
        sigma=sig_obs
    # constant of loglike
    const=np.sum(np.log(2*np.pi*(sigma)**2))

    #difference between observation and model

    diff=(ratio_calc - ratio_obs)

    #definition of chi
    if weighted:
        chi=np.sum(weights_obs*(diff)**2/ sigma**2)
    else:
        chi=np.sum((diff)**2/ sigma**2)
        
    #loglike
    loglikelihood =  -0.5 * (chi +const) 
    
    if timeit:
        time_4=time()
        print('Dictonary: ', time_2-time_1)
        print('Run model: ', time_3-time_2)
        print('Calc loglike: ', time_4-time_3)
    if debug:
        print('loglikelihood:',loglikelihood)
    if (not np.isfinite(loglikelihood)) or (np.isnan(loglikelihood)):
        return penalty
    else:
        return loglikelihood



def loglike_gas(cube,debug=False,timeit=False):
    '''
    This likelihood function is using in gas we are fitting the gas only.
    This means that the continuum does not need to be calculated.
    Also, we are using absolute uncertainties for the flux and not relative uncertainties.
    This is because we will have (very likely) negative fluxes and consequenctly negative sigma,
    which is a mess.
    The desription of absolute errors should also be introduced to the full fitting in the future.
    '''
    if timeit:
        time_1=time()
    if sample_all:
        var_dict,abundance_dict,slab_dict,sigma_dict=cube_to_dicts(cube,header_para=header_para,header_abund=header_abund,header_all=header,scale_prior=scale_prior,fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)

    else:
        var_dict,slab_dict,sigma_dict=cube_to_dict(cube,header=list(header_para)+list(header_slab)+list(header_sigma),fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err,log_coldens=log_coldens)

        abundance_dict={}
  
    if debug:
        print(var_dict)
        if sample_all:
            print(abundance_dict)
        print(slab_dict)
        if fit_conti_err or fit_obs_err:
            print(sigma_dict)
    if fixed_paras:
        for key in fixed_dict:
            if debug:
                print(f'Fixed {key}..')
            if key in header_abund:
                abundance_dict[key]=fixed_dict[key]
                if debug:
                    print('..added to abundance_dict')
            elif key in init_dict or key=='distance':
                var_dict[key]=fixed_dict[key]
                if debug:
                    print('..added to var_dict')
            elif key =='sigma_obs':
                sigma_dict['sigma_obs']=fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')
            elif key =='log_sigma_obs':
                sigma_dict['sigma_obs']=10**fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')
            elif key =='sigma_obs_abs':
                sigma_dict['sigma_obs_abs']=fixed_dict[key]
                if debug:
                    print(key,'..added to sigma_dict')
            elif key =='log_sigma_obs_abs':
                sigma_dict['sigma_obs_abs']=10**fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')

            elif ':' in key:
                idx=key.find(':')
                if key[:idx] not in slab_dict:
                    slab_dict[key[:idx]]={}
                if debug:
                    print('..added to slab_dict')

                slab_dict[key[:idx]][key[idx+1:]]=fixed_dict[key]
            else:
                print(f'{key} is in fixed_dict but not used for the retrieval. Please check that.')
     
                
    var_dict['bb_star']=use_bb_star
    
    #checking if the physics works out
    penalty=float(-10**100.0)
    sum_penalty=float(-10**100.0)
    trigger_penalty=False
    for key in slab_dict:
        if 'tmin' in slab_dict[key]: 
            if slab_dict[key]['tmin']>=slab_dict[key]['tmax']:
                trigger_penalty=True
                sum_penalty+=penalty*(slab_dict[key]['tmin']-slab_dict[key]['tmax'])
                if debug: print(f'Penalty {key} temp')
        if coldens_restriction:
            if 'ColDens_tmin' in slab_dict[key]: 
                if slab_dict[key]['ColDens_tmin']>slab_dict[key]['ColDens_tmax']:
                    trigger_penalty=True
                    sum_penalty+=penalty*(slab_dict[key]['ColDens_tmin']-slab_dict[key]['ColDens_tmax'])
                    if debug: print(f'Penalty {key} coldens')
    if trigger_penalty:
        if debug:
            print('Triggered penalty')
        return sum_penalty
    
    if timeit:
        time_2=time()    

    #here we are inserting the run of the model (with radii) and the fluxes
    '''
    '''

    interp_flux=con_model.run_model(variables=var_dict,dust_species=abundance_dict,slab_dict=slab_dict,output_all=False,timeit=False)

    
    if timeit:
        time_3=time()

    if 'sigma_obs' in sigma_dict:
        sigma=sigma_dict['sigma_obs']*flux_obs   
    elif 'sigma_obs_abs' in sigma_dict:
        sigma=sigma_dict['sigma_obs_abs']   
    else:
        sigma=sig_obs
    # constant of loglike
    const=np.sum(np.log(2*np.pi*(sigma)**2))


   #difference between observation and model

    diff=(interp_flux - flux_obs)

    #definition of chi
    if weighted:
        chi=np.sum(weights_obs*(diff)**2/ sigma**2)
    else:
        chi=np.sum((diff)**2/ sigma**2)

    #loglike
    loglikelihood =  -0.5 * (chi +const) 
 
    
    if timeit:
        time_4=time()
        print('Dictonary: ', time_2-time_1)
        print('Run model: ', time_3-time_2)
        print('Calc loglike: ', time_4-time_3)
    if debug:
        print('loglikelihood:',loglikelihood)
    if (not np.isfinite(loglikelihood)) or (np.isnan(loglikelihood)):
        return penalty
    else:
        return loglikelihood
    
def loglike(cube,debug=False,timeit=False,return_model=False):
    sigma_dict={}
    if timeit:
        time_1=time()
    if sample_all:
        var_dict,abundance_dict,slab_dict,abundance_dict_absorp,sigma_dict=cube_to_dicts(cube,header_para=header_para,header_abund=header_abund,header_all=complete_header,header_absorp=header_absorp,scale_prior=scale_prior,fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)

    else:
        var_dict,slab_dict,sigma_dict=cube_to_dict(cube,header=list(header_para)+list(header_slab)+list(header_sigma),fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err,log_coldens=log_coldens)

    if debug:
        print(var_dict)
        if sample_all:
            print(abundance_dict)
            print(abundance_dict_absorp)
            
        print(slab_dict)
        if fit_conti_err or fit_obs_err:
            print(sigma_dict)
    if fixed_paras:
        for key in fixed_dict:
            if debug:
                print(f'Fixed {key}..')
            if key in header_abund:
                abundance_dict[key]=fixed_dict[key]
                if debug:
                    print('..added to abundance_dict')
            elif key in header_absorp:
                key_abs=key[:-7]
                abundance_dict_absorp[key_abs]=fixed_dict[key]
                if debug:
                    print('..added to abundance_dict_absorp')
               
            elif key in init_dict or key=='distance':
                var_dict[key]=fixed_dict[key]
                if debug:
                    print('..added to var_dict')
            elif key =='sigma_obs':
                sigma_dict['sigma_obs']=fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')
            elif key =='log_sigma_obs':
                sigma_dict['sigma_obs']=10**fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')
            elif key =='sigma_obs_abs':
                sigma_dict['sigma_obs_abs']=fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')
            elif key =='log_sigma_obs_abs':
                sigma_dict['sigma_obs_abs']=10**fixed_dict[key]
                if debug:
                    print('..added to sigma_dict')

            elif ':' in key:
                idx=key.find(':')
                if key[:idx] not in slab_dict:
                    slab_dict[key[:idx]]={}
                if debug:
                    print('..added to slab_dict')

                slab_dict[key[:idx]][key[idx+1:]]=fixed_dict[key]
            else:
                print(f'{key} is in fixed_dict but not used for the retrieval. Please check that.')
  
                
    var_dict['bb_star']=use_bb_star
    
    #checking if the physics works out
    penalty=float(-10**100.0)
    sum_penalty=float(-10**100.0)
    trigger_penalty=False
    if use_dust_emis:
        if sur_powerlaw:
            if var_dict['tmin_s']>=var_dict['tmax_s']:
                trigger_penalty=True
                sum_penalty+=penalty*(var_dict['tmin_s']-var_dict['tmax_s'])
                if debug: print('Penalty t surface')
    if use_dust_absorp:
        if abs_powerlaw:
            if var_dict['tmin_abs']>=var_dict['tmax_abs']:
                trigger_penalty=True
                sum_penalty+=penalty*(var_dict['tmin_abs']-var_dict['tmax_abs'])
                if debug: print('Penalty t abs')
    
    if var_dict['tmin_mp']>=var_dict['tmax_mp']:
        trigger_penalty=True
        sum_penalty+=penalty*(var_dict['tmin_mp']-var_dict['tmax_mp'])
        if debug: print('Penalty t mp')
    
    if 't_rim' not in var_dict.keys():
        if var_dict['tmin_rim']>=var_dict['tmax_rim']:
            trigger_penalty=True
            sum_penalty+=penalty*(var_dict['tmin_rim']-var_dict['tmax_rim'])
            if debug: print('Penalty t rim')
    
    for key in slab_dict:
        if 'tmin' in slab_dict[key]: 
            if slab_dict[key]['tmin']>=slab_dict[key]['tmax']:
                trigger_penalty=True
                sum_penalty+=penalty*(slab_dict[key]['tmin']-slab_dict[key]['tmax'])
                if debug: print(f'Penalty {key} temp')
        if coldens_restriction:
            if 'ColDens_tmin' in slab_dict[key]: 
                if slab_dict[key]['ColDens_tmin']>slab_dict[key]['ColDens_tmax']:
                    trigger_penalty=True
                    sum_penalty+=penalty*(slab_dict[key]['ColDens_tmin']-slab_dict[key]['ColDens_tmax'])
                    if debug: print(f'Penalty {key} coldens')
    if trigger_penalty:
        if debug:
            print('Triggered penalty')
        return sum_penalty

    if timeit:
        time_2=time()    
    if sample_all:
        interp_flux=con_model.run_model_normalized(variables=var_dict,dust_species=abundance_dict,
                                                slab_dict=slab_dict,absorp_species=abundance_dict_absorp,max_flux_obs=max_flux_obs)


        
    else:
        interp_flux=con_model.run_fitted_to_obs(variables=var_dict,
                                                dust_species=init_abundance,
                                                absorp_species=init_abundance_absorp,
                                                slab_dict=slab_dict,
                                                flux_obs=flux_obs,lam_obs=lam_obs,save_mol_flux=save_mol_flux)

    if limit_integrated_flux:
        for key in slab_dict:
            if key in limit_flux_dict:
                if debug:
                    print('lim')
                    print(key)
                int_flux=con_model.calc_integrated_flux(key)
                if debug:
                    print('int_flux')
                    print(int_flux)
                if int_flux>limit_flux_dict[key]:
                    trigger_penalty=True
                    sum_penalty+=penalty*(0.01+abs(int_flux-limit_flux_dict[key]))
        if trigger_penalty:
            return sum_penalty                   
    if timeit:
        time_3=time()
    
    if 'sigma_obs' in sigma_dict:
        sigma=sigma_dict['sigma_obs']*flux_obs   
    elif 'sigma_obs_abs' in sigma_dict:
        sigma=sigma_dict['sigma_obs_abs']   
    else:
        sigma=sig_obs
    # constant of loglike
    const=np.sum(np.log(2*np.pi*(sigma)**2))

    #difference between observation and model

    diff=(interp_flux - flux_obs)

    #definition of chi
    if weighted:
        chi=np.sum(weights_obs*(diff)**2/ sigma**2)
    else:
        chi=np.sum((diff)**2/ sigma**2)
    #loglike
    loglikelihood =  -0.5 * (chi +const) 
    
    if continuum_penalty:
        continuum_residual=con_model.saved_continuum-flux_obs
        cliped_residual=np.clip(continuum_residual,a_min=0.0,a_max=None)

        
        if fit_conti_err:
            sigma_conti=sigma_dict['sigma_conti']*flux_obs
            if select_conti_like:
                idx_select=np.where(cliped_residual>0.0)[0]
                sigma=sigma[idx_select]
                cliped_residual=cliped_residual[idx_select]
            if sum_sigma:
                sig_tot=np.sqrt(sigma_conti**2+sigma**2)
            else:
                sig_tot=sigma_conti
        else:
            sig_tot=sigma
        # constant of loglike
        const=np.sum(np.log(2*np.pi*(sig_tot)**2))
        
        #definition of chi
        if not fit_conti_err:
            chi=np.sum((cliped_residual)**2/ sig_tot**2)*penalty_fact
        else:
            chi=np.sum((cliped_residual)**2/ sig_tot**2)

        #loglike
        loglikelihood -=  0.5 * (chi +const) 
    if timeit:
        time_4=time()
        print('Dictonary: ', time_2-time_1)
        print('Run model: ', time_3-time_2)
        print('Calc loglike: ', time_4-time_3)
    if debug:
        plt.loglog(con_model.xnew,interp_flux,label='model')
        plt.plot(lam_obs,flux_obs,label='Obs')
        plt.legend()
        plt.show()
        
        plt.loglog(con_model.xnew,interp_flux,label='model')
        plt.plot(lam_obs,flux_obs,label='Obs')
        plt.legend()
        plt.xlim([4,7])
        plt.show()
        plt.loglog(con_model.xnew,interp_flux,label='model')
        plt.plot(lam_obs,flux_obs,label='Obs')
        plt.legend()
        plt.xlim([10,20])
        plt.show()
    if debug:
        print('loglikelihood:',loglikelihood)
    if return_model:
        return con_model
    #print(loglikelihood)
    if (not np.isfinite(loglikelihood)) or (np.isnan(loglikelihood)):
        return penalty
    else:
        return loglikelihood

# initializing the model and reading in the data

# In[20]:


con_model=complete_model()

#Additing the molecules that are only in the fixed_dict ot the read in phase
#so that they are loaded as well
load_in_slab_dict={}
for key in slab_prior_dict:
    
    load_in_slab_dict[key]={}
for key in fixed_dict:
    if ':' in key:
        idx=key.find(':')
        if key[:idx] not in load_in_slab_dict:
            load_in_slab_dict[key[:idx]]={}
print(load_in_slab_dict)

        

if not fit_gas_only:
    con_model.read_data(variables=init_dict,dust_species=init_abundance, 
                        absorp_species=init_abundance_absorp, 
                        slab_dict=load_in_slab_dict,slab_prefix=slab_prefix,
                        stellar_file=stellar_file,wavelength_points=lam_obs,
                        dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)

else:
    try:
        print(len(lam_obs))
        con_model.read_data(variables=init_dict,dust_species=init_abundance,slab_dict=load_in_slab_dict,slab_prefix=slab_prefix,
                            stellar_file=stellar_file,wavelength_points=lam_obs,slab_only_mode=True,
                            dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)
    except NameError:
        con_model.read_data(variables=init_dict,dust_species=init_abundance,
                            slab_dict=load_in_slab_dict,slab_prefix=slab_prefix,
                            stellar_file=stellar_file,slab_only_mode=True,
                            dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)

if __name__ == "__main__":
    print(con_model)


# # Let's run

# In[25]:
if use_ultranest:
    try:
        slice_sampler
        print('Slice_sampler')
    except:
        print('Slice_sampler not set default is True')
        
        slice_sampler=True
        
    if slice_sampler:
        try:
            length_ultra
            print('length_ultra')
        except:
            print('length_ultra not set default is 2')
            
            length_ultra=2
        try:
            adaptive_nsteps
            print('Adaptive_nsteps')
        except:
            print('Adaptive_nsteps is not set')
            adaptive_nsteps=False
    try:
        n_live_points
        print('n_live_points')
    except:
        print('n_live_points not set')
        n_live_points=400
        print('Default is 400')
    try:
        evidence_tolerance
        print('evidence_tolerance')
    except:
        print('evidence_tolerance not set')
        evidence_tolerance=0.5
        print('Default is 0.5')  
    try:
        frac_remain 
        print('frac_remain ')
    except:
        print('frac_remain  not set')
        frac_remain=0.001
        print('Default is 0.5')  

else:
    try:
        n_live_points
        evidence_tolerance
        sampling_efficiency
    except NameError:
        print('Using fast_retrieval:',fast_retrival)
    
        if fast_retrival:
            n_live_points = 400#50
            evidence_tolerance = 5.0
            sampling_efficiency = 0.8
        else:
            n_live_points = 1000
            evidence_tolerance = 0.5
            sampling_efficiency = 0.3
    print('n_live_points',n_live_points)
    print('evidence_tolerance',evidence_tolerance)   
    print('sampling_efficiency',sampling_efficiency)   

if debug:
    print('N dims',len(upper_lim))
    test_vals=[]
    for i in range(len(upper_lim)):
        test_vals.append((upper_lim[i]+lower_lim[i])/2)
    print(len(test_vals))
    print(test_vals)
    if fit_water_ratios:
        loglike_ratios(test_vals,debug=True,timeit=False)
    elif fit_gas_only:
        loglike_gas(test_vals,debug=True,timeit=False)
    else:
        loglike(test_vals,debug=True,timeit=False)

if __name__ == "__main__":
    if not os.path.isfile(f'{prefix}start.time'):
        os.system(f'date > {prefix}start.time')
    if fit_water_ratios:
        if use_ultranest:

            sampler = ultranest.ReactiveNestedSampler(
            complete_header,
            loglike_ratios,
            prior_fast,
            log_dir=prefix,
            resume=True)
            
            result = sampler.run(min_num_live_points=n_live_points,Lepsilon=evidence_tolerance,frac_remain=frac_remain)
        else:
            result = solve(LogLikelihood=loglike_ratios, Prior=prior_fast, 
               n_dims=len(upper_lim), outputfiles_basename=prefix, verbose=True,
               n_live_points = n_live_points,evidence_tolerance = evidence_tolerance, 
               sampling_efficiency = sampling_efficiency, importance_nested_sampling=False)

    elif fit_gas_only:
        if use_ultranest:

            sampler = ultranest.ReactiveNestedSampler(
            complete_header,
            loglike_gas,
            prior_fast,
            log_dir=prefix,
            resume=True)
            
            result = sampler.run(min_num_live_points=n_live_points,Lepsilon=evidence_tolerance,frac_remain=frac_remain)
        else:
            result = solve(LogLikelihood=loglike_gas, Prior=prior_fast, 
               n_dims=len(upper_lim), outputfiles_basename=prefix, verbose=True,
               n_live_points = n_live_points,evidence_tolerance = evidence_tolerance, 
               sampling_efficiency = sampling_efficiency, importance_nested_sampling=False)
    else:
        if use_ultranest:
    

            
            if not slice_sampler:
                sampler = ReactiveNestedSampler(
                    complete_header,
                    loglike,
                    prior_fast,
                    log_dir=prefix,
                    resume=True)
                result = sampler.run(min_num_live_points=n_live_points,Lepsilon=evidence_tolerance,frac_remain=frac_remain)
        
            if slice_sampler:
                nsteps = length_ultra * len(complete_header)
                sampler = ReactiveNestedSampler(
                    complete_header,
                    loglike,
                    prior_fast,
                    log_dir=prefix,
                    resume=True)
                # create step sampler:
                sampler.stepsampler = SliceSampler(
                    nsteps=nsteps,
                    generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                    adaptive_nsteps=adaptive_nsteps,
                    # max_nsteps=400
                )
                print(np.shape(sampler))
                result = sampler.run(min_num_live_points=n_live_points,Lepsilon=evidence_tolerance,frac_remain=frac_remain)

                
            sampler.print_results()
        else:
            result = solve(LogLikelihood=loglike, Prior=prior_fast, 
                           n_dims=len(upper_lim), outputfiles_basename=prefix, verbose=True,
                           n_live_points = n_live_points,evidence_tolerance = evidence_tolerance, 
                           sampling_efficiency = sampling_efficiency, importance_nested_sampling=False)
    if not os.path.isfile(f'{prefix}end.time'):
        os.system(f'date > {prefix}end.time')


