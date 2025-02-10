import numpy as np
import time
import os
from scipy import interpolate
from scipy.optimize import nnls
import json
import uuid
import multiprocessing as mp
import glob
from PyAstronomy import pyasl
import corner
from matplotlib.lines import Line2D
import pickle 
import matplotlib as mpl
import matplotlib.pyplot as plt
from ast import literal_eval

import sys
import importlib
import argparse

from spectres.spectral_resampling_numba import spectres_numba  as spectres

from utils import *


filetype_fig='png'
plot_last_corner_plot=False



use_ultranest=False


#default values that can be overwritten by input
save_output=False
save_flux=False
save_mol_flux=False
load_output=False
plot_dust_individual=False
number_plotted_dust=0
zoom_list=[[None,None]]
reduce_posterior=False
ignore_spectrum_plot=False

ext_model=None
sur_powerlaw=False
abs_powerlaw=False
close_plots=False
run_all=False

if __name__ == "__main__":
    input_file=sys.argv[1]
    
    if len(sys.argv)>2:
       
        arg_list=sys.argv[1:]
        
        for i in range(len(arg_list)):
            argument=arg_list[i]
            if argument=='save':
                save_output=True
            elif argument=='load':
                load_output=True
            elif argument=='save_all':
                save_output=True
                save_flux=True   
            elif argument=='custom_list':
                zoom_list=np.array(literal_eval(arg_list[int(i+1)]),dtype='float64')

            elif argument=='custom':
                zoom_min=float(arg_list[int(i+1)])
                zoom_max=float(arg_list[int(i+2)])
                zoom_list=[[zoom_min,zoom_max]]
            elif argument=='plot_dust':
                plot_dust_individual=True
                if len(arg_list)-1!=i:
                    number_plotted_dust=int(arg_list[i+1])
                else:
                    number_plotted_dust=0
            elif argument=='ultranest':
                
                use_ultranest=True
                run_folder=str(arg_list[int(i+1)])
            elif argument=='reduce_post':
                reduce_posterior=True
                len_reduce_post=int(arg_list[i+1])
            elif argument=='no_spectrum':
                ignore_spectrum_plot=True
            elif argument=='close':
                 close_plots=True
            elif argument=='all':
                run_all=True
                save_output=True
            elif argument=='all_plus':
                run_all=True
                save_output=True
                save_flux=True  
            else:
                print('--------------')
                print('--------------')
                print('Unkown argument')
                print(argument)
                print('--------------')
                print('--------------')
print('save output')
print(save_output)
print('save fluxes')
print(save_flux)
print('plot dust')
print(plot_dust_individual)
print(number_plotted_dust)

print('Plot only the parameters and not the spectra')
print(ignore_spectrum_plot)



low_contribution=0.15
high_contribution=0.85
    
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
try:
    log_coldens
except NameError:
    log_coldens=False
    print("log_coldens not set in input file")
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

old_version=False
continuum_penalty=False

select_conti_like=False
sum_sigma=True
radial_version=True

if zoom_list[0][0]==None or zoom_list[0][1]==None:
    zoom_list=[[min(np.round(lam_obs,4)),max(np.round(lam_obs,4))]]
print('Zoom window')
print(zoom_list)    
print('Save:',save_output)
 
    
    
print('-----------------')                                     
print('-----------------')
print('Loading results from ')
if use_ultranest:
    print('UltraNest')
    import ultranest
    import ultranest.stepsampler
else:
    print('MultiNest')
    from pymultinest.solve import solve, run

print('-----------------')
print('-----------------')    
# here you hav to set the path where you want to save all the runs (bayesian_folder) and the dust_path where your Q-files are

   
debug=False
# here you have to set the path where you want to save all the runs (bayesian_folder) and the dust_path where your Q-files are


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


# initializing the model and reading in the data

# In[20]:

#Additing the molecules that are only in the fixed_dict ot the read in phase
#so that they are loaded as well
load_in_slab_dict={}
not_fitted_species=[]
for key in slab_prior_dict:
    
    load_in_slab_dict[key]={}
for key in fixed_dict:
    if ':' in key:
        idx=key.find(':')
        if key[:idx] not in load_in_slab_dict:
            load_in_slab_dict[key[:idx]]={}
            not_fitted_species.append(key[:idx])
not_fitted_species=np.array(not_fitted_species)
print(load_in_slab_dict)

con_model=complete_model()

if not fit_gas_only:
    con_model.read_data(variables=init_dict,dust_species=init_abundance,
                        absorp_species=init_abundance_absorp,
                        slab_dict=load_in_slab_dict,slab_prefix=slab_prefix,
                        stellar_file=stellar_file,wavelength_points=lam_obs,
                        dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)

else:
    try:
        print(len(lam_obs))
        con_model.read_data(variables=init_dict,dust_species=init_abundance,
                            slab_dict=load_in_slab_dict,slab_prefix=slab_prefix,
                            stellar_file=stellar_file,wavelength_points=lam_obs,slab_only_mode=True,
                            dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)
    except NameError:
        con_model.read_data(variables=init_dict,dust_species=init_abundance,
                            slab_dict=load_in_slab_dict,slab_prefix=slab_prefix,
                            stellar_file=stellar_file,slab_only_mode=True,
                            dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)


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
else:
    try:
        n_live_points
        evidence_tolerance
        sampling_efficiency
    except NameError:
        print('Using fast_retrieval:',fast_retrival)
    
        if fast_retrival:
            n_live_points = 1000#50
            evidence_tolerance = 5.0
            sampling_efficiency = 0.8
        else:
            n_live_points = 1000
            evidence_tolerance = 0.5
            sampling_efficiency = 0.3
    print('n_live_points',n_live_points)
    print('evidence_tolerance',evidence_tolerance)   
    print('sampling_efficiency',sampling_efficiency) 


try:
    sample_all
except NameError:
    sample_all=False
    print("Sample_all not set in multinest file")



scatter_obs=False



save_folder=f'{fold_string}figures/'

if not os.path.exists(save_folder):
    os.system(f'mkdir {save_folder}')
else:
    print(f'Folder {save_folder} exists')
print('')
print('')
print('----------------------------------')
print(f'saving figures in {save_folder}')
print('----------------------------------')
print('')
print('')

if not use_ultranest:
    samples=np.loadtxt(f'{prefix}post_equal_weights.dat')[:,:-1]
else:
    try:    
        samples=np.loadtxt(f'{prefix}/chains/equal_weighted_post.txt',skiprows=1,dtype='float32')
    except:
        samples=np.loadtxt(f'{prefix}/{run_folder}/chains/equal_weighted_post.txt',skiprows=1,dtype='float32')
    
print('Posterior shape:',np.shape(samples))

if reduce_posterior:
    print('Reducing the posterior points to')
    print(len_reduce_post)
    indices=np.random.choice(np.arange(0,len(samples)),len_reduce_post,replace=False)
    samples=samples[indices]
    print('New shape')
    print(np.shape(samples))
    reduce_str='_reduced'
else:
    reduce_str=''
    
def scatter_obs_gaussian(flux_obs,sig_obs,lam_obs,plot=False):
    new_flux=np.random.normal(flux_obs,sig_obs)

    if plot:
        plt.loglog(lam_obs,flux_obs)
        plt.fill_between(lam_obs,y1=flux_obs-sig_obs,y2=flux_obs+sig_obs,alpha=0.7)
        plt.scatter(lam_obs,new_flux,marker='+')
        plt.xlabel(r'$\lambda\rm [\mu m]$', fontsize=20)
        plt.ylabel(r'$F_\nu\rm [Jy]$', fontsize=20)
        plt.savefig(f'{save_folder}/scattered_observation_{run_number}.png')
        plt.show()
    return new_flux

#new_flux=scatter_obs_gaussian(flux_obs=flux_obs,sig_obs=sig_obs,lam_obs=lam_obs,plot=False)

#creating new wavelength grid that combines the observational wavelength and the standard model points

standard_wave=np.load('./standard_wave.npy')

    
# stiching part

above=np.unique(np.clip(standard_wave,a_min=np.max(con_model.xnew),a_max=None))
below=np.unique(np.clip(standard_wave,a_max=np.min(con_model.xnew),a_min=None))
#print(above)
out=np.append(above,below)
wave_new=np.sort(np.unique(np.append(out,con_model.xnew)))


scatter_obs=False

array_flux=[]
stellar_components=[]
rim_components=[]
midplane_components=[]
surface_components=[]

tot_samples=[]
con_model_new=complete_model()
if not fit_gas_only:
    
    con_model_new.read_data(variables=init_dict,wavelength_points=wave_new,dust_species=init_abundance,absorp_species=init_abundance_absorp,
                        slab_dict=load_in_slab_dict
    ,slab_prefix=slab_prefix,
                        stellar_file=stellar_file,dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)
else:
    con_model_new.read_data(variables=init_dict,wavelength_points=wave_new,dust_species=init_abundance,absorp_species=init_abundance_absorp,
                        slab_dict=load_in_slab_dict
    ,slab_prefix=slab_prefix,
                        stellar_file=stellar_file,dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model,slab_only_mode=True)
#print(con_model)

  
print(con_model)

print('Is extinction applied?')
print(con_model_new.use_extinction)
  
def get_scales_parallel(idx,obs_per_model,scatter_obs=scatter_obs, corr_noise=False,debug=False):
    if scatter_obs:
        print('----------------')
        print('----------------')
        print('This option has been removed!')
        print('----------------')
        print('----------------')
    
    samp=samples[idx]
    dict_fluxes={}
    var_dict,slab_dict,sigma_dict=cube_to_dict(samp,header=list(header_para)+list(header_slab))
    

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
            else:
                idx=key.find(':')
                if key[:idx] not in slab_dict:
                    slab_dict[key[:idx]]={}
                if debug:
                    print('..added to slab_dict')

                slab_dict[key[:idx]][key[idx+1:]]=fixed_dict[key]
  
  
    var_dict['bb_star']=use_bb_star
    if debug:
        print(var_dict)
        if sample_all:
            print(abundance_dict)
        print(slab_dict)
    interp_flux=con_model.run_fitted_to_obs(variables=var_dict,dust_species=init_abundance,absorp_species=init_abundance_absorp,
                                            slab_dict=slab_dict,flux_obs=flux_obs,lam_obs=lam_obs)
    
    scale_facs=con_model.scaleparas

#    tot_samples.append(np.append(samp,scale_facs))
    abundance_dict=init_abundance.copy()
    abundance_dict_absorp=init_abundance_absorp.copy()
    var_dict['sc_ir']=scale_facs[0]
    var_dict['sc_mid']=scale_facs[1]
    i=2
    if use_dust_emis:
        for key in abundance_dict:
            abundance_dict[key]=scale_facs[i]
            i+=1
    if use_dust_absorp:
        for key in abundance_dict_absorp:
            abundance_dict_absorp[key]=scale_facs[i]
            i+=1

    for key in slab_dict:
        
        if 'radius' not in slab_dict[key]:
            
            scale_facs[i]=np.sqrt(scale_facs[i])
            slab_dict[key]['radius']=scale_facs[i]
            i+=1
    tot_flux=con_model_new.run_model(variables=var_dict,dust_species=abundance_dict,absorp_species=abundance_dict_absorp,
                                     slab_dict=slab_dict,output_all=False)
    #section to get retrieved parameters from mol

    mol_results_dict=con_model_new.extract_emission_quantities(low_contribution=low_contribution,high_contribution=high_contribution)
    list_mol_results=[]

    for species in mol_results_dict:
        list_mol_results.append(mol_results_dict[species]['radius_eff'])
        list_mol_results.append(mol_results_dict[species]['tmin,tmax'][0])
        list_mol_results.append(mol_results_dict[species]['tmin,tmax'][1])
        list_mol_results.append(np.log10(mol_results_dict[species]['cmin,cmax'][0]))
        list_mol_results.append(np.log10(mol_results_dict[species]['cmin,cmax'][1]))
        if radial_version:
            list_mol_results.append(mol_results_dict[species]['rout,rin'][0])
            list_mol_results.append(mol_results_dict[species]['rout,rin'][1])
            
    #print(list_mol_results)
    list_mol_results=np.array(list_mol_results).flatten()
    dust_mass_ar=[]
    dust_mass_absorp_ar=[]
    if use_dust_emis:
        dust_mass_dict=con_model_new.calc_dust_masses(dust_path=dust_path,unit='msun')
        dust_mass_ar=np.array(list(dust_mass_dict.values()))
        if debug:
            print('Dust mass array')
            print(dust_mass_ar)
    if use_dust_absorp:
        dust_mass_dict=con_model_new.calc_dust_masses(dust_path=dust_path,absorption=True,unit='msun')
        dust_mass_absorp_ar=np.array(list(dust_mass_dict.values()))
        if debug:
            print('Dust mass absorption array')
            print(dust_mass_absorp_ar)
    if plot_dust_individual:
        scale_components=con_model_new.trans_flux
        if use_dust_emis:     
            dict_fluxes['individual_surface']={}
            for key in abundance_dict:    
                dict_fluxes['individual_surface'][key]=scale_components*con_model_new.surface_flux_individual[key]*abundance_dict[key]
                if debug:
                    print(np.max(dict_fluxes['individual_surface'][key]))
            dict_fluxes['interp_flux']=interp_flux
        if use_dust_absorp:     
            dict_fluxes['individual_absorp']={}
            for key in abundance_dict_absorp:    
                dict_fluxes['individual_absorp'][key]=scale_components*con_model_new.absorp_flux_individual[key]*abundance_dict_absorp[key]
                if debug:
                    print(np.max(dict_fluxes['individual_absorp'][key]))

    if not ignore_spectrum_plot:
        dict_fluxes['tot_flux']=tot_flux
        dict_fluxes['rim_flux']=con_model_new.rim_flux
        dict_fluxes['stellar_flux']=con_model_new.scaled_stellar_flux
        dict_fluxes['midplane_flux']=con_model_new.midplane_flux
        if use_dust_emis:
            dict_fluxes['surface_flux']=con_model_new.surface_flux_tot
        if use_dust_absorp:
            dict_fluxes['absorp_flux']=con_model_new.absorp_flux_tot
        dict_fluxes['emission_flux']=con_model_new.emission_flux
        dict_fluxes['interp_flux']=interp_flux

    return dict_fluxes, np.append(np.append(samp,scale_facs),list_mol_results),dust_mass_ar,dust_mass_absorp_ar




  

  



def get_full_model(idx,dummy,debug=False):
    samp=samples[idx]

    dict_fluxes={}
    sigma_dict={}
    if sample_all:
        var_dict,abundance_dict,slab_dict,abundance_dict_absorp,sigma_dict=cube_to_dicts(samp,header_para=header_para,header_abund=header_abund,header_absorp=header_absorp,header_all=complete_header,scale_prior=scale_prior)
    if fit_gas_only:
        abundance_dict={}
        var_dict,slab_dict,sigma_dict=cube_to_dict(samp,header=list(header_para)+list(header_slab))
    if debug:
        print(var_dict)
        if sample_all:
            print(abundance_dict)
        print(slab_dict)
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
            else:
                idx=key.find(':')
                if key[:idx] not in slab_dict:
                    slab_dict[key[:idx]]={}
                if debug:
                    print('..added to slab_dict')

                slab_dict[key[:idx]][key[idx+1:]]=fixed_dict[key]
  
  
    if debug:
        print('Slab dict',slab_dict)
    var_dict['bb_star']=use_bb_star
    
    if sample_all: interp_fluxes,scales=con_model.run_model_normalized(variables=var_dict,dust_species=abundance_dict,absorp_species=abundance_dict_absorp,
                                                slab_dict=slab_dict,max_flux_obs=max_flux_obs,translate_scales=True,debug=False)
    elif fit_gas_only:
        interp_flux=con_model.run_model(variables=var_dict,dust_species=abundance_dict,slab_dict=slab_dict,output_all=False,timeit=False)
    if sample_all:
        var_dict['sc_ir']=scales[0]
        var_dict['sc_mid']=scales[1]
        i=2
        if use_dust_emis:
            for key in abundance_dict:
                abundance_dict[key]=scales[i]
                i+=1
    
        if use_dust_absorp:
            for key in abundance_dict_absorp:
                abundance_dict_absorp[key]=scales[i]
                i+=1
    
    
        tot_flux=con_model_new.run_model(variables=var_dict,dust_species=abundance_dict,slab_dict=slab_dict,absorp_species=abundance_dict_absorp)
    elif fit_gas_only:
        tot_flux=con_model_new.run_model(variables=var_dict,dust_species=abundance_dict,slab_dict=slab_dict,output_all=False,timeit=False)
        #section to get retrieved parameters from mol
    mol_results_dict=con_model.extract_emission_quantities(low_contribution=low_contribution,high_contribution=high_contribution)
    list_mol_results=[]

    for species in mol_results_dict:

        list_mol_results.append(mol_results_dict[species]['radius_eff'])
        list_mol_results.append(mol_results_dict[species]['tmin,tmax'][0])
        list_mol_results.append(mol_results_dict[species]['tmin,tmax'][1])
        list_mol_results.append(np.log10(mol_results_dict[species]['cmin,cmax'][0]))
        list_mol_results.append(np.log10(mol_results_dict[species]['cmin,cmax'][1]))
        if radial_version:
            list_mol_results.append(mol_results_dict[species]['rout,rin'][0])
            list_mol_results.append(mol_results_dict[species]['rout,rin'][1])
    list_mol_results=np.array(list_mol_results).flatten()
    if sample_all:
        dust_mass_ar=[]
        dust_mass_absorp_ar=[]
        if use_dust_emis:
            dust_mass_dict=con_model_new.calc_dust_masses(dust_path=dust_path,unit='msun')
            dust_mass_ar=np.array(list(dust_mass_dict.values()))
        if use_dust_absorp:
            dust_mass_dict=con_model_new.calc_dust_masses(dust_path=dust_path,absorption=True,unit='msun')
            dust_mass_absorp_ar=np.array(list(dust_mass_dict.values()))    
        if plot_dust_individual:
            scale_components=con_model_new.trans_flux
    
            if use_dust_emis:
                con_model_new.set_surface(one_output=False)
            if use_dust_absorp:
                con_model_new.set_surface(absorption=True,one_output=False)
    
            if use_dust_emis:     
                dict_fluxes['individual_surface']={}
                for key in abundance_dict:    
                    dict_fluxes['individual_surface'][key]=scale_components*con_model_new.surface_flux_individual[key]*abundance_dict[key]
                    if debug:
                        print(np.max(dict_fluxes['individual_surface'][key]))
    
            if use_dust_absorp:     
                dict_fluxes['individual_absorp']={}
                for key in abundance_dict_absorp:    
                    dict_fluxes['individual_absorp'][key]=scale_components*con_model_new.absorp_flux_individual[key]*abundance_dict_absorp[key]
                    if debug:
                        print(np.max(dict_fluxes['individual_absorp'][key]))
    

    if not ignore_spectrum_plot:
        dict_fluxes['tot_flux']=tot_flux
        if sample_all:
            dict_fluxes['rim_flux']=con_model_new.rim_flux
            dict_fluxes['stellar_flux']=con_model_new.scaled_stellar_flux
            dict_fluxes['midplane_flux']=con_model_new.midplane_flux
            if use_dust_emis:
                dict_fluxes['surface_flux']=con_model_new.surface_flux_tot
            if use_dust_absorp:
                dict_fluxes['absorp_flux']=con_model_new.absorp_flux_tot
        dict_fluxes['emission_flux']=con_model_new.emission_flux
        dict_fluxes['interp_flux']=interp_flux
    if sample_all:
        samp_select=samp[:-len(scales)]
        return dict_fluxes, np.append(np.append(samp_select,scales),list_mol_results),dust_mass_ar,dust_mass_absorp_ar
    elif fit_gas_only:
        return dict_fluxes, np.append(samp,list_mol_results), [] , []
print('The next step takes a while')




parallel=True
if parallel:

    pool =  mp.get_context('fork').Pool(min(int(16),mp.cpu_count()))
    if sample_all or fit_gas_only:
        results = [pool.apply_async(get_full_model, args=(i,1)) for i in range(len(samples))]
        pool.close() 

        
    else:
        obs_per_model=1 # how many scatter observations are calculated per model
        results = [pool.apply_async(get_scales_parallel, args=(i,obs_per_model)) for i in range(len(samples))]
        pool.close() 
else:
    if sample_all or fit_gas_only:
        results = [get_full_model(i,1) for i in range(len(samples))]    

    else:
        obs_per_model=2 # how many scatter observations are calculated per model
        results = [get_scales_parallel(i,obs_per_model) for i in range(len(samples))]    

array_flux=[]
stellar_components=[]
rim_components=[]
midplane_components=[]
surface_components=[]
absorp_components=[]
emission_components=[]
tot_samples=[]
interp_fluxes=[]
array_flux=[]
tot_samples=[]
dust_mass_master_ar=[]



dict_individual_flux={}
dict_individual_flux_absorp={}
dust_mass_master_ar=[]
dust_mass_absorp_master_ar=[]

if scatter_obs:
    print('SCATTER_OBS has been removed!!!!!!')
            
else:
    for i in range(len(results)):
        if parallel:
            dict_flux,samp,dust_mass_ar,dust_mass_absorp_ar=results[i].get()
        else:
            dict_flux,samp,dust_mass_ar,dust_mass_absorp_ar=np.array(results,dtype='object')[i,j,0],np.array(results,dtype='object')[i,j,1].flatten(),np.array(results,dtype='object')[i,j,2].flatten()
        if plot_dust_individual:
            if use_dust_emis:
                for key in dict_flux['individual_surface']:
                    if key not in dict_individual_flux:
                        dict_individual_flux[key]=[]
                    dict_individual_flux[key].append(dict_flux['individual_surface'][key])            
            if use_dust_absorp:
                for key in dict_flux['individual_absorp']:
                    key_abs=key[:-7]
                    if key_abs not in dict_individual_flux_absorp:
                        dict_individual_flux_absorp[key_abs]=[]
                    dict_individual_flux_absorp[key_abs].append(dict_flux['individual_absorp'][key])
        
        if not ignore_spectrum_plot:
            array_flux.append(dict_flux['tot_flux'])
            if not fit_gas_only:
                stellar_components.append(dict_flux['stellar_flux'])
                rim_components.append(dict_flux['rim_flux']+dict_flux['stellar_flux'])
                midplane_components.append(dict_flux['midplane_flux'])
                if use_dust_emis:
                    surface_components.append(dict_flux['surface_flux'])
                if use_dust_absorp:
                    absorp_components.append(dict_flux['absorp_flux'])
            emission_components.append(dict_flux['emission_flux'])
            interp_fluxes.append(dict_flux['interp_flux'])
        tot_samples.append(samp)
        if not fit_gas_only:
            if use_dust_emis:
                dust_mass_master_ar.append(dust_mass_ar)
            if use_dust_absorp:
                dust_mass_absorp_master_ar.append(dust_mass_absorp_ar)


if plot_dust_individual:
    if use_dust_emis:
        if save_flux:
            with open(f'{prefix}dict_fluxes{reduce_str}.pkl', 'wb') as f:
                pickle.dump(dict_individual_flux, f)
    if use_dust_absorp:
        if save_flux:
            with open(f'{prefix}dict_fluxes_absorp{reduce_str}.pkl', 'wb') as f:
                pickle.dump(dict_individual_flux_absorp, f)
        
if not ignore_spectrum_plot:
    array_flux=np.array(array_flux,dtype='float32')
    interp_fluxes=np.array(interp_fluxes,dtype='float32')
    stellar_components=np.array(stellar_components,dtype='float32')
    rim_components=np.array(rim_components,dtype='float32')
    midplane_components=np.array(midplane_components,dtype='float32')
    surface_components=np.array(surface_components,dtype='float32')
    absorp_components=np.array(absorp_components,dtype='float32')
    emission_components=np.array(emission_components,dtype='float32')
tot_samples=np.array(tot_samples)
dust_mass_master_ar=np.array(dust_mass_master_ar)
dust_mass_absorp_master_ar=np.array(dust_mass_absorp_master_ar)






if save_output:
    
    print('Shape tot sample',np.shape(tot_samples))
    print('Shape samples',np.shape(samples))
    np.save(f'{prefix}complete_posterior{reduce_str}',tot_samples)
    if use_dust_emis:
        np.save(f'{prefix}dust_masses{reduce_str}',dust_mass_master_ar)
    if use_dust_absorp:
        np.save(f'{prefix}dust_absorp_masses{reduce_str}',dust_mass_absorp_master_ar)
    #exit()

if 'log_sigma_obs' in complete_header:
    idx_sigma=np.where(complete_header=='log_sigma_obs')[0]
    sig_obs=flux_obs*10**np.median(tot_samples[:,idx_sigma])

elif 'sigma_obs' in complete_header:
    idx_sigma=np.where(complete_header=='sigma_obs')[0]
    sig_obs=flux_obs*np.median(tot_samples[:,idx_sigma])
elif 'log_sigma_obs_abs' in complete_header:
    idx_sigma=np.where(complete_header=='log_sigma_obs_abs')[0]
    sig_obs=10**np.median(tot_samples[:,idx_sigma])

elif 'sigma_obs_abs' in complete_header:
    idx_sigma=np.where(complete_header=='sigma_ob_abs')[0]
    sig_obs=np.median(tot_samples[:,idx_sigma])

    
if not ignore_spectrum_plot:
    if save_flux:
        
        np.save(f'{prefix}mol_flux{reduce_str}',emission_components)
        np.save(f'{prefix}array_flux{reduce_str}',array_flux)
        np.save(f'{prefix}interp_flux{reduce_str}',interp_fluxes)

    
def nicer_labels_single(lab,with_size=True):
    new_lab=''
    if 'Silica' in lab:
        new_lab+='Silica '
    elif 'Fo_Sogawa' in lab or 'Forsterite' in lab or 'Fo_Zeidler' in lab:
        new_lab+='Forsterite '
    elif 'En_Jaeger' in lab or 'Enstatite' in lab:
        new_lab+='Enstatite  '
    elif 'Mgolivine' in lab or 'MgOlivine' in lab :
        new_lab+='Am Mgolivine '
    elif 'Olivine' in lab:
        new_lab+='Olivine '
    elif 'Mgpyroxene' in lab or 'MgPyroxene' in lab:
        new_lab+='Am Mgpyroxene '
    elif 'Pyroxene' in lab:
        new_lab+='Pyroxene '
    elif 'Fayalite' in lab:
        new_lab+='Fayalite '

    if with_size:
        idx=lab.find('_rv')
        rv=lab[idx+3:idx+6]
        new_lab+=rv
    return new_lab    
    

if not ignore_spectrum_plot:    
    min_wave=0
    max_wave=' '
    def plot_model_uncertainties_names(flux_obs,sig_obs,wave_obs,y_predict_set,model_wave,folder,save=False,
                                       save_name='',min_wave=min_wave,ylim='',max_wave=max_wave,zoom=True,
                                       obs_as_line=True, zoom_in_list=[[5,30]],
                                       plot_components=True,stellar_components=stellar_components,
                                       rim_components=rim_components,midplane_components=midplane_components,
                                       surface_components=surface_components,absorp_components=absorp_components,emission_components=emission_components,
                                       individual_surface=dict_individual_flux, individual_absorp=dict_individual_flux_absorp,
                                       plot_individual_surface=False,number_plotted_dust=0,debug=True):
        comp_dict={}
        indi_dust_dict={}
        fig = plt.figure(figsize=(9,6))
        ax  = fig.add_subplot(1,1,1)
        xmin = 0.07
        xmax = np.amax(model_wave)*1.5

        ax.set_xscale('log')
        ax.set_xlim([xmin,xmax])
        ax.set_xlabel(r'$\lambda\rm [\mu m]$', fontsize=20)
        ax.set_ylabel(r'$F_\nu\rm [Jy]$', fontsize=20)
        ax.tick_params(labelsize=15)
        if max_wave!=' ':
            idx=np.where(model_wave<=max_wave)[0]

            x_model=model_wave[idx]
            y_model=y_predict_set[:,idx]
        else:
            x_model=model_wave
            y_model=y_predict_set
        #print(np.shape(y_model))
        y_median=np.median(y_model,axis=0)
        y_std=np.percentile(y_model,50+68/2,axis=0)
        y_2std=np.percentile(y_model,50+95/2,axis=0)
        y_3std=np.percentile(y_model,50+99.9/2,axis=0)
        y_std_min=np.percentile(y_model,50-68/2,axis=0)
        y_2std_min=np.percentile(y_model,50-95/2,axis=0)
        y_3std_min=np.percentile(y_model,50-99.9/2,axis=0)
        #print(np.shape(y_3std_min),np.shape(y_3std_min),np.shape(x_model))
        ax.fill_between(x_model,y_3std_min,y_3std,color='black',alpha=0.1,zorder=100)
        ax.fill_between(x_model,y_2std_min,y_2std,color='black',alpha=0.3,zorder=100)
        ax.fill_between(x_model,y_std_min,y_std,color='black',alpha=0.5,zorder=100)
        ax.plot(x_model,y_median,label='Model',alpha=1,color='black',zorder=100)
        min_val=np.min(y_median)
        max_val=np.max(y_median)
        if plot_components:
            comp_names=['Stellar flux','Stellar + rim flux','Midplane flux']
            if use_dust_emis:
                comp_names.append('Surface flux')
            if use_dust_absorp:
                comp_names.append('Absorption flux x (-1)'),
            if np.mean(emission_components)>=0.0:
                comp_names.append('Emission flux')
            else:
                comp_names.append('Mol. absorption flux x (-1)')
            comp_colors=['tab:orange','tab:green','tab:purple','tab:brown','tab:olive','tab:red']
            
            comp_list=[stellar_components,rim_components,midplane_components]
            
            if use_dust_emis:
                comp_list.append(surface_components)
            if use_dust_absorp:
                comp_list.append(absorp_components*(-1.0))
            if np.mean(emission_components)>=0.0:      
                comp_list.append(emission_components)
            else:      
                comp_list.append(emission_components*(-1.0))
            for idx_comp in range(len(comp_list)):
                print('Adding ',comp_names[idx_comp])
                comp=comp_list[idx_comp]
                y_median_comp=np.median(comp,axis=0)
                y_std_comp=np.percentile(comp,50+68/2,axis=0)
                y_2std_comp=np.percentile(comp,50+95/2,axis=0)
                y_3std_comp=np.percentile(comp,50+99.9/2,axis=0)
                y_std_min_comp=np.percentile(comp,50-68/2,axis=0)
                y_2std_min_comp=np.percentile(comp,50-95/2,axis=0)
                y_3std_min_comp=np.percentile(comp,50-99.9/2,axis=0)
                comp_dict[comp_names[idx_comp]]={}
                comp_dict[comp_names[idx_comp]]['median']=y_median_comp
                comp_dict[comp_names[idx_comp]]['std']=y_std_comp
                comp_dict[comp_names[idx_comp]]['std2']=y_2std_comp
                comp_dict[comp_names[idx_comp]]['std3']=y_3std_comp
                comp_dict[comp_names[idx_comp]]['std_min']=y_std_min_comp
                comp_dict[comp_names[idx_comp]]['std2_min']=y_2std_min_comp
                comp_dict[comp_names[idx_comp]]['std3_min']=y_3std_min_comp

                ax.fill_between(x_model,comp_dict[comp_names[idx_comp]]['std3_min'],comp_dict[comp_names[idx_comp]]['std3'],color=comp_colors[idx_comp],alpha=0.1)
                ax.fill_between(x_model,comp_dict[comp_names[idx_comp]]['std2_min'],comp_dict[comp_names[idx_comp]]['std2'],color=comp_colors[idx_comp],alpha=0.3)
                ax.fill_between(x_model,comp_dict[comp_names[idx_comp]]['std_min'],comp_dict[comp_names[idx_comp]]['std'],color=comp_colors[idx_comp],alpha=0.5)
                ax.plot(x_model,comp_dict[comp_names[idx_comp]]['median'],label=comp_names[idx_comp],alpha=1,color=comp_colors[idx_comp])
            if plot_individual_surface:
                dust_emission_plot=True
                if len(list(individual_surface.keys()))==0:
                    comp_keys=list(individual_absorp.keys())
                    dust_emission_plot=False
                else:
                    comp_keys=list(individual_surface.keys())
                comp_names_dust=[]
                for lab in comp_keys:
                    comp_names_dust.append(nicer_labels_single(lab))

                comp_colors_dust=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
                                '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                                '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
                comp_list_dust=[]
                max_vals_dust=[]
                for comp in comp_keys:
                    if dust_emission_plot:
                        comp_list_dust.append(individual_surface[comp])
                    else:
                        comp_list_dust.append(individual_absorp[comp]*-1.0)
                for idx_comp in range(len(comp_list_dust)):
                    comp=comp_list_dust[idx_comp]
                    y_median_comp=np.median(comp,axis=0)
                    y_std_comp=np.percentile(comp,50+68/2,axis=0)
                    y_2std_comp=np.percentile(comp,50+95/2,axis=0)
                    y_3std_comp=np.percentile(comp,50+99.9/2,axis=0)
                    y_std_min_comp=np.percentile(comp,50-68/2,axis=0)
                    y_2std_min_comp=np.percentile(comp,50-95/2,axis=0)
                    y_3std_min_comp=np.percentile(comp,50-99.9/2,axis=0)
                    indi_dust_dict[comp_names_dust[idx_comp]]={}
                    indi_dust_dict[comp_names_dust[idx_comp]]['median']=y_median_comp
                    indi_dust_dict[comp_names_dust[idx_comp]]['std']=y_std_comp
                    indi_dust_dict[comp_names_dust[idx_comp]]['std2']=y_2std_comp
                    indi_dust_dict[comp_names_dust[idx_comp]]['std3']=y_3std_comp
                    indi_dust_dict[comp_names_dust[idx_comp]]['std_min']=y_std_min_comp
                    indi_dust_dict[comp_names_dust[idx_comp]]['std2_min']=y_2std_min_comp
                    indi_dust_dict[comp_names_dust[idx_comp]]['std3_min']=y_3std_min_comp
                    max_vals_dust.append(np.max(y_median_comp))

                max_vals_dust=np.array(max_vals_dust)
                if number_plotted_dust!=0:
                    sorted_values=np.flip(np.sort(max_vals_dust))
                    comp_list_select=[]
                    for i in range(number_plotted_dust):
                        if debug:
                            print('select dust species')
                            print(i)
                            print(max_vals_dust)
                            print(sorted_values[i])

                        comp_list_select.append(np.where(max_vals_dust==sorted_values[i])[0][0])
                else:
                    comp_list_select=np.arange(0,len(comp_list_dust),1)

                idx_colour_count=0
                for idx_comp in comp_list_select:

                    if debug:
                        print('idx comp', idx_comp)
                        print('dust name',comp_names_dust[idx_comp])
                    idx_comp_color=idx_colour_count
                    while idx_comp_color>=len(comp_colors_dust):
                        idx_comp_color-=len(comp_colors_dust)
                        if debug:
                            print('Adjust color to',idx_comp_color)
                    ax.fill_between(x_model,indi_dust_dict[comp_names_dust[idx_comp]]['std3_min'],indi_dust_dict[comp_names_dust[idx_comp]]['std3'],color=comp_colors_dust[idx_comp_color],alpha=0.1)
                    ax.fill_between(x_model,indi_dust_dict[comp_names_dust[idx_comp]]['std2_min'],indi_dust_dict[comp_names_dust[idx_comp]]['std2'],color=comp_colors_dust[idx_comp_color],alpha=0.3)
                    ax.fill_between(x_model,indi_dust_dict[comp_names_dust[idx_comp]]['std_min'],indi_dust_dict[comp_names_dust[idx_comp]]['std'],color=comp_colors_dust[idx_comp_color],alpha=0.5)
                    ax.plot(x_model,indi_dust_dict[comp_names_dust[idx_comp]]['median'],label=comp_names_dust[idx_comp],alpha=1,color=comp_colors_dust[idx_comp_color])
                    idx_colour_count+=1
        sig_obs=np.array(sig_obs)

        if obs_as_line:

            print('Adding  Observation')
            ax.fill_between(wave_obs,flux_obs-sig_obs,flux_obs+sig_obs,alpha=0.5,color='tab:blue',zorder=1000)
            ax.plot(wave_obs,flux_obs,label='Observation',color='tab:blue',alpha=0.7,zorder=1000)

        else:
            ax.errorbar(wave_obs,flux_obs,yerr=sig_obs, label='Observation',alpha=0.7,zorder=1000)    

        axbox = ax.get_position()

        if ylim!='':

            ax.set_ylim(bottom=ylim[0],top=ylim[1])
        else:
            ax.set_ylim(bottom=min_val*0.9,top=max_val*1.1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        if max_wave!=' ':
            ax.set_xlim(right=max_wave)
        else:
            ax.set_xlim(left=np.min(x_model),right=np.max(x_model))
        legend = ax.legend(frameon=True,ncol=1,markerscale=0.7,scatterpoints=1,labelspacing=0.0,fancybox=True)
        for label in legend.get_texts():
            label.set_fontsize('large')

        plt.tight_layout()
        if save:
            if plot_individual_surface:
                plt.savefig(save_name+reduce_str+'_dust'+f'.{filetype_fig}',bbox_inches='tight')
            else:
                plt.savefig(save_name+reduce_str+f'.{filetype_fig}',bbox_inches='tight')
        if close_plots:
            plt.close()
        else:

            plt.show()

        if zoom:
            for zoom_in in zoom_in_list:
                print('Plotting zoomed in')
                print('at',zoom_in)

                fig = plt.figure(figsize=(9,6))
                ax  = fig.add_subplot(1,1,1)
                xmin = zoom_in[0]
                xmax = zoom_in[1]

                ax.set_xlabel(r'$\lambda\rm [\mu m]$', fontsize=20)
                ax.set_ylabel(r'$F_\nu\rm [Jy]$', fontsize=20)
                ax.tick_params(labelsize=15)
                idx=np.where(model_wave<=xmax)[0]
                idx2=np.where(model_wave[idx]>=xmin)[0]

                y_model_select=y_predict_set[:,idx[idx2]]

                max_val=np.max(y_model_select)
                min_val=np.min(y_model_select)

                if obs_as_line:

                    print('Adding  Observation')
                    ax.fill_between(wave_obs,flux_obs-sig_obs,flux_obs+sig_obs,alpha=0.5,color='tab:blue',zorder=1000)
                    ax.plot(wave_obs,flux_obs,label='Observation',color='tab:blue',alpha=0.7,zorder=1000)
                else:
                    ax.errorbar(wave_obs,flux_obs,yerr=sig_obs, label='Observation',alpha=0.7,zorder=1000)    


                print('Adding  Full model')

                ax.fill_between(x_model,y_3std_min,y_3std,color='black',alpha=0.1,zorder=100)
                ax.fill_between(x_model,y_2std_min,y_2std,color='black',alpha=0.3,zorder=100)
                ax.fill_between(x_model,y_std_min,y_std,color='black',alpha=0.5,zorder=100)
                ax.plot(x_model,y_median,label='Model',alpha=1,color='black',zorder=100)




                if plot_components:

                    for idx_comp in range(len(comp_dict)):

                        print('Adding ',comp_names[idx_comp])

                        ax.fill_between(x_model,comp_dict[comp_names[idx_comp]]['std3_min'],comp_dict[comp_names[idx_comp]]['std3'],color=comp_colors[idx_comp],alpha=0.1)
                        ax.fill_between(x_model,comp_dict[comp_names[idx_comp]]['std2_min'],comp_dict[comp_names[idx_comp]]['std2'],color=comp_colors[idx_comp],alpha=0.3)
                        ax.fill_between(x_model,comp_dict[comp_names[idx_comp]]['std_min'],comp_dict[comp_names[idx_comp]]['std'],color=comp_colors[idx_comp],alpha=0.5)
                        ax.plot(x_model,comp_dict[comp_names[idx_comp]]['median'],label=comp_names[idx_comp],alpha=1,color=comp_colors[idx_comp])
                if plot_individual_surface:
                    idx_colour_count=0
                    for idx_comp in comp_list_select:

                        idx_comp_color=idx_colour_count
                        while idx_comp_color>=len(comp_colors_dust):
                            idx_comp_color-=len(comp_colors_dust)
                            if debug:
                                print('Adjust color to',idx_comp_color)

                        ax.fill_between(x_model,indi_dust_dict[comp_names_dust[idx_comp]]['std3_min'],indi_dust_dict[comp_names_dust[idx_comp]]['std3'],color=comp_colors_dust[idx_comp_color],alpha=0.1)
                        ax.fill_between(x_model,indi_dust_dict[comp_names_dust[idx_comp]]['std2_min'],indi_dust_dict[comp_names_dust[idx_comp]]['std2'],color=comp_colors_dust[idx_comp_color],alpha=0.3)
                        ax.fill_between(x_model,indi_dust_dict[comp_names_dust[idx_comp]]['std_min'],indi_dust_dict[comp_names_dust[idx_comp]]['std'],color=comp_colors_dust[idx_comp_color],alpha=0.5)
                        ax.plot(x_model,indi_dust_dict[comp_names_dust[idx_comp]]['median'],label=comp_names_dust[idx_comp],alpha=1,color=comp_colors_dust[idx_comp_color])
                        idx_colour_count+=1
                axbox = ax.get_position()

                #ax.set_xscale('log')
                #ax.set_yscale('log')
                ax.set_xlim(left=xmin,right=xmax)
                ax.set_ylim(bottom=min_val*0.9,top=max_val*1.1)
                legend = ax.legend(frameon=True,ncol=1,markerscale=0.7,scatterpoints=1,labelspacing=0.0,fancybox=True)
                for label in legend.get_texts():
                    label.set_fontsize('large')

                plt.tight_layout()
                if save:
                    if plot_individual_surface:
                        plt.savefig(f'{save_name}_zoom_{str(zoom_in[0])}_{str(zoom_in[1])}{reduce_str}_dust.{filetype_fig}',bbox_inches='tight')
                    else:
                        plt.savefig(f'{save_name}_zoom_{str(zoom_in[0])}_{str(zoom_in[1])}{reduce_str}.{filetype_fig}',bbox_inches='tight')

                if close_plots:
                    plt.close()
                else:

                    plt.show()



    if 'sigma_obs'in fixed_dict:
        sig_obs=fixed_dict['sigma_obs']*flux_obs
        
    elif 'log_sigma_obs'in fixed_dict:
        sig_obs=10**fixed_dict['log_sigma_obs']*flux_obs
    elif 'sigma_obs'in fixed_dict:
        sig_obs=fixed_dict['sigma_obs_abs']
        
    elif 'sigma_obs'in fixed_dict:
        sig_obs=10**fixed_dict['log_sigma_obs_abs']
    else:
        sig_obs=np.zeros_like(flux_obs)
        print('Plot sig obs as 0')


    print('Plotting component plot...')
    print('Zoom list',zoom_list)

    if plot_dust_individual:
        print('Plotting dust version of the component plot')
        plot_model_uncertainties_names(flux_obs,sig_obs,lam_obs,array_flux,con_model_new.xnew,
                                   'folder',zoom_in_list=zoom_list,save=True,
                                       save_name=save_folder+str(run_number)+'_Component_plot',
                                       min_wave=min_wave,ylim='',max_wave=max_wave,
                                       individual_surface=dict_individual_flux,
                                       plot_individual_surface=True,number_plotted_dust=number_plotted_dust)

    if fit_gas_only:
        plot_model_uncertainties_names(flux_obs,sig_obs,lam_obs,array_flux,con_model_new.xnew,
                                   'folder', zoom_in_list=zoom_list, save=True, save_name=save_folder+str(run_number)+'_Component_plot', min_wave=min_wave,ylim='',max_wave=max_wave,plot_components=False)

    else:
        plot_model_uncertainties_names(flux_obs,sig_obs,lam_obs,array_flux,con_model_new.xnew,
                                   'folder', zoom_in_list=zoom_list, save=True, save_name=save_folder+str(run_number)+'_Component_plot', min_wave=min_wave,ylim='',max_wave=max_wave)




    print('...Saved')        


if not ignore_spectrum_plot:
    print('Plotting the residuals..')
    def plot_residual(flux_obs,sig_obs,wave_obs,interp_fluxes,folder,save=False,
                                       save_name='',percent=True):
        # calculate the residual
        diff=[]
        for i in range(len(interp_fluxes)):
            if percent:
                diff.append(((interp_fluxes[i]-flux_obs)/interp_fluxes[i])*100) # difference in percent
            else:

                diff.append((interp_fluxes[i]-flux_obs)) # difference in Jy


        y_median=np.median(diff,axis=0)
        y_std=np.percentile(diff,50+68/2,axis=0)
        y_2std=np.percentile(diff,50+95/2,axis=0)
        y_3std=np.percentile(diff,50+99.9/2,axis=0)
        y_std_min=np.percentile(diff,50-68/2,axis=0)
        y_2std_min=np.percentile(diff,50-95/2,axis=0)
        y_3std_min=np.percentile(diff,50-99.9/2,axis=0)


        fig = plt.figure(figsize=(9,6))
        ax  = fig.add_subplot(1,1,1)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\lambda\rm [\mu m]$', fontsize=20)
        if percent:
            ax.set_ylabel(r'$(F_{\rm model}-F_{\rm obs})/F_{\rm model}\cdot100 \, \rm [\%]$', fontsize=20)

        else:
            ax.set_ylabel(r'$F_{\rm model}- F_{\rm obs} \, \rm [Jy]$', fontsize=20)
        ax.tick_params(labelsize=15)

        ax.plot(wave_obs,np.zeros(len(wave_obs)),label='Perfect prediction')
        if not percent:
            ax.fill_between(wave_obs,sig_obs,-sig_obs,color='tab:blue',alpha=0.5,label=r'$1\sigma_{\rm obs}$')

        ax.fill_between(wave_obs,y_3std_min,y_3std,color='tab:orange',alpha=0.1)
        ax.fill_between(wave_obs,y_2std_min,y_2std,color='tab:orange',alpha=0.3)
        ax.fill_between(wave_obs,y_std_min,y_std,color='tab:orange',alpha=0.5)
        ax.plot(wave_obs,y_median,label='Residual',alpha=1,color='tab:orange')


        axbox = ax.get_position()
        legend = ax.legend(frameon=True,ncol=2,markerscale=0.7,scatterpoints=1,labelspacing=0.0,loc=(axbox.x0-0.02, axbox.y0 - 0.02),fancybox=True)
        for label in legend.get_texts():
            label.set_fontsize('large')

        plt.tight_layout()
        if save:
            plt.savefig(save_name,bbox_inches='tight')
        if close_plots:
            plt.close()
        else:

            plt.show()




    if not fit_gas_only:
        plot_residual(flux_obs,sig_obs,lam_obs,interp_fluxes,'folder',
                      save=True,save_name=save_folder+str(run_number)+f'_Residual_percent{reduce_str}.{filetype_fig}',percent=True)

    plot_residual(flux_obs,sig_obs,lam_obs,interp_fluxes,'folder',
                  save=True,save_name=save_folder+str(run_number)+f'_Residual_jansky{reduce_str}.{filetype_fig}',percent=False)


    print('..Saved')




print('Plotting cornerplot multinest...')
lims=[]
for i in range(len(header_para)):
    lims.append([lower_lim[i],upper_lim[i]])  


CORNER_KWARGS = dict(
    smooth=.9,
    label_kwargs=dict(fontsize=20,rotation=45),
    title_kwargs=dict(fontsize=24,loc='left'),
    levels=[0.68, 0.95],
    quantiles=[0.5],
    plot_density=False,
    plot_datapoints=True,
    fill_contours=True,
    plot_contours=True,
    show_titles=True,
    title_quantiles=[0.16,0.5,0.84],
    title_fmt='10.4e',
#        truths=val_obj_conv,
    range=lims)
fig = corner.corner(samples[:,:len(header_para)], labels=header_para, color='tomato', **CORNER_KWARGS)

plt.savefig(f'{save_folder}{str(run_number)}_Cornerplot_multinest_parameters{reduce_str}.{filetype_fig}',bbox_inches='tight')
if close_plots:
    plt.close()
else:

    plt.show()


header_para_slab=list(header_para)+list(header_slab)
if fit_conti_err or fit_obs_err:
    header_para_slab+=list(header_sigma)
    
lims=[]
for i in range(len(header_para_slab)):
    lims.append([lower_lim[i],upper_lim[i]])
print(len(lims))
print(np.shape(samples[:,:len(header_para_slab)]))



CORNER_KWARGS = dict(
    smooth=.9,
    label_kwargs=dict(fontsize=20,rotation=45),
    title_kwargs=dict(fontsize=24,loc='left'),
    levels=[0.68, 0.95],
    quantiles=[0.5],
    plot_density=False,
    plot_datapoints=True,
    fill_contours=True,
    plot_contours=True,
    show_titles=True,
    title_quantiles=[0.16,0.5,0.84],
    title_fmt='10.4e',
#        truths=val_obj_conv,
    range=lims)
fig = corner.corner(samples[:,:len(header_para_slab)], labels=header_para_slab, color='tomato', **CORNER_KWARGS)

plt.savefig(f'{save_folder}{str(run_number)}_Cornerplot_multinest_parameters_all{reduce_str}.{filetype_fig}',bbox_inches='tight')
if close_plots:
    plt.close()
else:

    plt.show()

print('...Saved')

def nicer_labels(init_abundance=init_abundance,with_size=True):
    labels=list(init_abundance.keys())
    new_labels=[]
    for lab in labels:
        new_lab=''
        if 'Silica' in lab:
            new_lab+='Silica '
        elif 'Fo_Sogawa' in lab or 'Forsterite' in lab or 'Fo_Zeidler' in lab:
            new_lab+='Forsterite '
        elif 'En_Jaeger' in lab or 'Enstatite' in lab:
            new_lab+='Enstatite  '
        elif 'Mgolivine' in lab or 'MgOlivine' in lab :
            new_lab+='Am Mg-olivine '
        elif 'Olivine' in lab:
            new_lab+='Olivine '
        elif 'Mgpyroxene' in lab or 'MgPyroxene' in lab:
            new_lab+='Am Mg-pyroxene '
        elif 'Pyroxene' in lab:
            new_lab+='Pyroxene '
        elif 'Fayalite' in lab:
            new_lab+='Fayalite '
            
        if with_size:
            idx=lab.find('_rv')
            rv=lab[idx+3:idx+6]
            new_lab+=rv
        new_labels.append(new_lab)
    return new_labels

def set_slab_labels(slab_prior_dict):
    new_labels=[]
    for key in slab_prior_dict:
        if 'radius' not in slab_prior_dict[key] and 'log_radius' not in slab_prior_dict[key]:
            label=key+': radius'
            new_labels.append(label)
    return new_labels
slab_labels=set_slab_labels(slab_prior_dict=slab_prior_dict)

if not fit_gas_only:
    header_all=list(header_para_slab)+['sc_ir']+['sc_mid']
    ugly_header=list(header_para_slab)+['sc_ir']+['sc_mid']
else:
    
    header_all=list(header_para_slab)
    ugly_header=list(header_para_slab)

if use_dust_emis:
    nicer_labels_output=nicer_labels(init_abundance=init_abundance)
    for lab in nicer_labels_output:
        header_all.append(lab)
    
    for key in init_abundance:
        ugly_header.append(key)
if use_dust_absorp:
    nicer_labels_output_absorp=nicer_labels(init_abundance=init_abundance_absorp)

    for lab in nicer_labels_output_absorp:
        header_all.append(lab+'_absorp')
    
    for key in init_abundance_absorp:
        ugly_header.append(key+'_absorp')        
if not fit_gas_only:
    header_all=header_all+slab_labels
    ugly_header=ugly_header+slab_labels


if sample_all:
    header_all=list(header_para_slab)+['sc_ir']+['sc_mid']
    ugly_header=list(header_para_slab)+['sc_ir']+['sc_mid']
    if use_dust_emis:   
        for lab in nicer_labels_output:
            header_all.append(lab)
        for key in init_abundance:
            ugly_header.append(key)
    if use_dust_absorp:        
        for lab in nicer_labels_output_absorp:
            header_all.append(lab+'_absorp')
        for key in init_abundance_absorp:
            ugly_header.append(key+'_absorp')
    


'''
Converting the abundances from meaningless values to relative mass fractions
If the curves are already given in Kappa, you can set q_curves=False
'''
if fit_gas_only:
    tot_samples_rel=tot_samples


if use_dust_emis:
    
    q_curves=True
    # converting the scale factors in 
    if q_curves:
        factor_dict={}
        for key in init_abundance:
    
            with open(dust_path+key,'r') as f:
                lines=f.readlines()
            old_data=True
            for line in lines:
                if 'density' in line:
                    dens=line.split()[3]
                    old_data=False
                    break
             
            idx_rv=key.find('rv')
            rad=key[idx_rv+2:-4]
            if old_data:
                with open(dust_path+key,'r') as f:
                    rad,dens=f.readline().split()[1:3]
            #print(key,rad,dens)
            rad=float(rad)
            dens=float(dens)
            fact=dens*rad
            factor_dict[key]=fact
            for i in range(len(header_all)):
                if header_all[i]==nicer_labels({key:None})[0]:
                    break
            tot_samples[:,i]=tot_samples[:,i]*factor_dict[key]
    
    #getting all indices that have abundances
    idxs=[]
    for i in range(len(nicer_labels_output)):
        idx=np.where(np.array(header_all)==nicer_labels_output[i])[0][0]                                
        idxs.append(idx)
    #print(idxs)    
    
    tot_samples_rel=tot_samples.copy()
    for i in range(len(tot_samples)):
        abund=tot_samples[i,idxs].copy()
        tot=np.sum(abund)
        if tot==0.0:
            rel_abund=0.0
        else:        
            rel_abund=abund/tot
        tot_samples_rel[i,idxs]=rel_abund

if use_dust_absorp:
    q_curves=True
    # converting the scale factors in 
    if q_curves:
        factor_dict={}
        for key in init_abundance_absorp:
    
            with open(dust_path+key,'r') as f:
                lines=f.readlines()
            old_data=True
            for line in lines:
                if 'density' in line:
                    dens=line.split()[3]
                    old_data=False
                    break
             
            idx_rv=key.find('rv')
            rad=key[idx_rv+2:-4]
            if old_data:
                with open(dust_path+key,'r') as f:
                    rad,dens=f.readline().split()[1:3]
            #print(key,rad,dens)
            rad=float(rad)
            dens=float(dens)
            fact=dens*rad
            factor_dict[key]=fact
            for i in range(len(header_all)):
                if header_all[i]==nicer_labels({key:None})[0]+'_absorp':
                    break
            tot_samples[:,i]=tot_samples[:,i]*factor_dict[key]
    
    #getting all indices that have abundances
    idxs_absorp=[]
    for i in range(len(nicer_labels_output_absorp)):
        idx=np.where(np.array(header_all)==nicer_labels_output_absorp[i]+'_absorp')[0][0]                                
        idxs_absorp.append(idx)
    print(idxs_absorp)    
    
    tot_samples_rel=tot_samples.copy()
    for i in range(len(tot_samples)):
        abund=tot_samples[i,idxs_absorp].copy()
        tot=np.sum(abund)
        if tot==0.0:
            rel_abund=0.0
        else:        
            rel_abund=abund/tot
        tot_samples_rel[i,idxs_absorp]=rel_abund
        
'''
Histogram of the dust abundances
'''
print('Plotting histograms...')
if use_dust_emis:
    dust_analysis={}
    for idx in idxs:
        dust_array=tot_samples_rel[:,idx]
        median=np.median(dust_array)
        plus_std=np.percentile(dust_array,50+68/2)
        minus_std=np.percentile(dust_array,50-68/2)
        dust_analysis[ugly_header[idx]]=[median,plus_std,minus_std]
if use_dust_absorp:
    dust_analysis_absorp={}
    for idx in idxs_absorp:
        dust_absorp_array=tot_samples_rel[:,idx]
        median=np.median(dust_absorp_array)
        plus_std=np.percentile(dust_absorp_array,50+68/2)
        minus_std=np.percentile(dust_absorp_array,50-68/2)
        dust_analysis_absorp[ugly_header[idx]]=[median,plus_std,minus_std]

if use_dust_emis:
    dust_analysis_abs={}
    dust_fraction_used={}
    tot_model_number=len(dust_mass_master_ar)
    for i in range(len(idxs)):
        dust_array=dust_mass_master_ar[:,i]
        
        median=np.median(dust_array)
        plus_std=np.percentile(dust_array,50+68/2)
        minus_std=np.percentile(dust_array,50-68/2)
        dust_analysis_abs[ugly_header[idxs[i]]]=[median,plus_std,minus_std]
        
        
        idx_used=np.where(dust_mass_master_ar[:,i]!=0.0)[0]
        dust_fraction_used[ugly_header[idxs[i]]]=len(idx_used)/tot_model_number
if use_dust_absorp:
    dust_analysis_abs_absorp={}
    dust_fraction_used_absorp={}
    tot_model_number=len(dust_mass_absorp_master_ar)
    for i in range(len(idxs_absorp)):
        dust_absorp_array=dust_mass_absorp_master_ar[:,i]
        
        median=np.median(dust_absorp_array)
        plus_std=np.percentile(dust_absorp_array,50+68/2)
        minus_std=np.percentile(dust_absorp_array,50-68/2)
        dust_analysis_abs_absorp[ugly_header[idxs_absorp[i]]]=[median,plus_std,minus_std]
        
        
        idx_used=np.where(dust_mass_absorp_master_ar[:,i]!=0.0)[0]
        dust_fraction_used_absorp[ugly_header[idxs_absorp[i]]]=len(idx_used)/tot_model_number
        
    

def plot_histograms(dust_analysis,scale='linear',suffix='',indicate_regions=True,plot_legend=False,debug=False):
    colour_list=['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    colour_count=0
    medians=[]
    plus_stds=[]
    minus_stds=[]
    first=True
    i=0
    
    plt.figure(figsize=(12,6))
    for key_run in dust_analysis:
        key=key_run[:-7]
        if debug:
            print(key)
        new_dust=False
        if first:
            key_old=key
        if not first and key[:5]!=key_old[:5]:
            new_dust=True
        if debug:
            print(first,new_dust)
        if new_dust:
            x_range=np.arange(len(medians))+i-len(medians)+1
            idx_key=key_old.find('rv')
            label=nicer_labels({key_old:None},with_size=False)[0]
            if debug:
                print('Plotting..')
                print(i)
                print(key_old)
                print(x_range)
                print(medians)
                print(plus_stds)
                print(minus_stds)
                print(key_old[:idx_key])
                print(label)
            
            
            while colour_count>=len(colour_list):
                colour_count=colour_count-len(colour_list)
            if indicate_regions:
                plt.fill_between([x_range[0]-0.5,x_range[-1]+0.5],
                                 y1=[1,1],y2=[0.95,0.95],alpha=0.7,color=colour_list[colour_count])
                if not plot_legend:
                    plt.text(x_range[0]-0.4,0.96,label)

            plt.bar(x_range, medians,label=label,color=colour_list[colour_count])
            

            plt.errorbar(x_range, medians,yerr=[minus_stds,plus_stds],color='black',linestyle='',capsize=2)
            colour_count+=1    
            new_dust=False  
            medians=[]
            plus_stds=[]
            minus_stds=[]
        key_old=key
        if first:  first=False
        medians.append(dust_analysis[key_run][0])
        plus_stds.append(dust_analysis[key_run][1]-dust_analysis[key_run][0])
        minus_stds.append(dust_analysis[key_run][0]-dust_analysis[key_run][2])
        i+=1

    x_range=np.arange(len(medians))+i-len(medians)+1
    idx_key=key_old.find('rv')
    label=nicer_labels({key_old:None},with_size=False)[0]
    if debug:
        print('Plotting..')
        print(key_old)
        print(x_range)
        print(medians)
        print(key_old[:idx_key])
        print(label)

    if indicate_regions:
        plt.fill_between([x_range[0]-0.5,x_range[-1]+0.5],
                         y1=[1,1],y2=[0.95,0.95],alpha=0.7,color=colour_list[colour_count])
        if not plot_legend:
            plt.text(x_range[0]-0.4,0.96,label)
            
    plt.bar(x_range, medians,label=label,color=colour_list[colour_count])
    plt.errorbar(x_range, medians,yerr=[minus_stds,plus_stds],color='black',linestyle='',capsize=2)
    rvs=[]
    count=0
    for key in dust_analysis:
        if debug:
            print(key)
        idx=key.find('rv')
        if '_absorp' in key:
            idx_end=key.find('_absorp')
        else:
            idx_end=-3
            
        rv=key[idx+2:idx+5]
        rvs.append(rv)
        count+=1
    if debug:
        print(rvs)
        print(np.arange(1,count+1))
    plt.xticks(np.arange(1,count+1),rvs)
    if scale=='log':
        plt.yscale('log')
    plt.xlabel(r'Grain size ($\mu m$)')
    plt.ylabel('Mass fraction')
    if scale=='linear':
        plt.ylim(bottom=0,top=1)
    else:
        plt.ylim(top=1)
        
    if plot_legend:
        plt.legend()
    if scale=='log':
        plt.savefig(f'{save_folder}{str(run_number)}_histogram_mass_fractions_log{suffix}.{filetype_fig}',bbox_inches='tight')
    else:
        plt.savefig(f'{save_folder}{str(run_number)}_histogram_mass_fractions{suffix}.{filetype_fig}',bbox_inches='tight')
        
    plt.show()



def plot_histograms_abs(dust_analysis_abs,scale='linear',suffix='',indicate_regions=True,plot_legend=False,debug=False):
    colour_list=['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    colour_count=0
    medians=[]
    plus_stds=[]
    minus_stds=[]
    first=True
    i=0
    
    max_val=np.max(list(dust_analysis_abs.values()))*1.1
    min_val=np.min(list(dust_analysis_abs.values()))*0.9
    
    plt.figure(figsize=(12,6))
    for key in dust_analysis_abs:
        if debug:
            print(key)
        new_dust=False
        if first:
            key_old=key
        if not first and key[:5]!=key_old[:5]:
            new_dust=True
        if debug:
            print(first,new_dust)
        if new_dust:
            x_range=np.arange(len(medians))+i-len(medians)+1
            idx_key=key_old.find('rv')
            label=nicer_labels({key_old:None},with_size=False)[0]
            if debug:
                print('Plotting..')
                print(i)
                print(key_old)
                print(x_range)
                print(medians)
                print(plus_stds)
                print(minus_stds)
                print(key_old[:idx_key])
                print(label)
            
            
            while colour_count>=len(colour_list):
                colour_count=colour_count-len(colour_list)
            if indicate_regions:
                plt.fill_between([x_range[0]-0.5,x_range[-1]+0.5],
                                 y1=[max_val,max_val],y2=[max_val-(max_val-min_val)*0.05,max_val-(max_val-min_val)*0.05],alpha=0.7,color=colour_list[colour_count])
                if not plot_legend:
                    plt.text(x_range[0]-0.4,max_val-(max_val-min_val)*0.04,label)
            plt.bar(x_range, medians,label=label,color=colour_list[colour_count])
            

            plt.errorbar(x_range, medians,yerr=[minus_stds,plus_stds],color='black',linestyle='',capsize=2)
            colour_count+=1    
            new_dust=False  
            medians=[]
            plus_stds=[]
            minus_stds=[]
        key_old=key
        if first:  first=False
        medians.append(dust_analysis_abs[key][0])
        plus_stds.append(dust_analysis_abs[key][1]-dust_analysis_abs[key][0])
        minus_stds.append(dust_analysis_abs[key][0]-dust_analysis_abs[key][2])
        i+=1

    x_range=np.arange(len(medians))+i-len(medians)+1
    idx_key=key_old.find('rv')
    label=nicer_labels({key_old:None},with_size=False)[0]
    if debug:
        print('Plotting..')
        print(key_old)
        print(x_range)
        print(medians)
        print(key_old[:idx_key])
        print(label)

    if indicate_regions:
        plt.fill_between([x_range[0]-0.5,x_range[-1]+0.5],
                         y1=[max_val,max_val],y2=[max_val-(max_val-min_val)*0.05,max_val-(max_val-min_val)*0.05],alpha=0.7,color=colour_list[colour_count])
        if not plot_legend:
            plt.text(x_range[0]-0.4,max_val-(max_val-min_val)*0.04,label)
    plt.bar(x_range, medians,label=label,color=colour_list[colour_count])
    plt.errorbar(x_range, medians,yerr=[minus_stds,plus_stds],color='black',linestyle='',capsize=2)
    rvs=[]
    count=0
    for key in dust_analysis_abs:
        if debug:
            print(key)
        idx=key.find('rv')
        if '_absorp' in key:
            idx_end=key.find('_absorp')
        else:
            idx_end=-3
        rv=key[idx+2:idx+5]
            
        rvs.append(rv)
        count+=1
    if debug:
        print(rvs)
        print(np.arange(1,count+1))
    plt.xticks(np.arange(1,count+1),rvs)
    if scale=='log':
        plt.yscale('log')
    plt.xlabel(r'Grain size ($\mu m$)')
    plt.ylabel(r'$M_{\rm dust, thin} [\rm M_{\rm sun}]$')
    plt.ylim(bottom=min_val,top=max_val)
    if scale=='log':
        plt.yscale('log')
    if plot_legend:  
        plt.legend()
    if scale=='log':
        plt.savefig(f'{save_folder}{str(run_number)}_histogram_mass_abs_log{suffix}.{filetype_fig}',bbox_inches='tight')
    else:
        plt.savefig(f'{save_folder}{str(run_number)}_histogram_mass_abs{suffix}.{filetype_fig}',bbox_inches='tight')
        
    plt.show()



if use_dust_emis:
    plot_histograms(dust_analysis=dust_analysis,scale='linear')
    plot_histograms_abs(dust_analysis_abs=dust_analysis_abs,scale='linear')
    #plot_histograms(dust_analysis=dust_analysis,scale='log')

if use_dust_absorp:
    plot_histograms(dust_analysis=dust_analysis_absorp,scale='linear',suffix='_absorp',debug=False)
    plot_histograms_abs(dust_analysis_abs=dust_analysis_abs_absorp,scale='linear',suffix='_absorp')

def plot_histograms_both(dust_analysis,dust_analysis_abs,dust_fraction_used,suffix='',scale='linear',scale2='linear',indicate_regions=True,debug=False):
    colour_list=['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    colour_count=0
    medians=[]
    plus_stds=[]
    minus_stds=[]
    used_fract=[]
    

    custom_lines = []
    custom_labels= []
    first=True
    i=0
    
    fig,ax = plt.subplots(figsize=(12,6))
    for key_run in dust_analysis:
        key=key_run[:-7]
        if debug:
            print(key)
        new_dust=False
        if first:
            key_old=key
        if not first and key[:5]!=key_old[:5]:
            new_dust=True
        if debug:
            print(first,new_dust)
        if new_dust:
            x_range=np.arange(len(medians))+i-len(medians)+1
            idx_key=key_old.find('rv')
            label=nicer_labels({key_old:None},with_size=False)[0]
            if debug:
                print('Plotting..')
                print(i)
                print(key_old)
                print(x_range)
                print(medians)
                print(plus_stds)
                print(minus_stds)
                print(key_old[:idx_key])
                print(label)
                print(len(x_range),len(medians),len(used_fract))
            
            
            while colour_count>=len(colour_list):
                colour_count=colour_count-len(colour_list)
            if indicate_regions:
                ax.fill_between([x_range[0]-0.5,x_range[-1]+0.5],
                                 y1=[1,1],y2=[0.95,0.95],alpha=0.7,color=colour_list[colour_count])
            ax.bar(x_range, medians,label=label,color=colour_list[colour_count],align='edge',width=-0.48,
                   linewidth=1,linestyle='dashed',edgecolor='black')
            
            ax.scatter(x_range,used_fract,marker='o',c=colour_list[colour_count],edgecolor='black')
            ax.errorbar(x_range-0.24, medians,yerr=[minus_stds,plus_stds],color='black',linestyle='',capsize=2)
            custom_lines.append(Line2D([0], [0], color=colour_list[colour_count], lw=4))
            custom_labels.append(label)
            colour_count+=1    
            new_dust=False  
            medians=[]
            plus_stds=[]
            minus_stds=[]
            used_fract=[]
        key_old=key
        if first:  first=False
        medians.append(dust_analysis[key_run][0])
        plus_stds.append(dust_analysis[key_run][1]-dust_analysis[key_run][0])
        minus_stds.append(dust_analysis[key_run][0]-dust_analysis[key_run][2])
        used_fract.append(dust_fraction_used[key_run])
        i+=1

    x_range=np.arange(len(medians))+i-len(medians)+1
    idx_key=key_old.find('rv')
    label=nicer_labels({key_old:None},with_size=False)[0]
    if debug:
        print('Plotting..')
        print(key_old)
        print(x_range)
        print(medians)
        print(key_old[:idx_key])
        print(label)

    if indicate_regions:
        ax.fill_between([x_range[0]-0.5,x_range[-1]+0.5],
                         y1=[1,1],y2=[0.95,0.95],alpha=0.7,color=colour_list[colour_count])
    ax.bar(x_range, medians,label=label,color=colour_list[colour_count],align='edge',width=-0.48,
           linewidth=1,linestyle='dashed',edgecolor='black')
    ax.errorbar(x_range-0.24, medians,yerr=[minus_stds,plus_stds],color='black',linestyle='',capsize=2)
    ax.scatter(x_range,used_fract,marker='o',c=colour_list[colour_count],edgecolor='black')
            
    custom_lines.append(Line2D([0], [0], color=colour_list[colour_count], lw=4))
    custom_labels.append(label)
    if scale=='log':
        plt.yscale('log')
    ax.set_xlabel(r'Grain size [$\mu m$]')
    ax.set_ylabel('$f_{\mathrm{mass}}$ \n $f_{ \mathrm{model}}$')
    if scale=='linear':
        ax.set_ylim(bottom=0,top=1)
    else:
        ax.set_ylim(top=1)
        
        
        
    ax2=ax.twinx()   
    colour_count=0
    medians=[]
    plus_stds=[]
    minus_stds=[]
    first=True
    i=0
    
    max_val=np.max(list(dust_analysis_abs.values()))*1.1
    min_val=np.min(list(dust_analysis_abs.values()))*0.9
    
    for key_run in dust_analysis_abs:
        key=key_run[:-7]
        if debug:
            print(key)
        new_dust=False
        if first:
            key_old=key
        if not first and key[:5]!=key_old[:5]:
            new_dust=True
        if debug:
            print(first,new_dust)
        if new_dust:
            x_range=np.arange(len(medians))+i-len(medians)+1
            idx_key=key_old.find('rv')
            label=nicer_labels({key_old:None},with_size=False)[0]
            if debug:
                print('Plotting..')
                print(i)
                print(key_old)
                print(x_range)
                print(medians)
                print(plus_stds)
                print(minus_stds)
                print(key_old[:idx_key])
                print(label)
            
            
            while colour_count>=len(colour_list):
                colour_count=colour_count-len(colour_list)
            ax2.bar(x_range, medians,color=colour_list[colour_count],align='edge',width=0.48,
           linewidth=1,linestyle='dotted',edgecolor='black')
            

            ax2.errorbar(x_range+0.24, medians,yerr=[minus_stds,plus_stds],color='black',linestyle='',capsize=2)
            colour_count+=1    
            new_dust=False  
            medians=[]
            plus_stds=[]
            minus_stds=[]
        key_old=key
        if first:  first=False
        medians.append(dust_analysis_abs[key_run][0])
        plus_stds.append(dust_analysis_abs[key_run][1]-dust_analysis_abs[key_run][0])
        minus_stds.append(dust_analysis_abs[key_run][0]-dust_analysis_abs[key_run][2])
        i+=1

    x_range=np.arange(len(medians))+i-len(medians)+1
    idx_key=key_old.find('rv')
    label=nicer_labels({key_old:None},with_size=False)[0]
    if debug:
        print('Plotting..')
        print(key_old)
        print(x_range)
        print(medians)
        print(key_old[:idx_key])
        print(label)

    ax2.bar(x_range, medians,color=colour_list[colour_count],align='edge',width=0.48,
           linewidth=1,linestyle='dotted',edgecolor='black')
    ax2.errorbar(x_range+0.24, medians,yerr=[minus_stds,plus_stds],color='black',linestyle='',capsize=2)

    if scale2=='log':
        ax2.yscale('log')
                
    ax2.set_ylabel(r'$M_{\rm dust, thin} [\rm M_\odot]$')
    ax2.set_ylim(bottom=min_val,top=max_val)    
    
    custom_lines.append(Line2D([0], [0], color='black',linestyle='dashed', lw=2))
    custom_labels.append(r'$f_{\rm mass}$')
    
    custom_lines.append(Line2D([0], [0], color='black',linestyle='dotted', lw=2))
    custom_labels.append(r'$M_{\rm dust, thin}$')
    
    
    custom_lines.append(Line2D([0], [0], color='w',marker='o',markerfacecolor='black', lw=4))
    custom_labels.append(r'$f_{\rm model}$')
    ax.legend(custom_lines,custom_labels)   
    
    
    rvs=[]
    count=0
    for key in dust_analysis:
        if debug:
            print(key)
        idx=key.find('rv')
        rv=key[idx+2:idx+5]
        rvs.append(rv)
        count+=1
    if debug:
        print(rvs)
        print(np.arange(1,count+1))
    ax.set_xticks(np.arange(1,count+1))
    ax.set_xticklabels(rvs)
    if scale=='log':
        plt.savefig(f'{save_folder}{str(run_number)}_histogram_mass_complete_plot_log{suffix}{reduce_str}.{filetype_fig}',bbox_inches='tight')
    else:
        plt.savefig(f'{save_folder}{str(run_number)}_histogram_mass_complete_plot{suffix}{reduce_str}.{filetype_fig}',bbox_inches='tight')
        
    if close_plots:
        plt.close()
    else:

        plt.show()
    
    
if use_dust_emis:
    plot_histograms_both(dust_analysis=dust_analysis,dust_fraction_used=dust_fraction_used
                         ,dust_analysis_abs=dust_analysis_abs,scale='linear',scale2='linear',
                        debug=False)
if use_dust_absorp:
    plot_histograms_both(dust_analysis=dust_analysis_absorp,dust_fraction_used=dust_fraction_used_absorp
                         ,dust_analysis_abs=dust_analysis_abs_absorp,scale='linear',scale2='linear',
                        debug=False,suffix='_absorp')
 
     


print('Saved')


print('Plot molecular cornerplot...')
#adding header for derived molecular quantaties
header_derived=[]
for species in load_in_slab_dict:
        print(species)
        header_derived.append(species+':\n'+r'$r_{eff}$')
        header_derived.append(species+':\n'+'t at '+str(low_contribution))
        header_derived.append(species+':\n'+'t at '+str(high_contribution))
        header_derived.append(species+':\n'+'logDensCol at '+str(low_contribution))
        header_derived.append(species+':\n'+'logDensCol at '+str(high_contribution))
        if radial_version:
            header_derived.append(species+':\n'+'rout')
            header_derived.append(species+':\n'+'rin')
    
# checking which species where only included in the fixed dict and shouldn't be part of the arrays
if len(not_fitted_species)>0:
    tot_samples_rel_new=tot_samples_rel[:,:-len(header_derived)].copy()
    header_derived_new=[]
    print('Deleting not fitted molecules from output data')
    idxs_pop=[]
    for i in range(len(header_derived)):
        for mol in not_fitted_species:
            if mol in header_derived[i]:
                print('Deleting:',header_derived[i])
                idxs_pop.append(i)
                
                
    
    for i in range(len(header_derived)):
        if i not in idxs_pop:
            header_derived_new.append(header_derived[i])
            appendix=np.expand_dims(tot_samples_rel[:,-len(header_derived)+i],axis=1)
            
            tot_samples_rel_new=np.append(tot_samples_rel_new,appendix,axis=1)
    header_derived=header_derived_new
    tot_samples_rel=tot_samples_rel_new

    
# check if rin is 0  all the time
#this will be the case if we have a single slab emission
print('Checking if single slab was used and so some of the derived quantities make no sense')
print('Shape derived header:',len(header_derived))
print('Shape data array:',np.shape(tot_samples_rel))
tot_samples_rel_new=tot_samples_rel[:,:-len(header_derived)].copy()
header_derived_new=[]
idxs_pop=[]
for species in load_in_slab_dict:
    if species not in not_fitted_species:
        idx_pop = np.where(np.array(header_derived)==species+':\n'+'rin')[0][0]
        if np.min(tot_samples_rel[:,-len(header_derived)+idx_pop])==np.max(tot_samples_rel[:,-len(header_derived)+idx_pop]):
            idxs_pop.append(idx_pop)
            idx_pop = np.where(np.array(header_derived)==species+':\n'+'rout')[0][0]
            idxs_pop.append(idx_pop)
        

for i in range(len(header_derived)):
    if i not in idxs_pop:
        header_derived_new.append(header_derived[i])
        appendix=np.expand_dims(tot_samples_rel[:,-len(header_derived)+i],axis=1)
        print('shape appendix',np.shape(appendix))
        print('shape core',np.shape(tot_samples_rel_new))
        
        tot_samples_rel_new=np.append(tot_samples_rel_new,appendix,axis=1)
header_derived=header_derived_new
tot_samples_rel=tot_samples_rel_new
print('Shape new derived header:',len(header_derived))
print('Shape new data array:',np.shape(tot_samples_rel))
for i in range(len(header_derived)):
    print(header_derived[i],np.min(tot_samples_rel[:,-len(header_derived)+i]),np.max(tot_samples_rel[:,-len(header_derived)+i]))
CORNER_KWARGS = dict(
    smooth=.9,
    label_kwargs=dict(fontsize=20,rotation=45),
    title_kwargs=dict(fontsize=24,loc='left'),
    levels=[0.68, 0.95],
    quantiles=[0.5],
    plot_density=False,
    plot_datapoints=True,
    fill_contours=True,
    plot_contours=True,
    
    title_quantiles=[0.16,0.5,0.84],
    show_titles=True)
#        truths=val_obj_conv,)
fig = corner.corner(tot_samples_rel[:,-len(header_derived):], labels=header_derived, color='tomato', **CORNER_KWARGS)

plt.savefig(f'{save_folder}{str(run_number)}_Cornerplot_derived_mol{reduce_str}.{filetype_fig}',bbox_inches='tight')
if close_plots:
    plt.close()
else:

    plt.show()


print('Done!')


header_all=header_all+header_derived

print(header_all)
if save_output:
    np.save(f'{prefix}header_complete_posterior',np.array(header_all))
    
print('Saved')



print('Plot full cornerplot...')

'''
Here you can change the settings on how the scaling values are displayed
This means: should they be on log scale and if so where do you cut them.

'''
display_scale_log=True
clip_value=1e-25
scale_ir_log=False  # should the inner rim scaling factor be on a log scale 
clip_value_ir=1e-10
scale_mid_log=False # should the midplane scaling factor be on a log scale
clip_value_mid=1e-10

for i in range(len(header_all)):
    print(header_all[i],np.mean(tot_samples_rel[:,i]))


if display_scale_log:
    tot_samples_log_scale=[]
    for i in range(np.shape(tot_samples_rel)[1]):
        if use_dust_emis and use_dust_absorp:
            check_list=nicer_labels_output+nicer_labels_output_absorp
        elif use_dust_emis:
            check_list=nicer_labels_output
        elif use_dust_absorp:
            check_list=nicer_labels_output_absorp
        else:
            check_list=[]
            
        if header_all[i] in check_list:


            tot_samples_log_scale.append(np.log10(np.clip(tot_samples_rel[:,i],a_min=clip_value,a_max=None)))
        elif header_all[i]=='sc_ir' and scale_ir_log:
            tot_samples_log_scale.append(np.log10(np.clip(tot_samples_rel[:,i],a_min=clip_value_ir,a_max=None)))
        elif header_all[i]=='sc_mid' and scale_mid_log:
            tot_samples_log_scale.append(np.log10(np.clip(tot_samples_rel[:,i],a_min=clip_value_mid,a_max=None)))
            
        else:
            tot_samples_log_scale.append(tot_samples_rel[:,i])
    tot_samples_log_scale=np.array(tot_samples_log_scale).T
    print(np.shape(tot_samples_log_scale))


    


selected_posterior=[]
selected_header=[]

for i in range(len(header_all)):
    if all(tot_samples_rel[:,i]==np.max(tot_samples_rel[:,i])):
        print('-------------------')
        print('-------------------')
        print(f'For {header_all[i]} all retrieved values are {np.mean(tot_samples_rel[:,i])}')
        print(f'Therefore, there is no evidence for {header_all[i]}')
        print('-------------------')
        print('-------------------')
    else:
        if display_scale_log: 
            if use_dust_emis and use_dust_absorp:
                check_list=nicer_labels_output+nicer_labels_output_absorp
            elif use_dust_emis:
                check_list=nicer_labels_output
            elif use_dust_absorp:
                check_list=nicer_labels_output_absorp
            else:
                check_list=[]
            if (header_all[i] in check_list) or (header_all[i]=='sc_ir' and scale_ir_log) or(header_all[i]=='sc_mid' and scale_mid_log):
                selected_header.append(f'log {header_all[i]}')
            else:
                selected_header.append(header_all[i])
            selected_posterior.append(tot_samples_log_scale[:,i])
        else:
            selected_header.append(header_all[i])
        
            selected_posterior.append(tot_samples_rel[:,i])

selected_posterior=np.array(selected_posterior).T
print(np.shape(selected_posterior))




print('Writing values to txt file')

if use_dust_emis and use_dust_absorp:
    check_list=nicer_labels_output+nicer_labels_output_absorp
elif use_dust_emis:
    check_list=nicer_labels_output
elif use_dust_absorp:
    check_list=nicer_labels_output_absorp
else:
    check_list=[]

y_median=np.median(tot_samples_rel,axis=0)
y_std=np.percentile(tot_samples_rel,50+68/2,axis=0)

y_std_min=np.percentile(tot_samples_rel,50-68/2,axis=0)

with open(f'{save_folder}{str(run_number)}_posterior_values{reduce_str}.txt','w') as f:
    f.write('Parameter Median_value 1sigma_up 1sigma_down \n')
    for i in range(len(header_all)):
        name=header_all[i]
        med=y_median[i]
        minus=y_std_min[i]
        plus=y_std[i]
        if name not in check_list:
            f.write('%10s %.5e %.5e %.5e \n'%(name,med,plus,minus))

    if use_dust_emis:
        f.write('-------------------------------------------- \n')
        f.write('-------------------------------------------- \n')
        f.write('DUST FRACTIONS \n')
        f.write('Parameter Median_value 1sigma_up 1sigma_down \n')
        for key in dust_analysis:
            name=key
            med=dust_analysis[key][0]
            minus=dust_analysis[key][2]
            plus=dust_analysis[key][1]
            
            f.write('%s %.5e %.5e %.5e \n'%(name,med,plus,minus))
            
        f.write('-------------------------------------------- \n')
        f.write('-------------------------------------------- \n')
        f.write('ABSOLUTE OPTICALLY THIN DUST MASSES [M_sun] \n')
        f.write('Parameter Median_value 1sigma_up 1sigma_down \n')
        for key in dust_analysis_abs:
            name=key
            med=dust_analysis_abs[key][0]
            minus=dust_analysis_abs[key][2]
            plus=dust_analysis_abs[key][1]
            
            f.write('%s %.5e %.5e %.5e \n'%(name,med,plus,minus))
    if use_dust_absorp:
        f.write('-------------------------------------------- \n')
        f.write('-------------------------------------------- \n')
        f.write('DUST ABSORPTION FRACTIONS \n')
        f.write('Parameter Median_value 1sigma_up 1sigma_down \n')
        for key in dust_analysis_abs_absorp:
            name=key
            med=dust_analysis_absorp[key][0]
            minus=dust_analysis_absorp[key][2]
            plus=dust_analysis_absorp[key][1]
            
            f.write('%s %.5e %.5e %.5e \n'%(name,med,plus,minus))
            
        f.write('-------------------------------------------- \n')
        f.write('-------------------------------------------- \n')
        f.write('ABSOLUTE OPTICALLY THIN DUST ABSORPTION MASSES [M_sun] \n')
        f.write('Parameter Median_value 1sigma_up 1sigma_down \n')
        for key in dust_analysis_abs_absorp:
            name=key
            med=dust_analysis_abs_absorp[key][0]
            minus=dust_analysis_abs_absorp[key][2]
            plus=dust_analysis_abs_absorp[key][1]
            
            f.write('%s %.5e %.5e %.5e \n'%(name,med,plus,minus))
if plot_last_corner_plot:

    CORNER_KWARGS = dict(
        smooth=.9,
        label_kwargs=dict(fontsize=20,rotation=45),
        title_kwargs=dict(fontsize=24,loc='left'),
        levels=[0.68, 0.95],
        quantiles=[0.5],
        plot_density=False,
        plot_datapoints=True,
        fill_contours=True,
        plot_contours=True,
        title_quantiles=[0.16,0.5,0.84],
        title_fmt='10.4e',
        show_titles=True)


    fig = corner.corner(selected_posterior, labels=selected_header, color='tomato', **CORNER_KWARGS)

    plt.savefig(f'{save_folder}{str(run_number)}_Cornerplot_all_parameters{reduce_str}.{filetype_fig}',bbox_inches='tight')
    if close_plots:
        plt.close()
    else:

        plt.show()
print('Done!!')

if run_all:
    print('Starting the other plotting routines')
    print('Starting plot_mol_conditions-input.py')
    os.system(f'python plot_mol_conditions-input.py {input_file}')
    print('Starting plot_mol_contributions-input.py ')
    os.system(f'python plot_mol_contributions-input.py  {input_file}')
print('Finished!!')