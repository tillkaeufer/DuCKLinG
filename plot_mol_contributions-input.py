# %%
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

#from pymultinest.solve import solve, run
import matplotlib.pyplot as plt


import sys
import importlib

from spectres import spectres
from utils import *

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
colormap={0:'red',1:'green'}
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True 
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.major.size'] = plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['xtick.major.width'] = plt.rcParams['ytick.major.width'] = 1.6
plt.rcParams['font.size'] = 12


close_plots=True
preliminary=False

complete_header=True

# %%
if __name__ == "__main__":
    input_file=sys.argv[1]
    if len(sys.argv)>2:
   
        arg_list=sys.argv[1:]
        
        for i in range(len(arg_list)):
            argument=arg_list[i]
            if argument=='preliminary':
                preliminary=True
                complete_header=False
            elif argument=='simple':
                complete_header=False
            if argument=='close':
                close_plots=True
            if argument=='open':
                close_plots=False


# %%
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

#if __name__ == "__main__":
#    input_file=sys.argv[1]

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
    os.system(f'cp {input_file} {unique_filename}.py')
    mdl = importlib.import_module(unique_filename)
    os.system(f'rm {unique_filename}.py')
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
    import ultranest.stepsampler
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
    
try:
    save_mol_flux
    print('save_mol_flux')
    print(save_mol_flux)
except NameError:
    if limit_integrated_flux:
        save_mol_flux=True
    
    else:
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


debug=False

    

# %%
folder=bayesian_folder+subfold
if use_ultranest:
    try:
        list_files=glob.glob(folder+f'test_{run_number}/*/chains/equal_weighted_post.txt')
        list_files.sort()
        print(len(list_files))
        print(list_files[0])
    except:
        list_files=glob.glob(folder+f'test_{run_number}/*hains/equal_weighted_post.txt')
        list_files.sort()
        print(len(list_files))
        print(list_files[0])
else:
    list_files=[folder+f'test_{run_number}post_equal_weights.dat']
if complete_header:
    list_complete_post=glob.glob(folder+f'*_{run_number}complete_posterior.npy')
    list_complete_post.sort()
    print(len(list_complete_post))
    print(list_complete_post[0])


def read_file(filename,ultranest=True):
    if ultranest:
        data=np.loadtxt(filename,skiprows=1,dtype='float32')
        paras=data
        like=np.zeros((len(data)))
        return paras, like
    else:
        data=np.loadtxt(filename)
        if preliminary:
            
            paras=data[:-1]
            like=data[-1]
        else:
            paras=data[:,:-1]
            like=data[:,-1]
            
        return paras, like
# %%
def median_probable_model(filename,file_complete='',complete_header=False,debug=False):
    
    #reading file
    if not complete_header:
        paras,like=read_file(filename,ultranest=use_ultranest)
    else:
        paras_post,like=read_file(filename,ultranest=use_ultranest)
        paras=np.load(file_complete)
    if debug:
        print(np.shape(paras))
        if complete_header:
            print(np.shape(paras_post))
    
    nmodels=len(paras)
 
    #starting cut offs
    i1=int(nmodels/4)
    i2=int(3*nmodels/4)
    k=0
    k2=0
    
    #sorting the parameter 
    sorted_paras=np.sort(paras,axis=0)

    loop=True
    while loop: 
     
        #selecting the cutoff parameter values
        lower_lim=sorted_paras[i1]
        upper_lim=sorted_paras[i2]
        
        ibest=0 # id of best model
        lbest=-1e99 # log like of best model
        found=0 # how many models are selected this round
        
        loop=False
        
        for i in range(nmodels):
            model_para=paras[i]
            
            further=True    
            for j in range(len(model_para)):
                #print(j,lower_lim[j],model_para[j],upper_lim[j])
                if not ((not (model_para[j]<lower_lim[j]) ) and (not (model_para[j]>upper_lim[j]))):
                    further=False
                    break

            if further:    
                found+=1
                if (like[i]>lbest): #checking if this is the most likely model of all selected
                    ibest=i
                    lbest=like[i]
        if found!=1:
            if found>1 and k==0:
                i1=i1+1
                i2=i2-1
                k2+=1
                if i1<i2:
                    loop=True
                print(f'Decreasing boundaries: {k2}',end='\r',flush=True)
            elif found==0:
                k=k+1
                i1=i1-1
                i2=i2+1
                if i1>=0 and i2<nmodels:
                    loop=True
                else:
                    print('Nothing found')
                
                print(f'Increasing boundaries: {k}',end='\r',flush=True)
#        print(found,end='\r',flush=True)

    
    if complete_header:
        return ibest,like[i],paras_post[i],paras[i]
    else:
        return ibest,like[i],paras[i]        
                
print('Searching for the median probably model:')
# %%
if complete_header:
    ibest,like,paras,paras_complete=median_probable_model(list_files[0],list_complete_post[0],
                                                          complete_header=complete_header,debug=True)
else:
    if preliminary:
        paras,like=read_file(filename=list_files[0],ultranest=use_ultranest)
        ibest=0
        if len(np.shape(paras))==2:
                
            print('--------------------------------------')
            print('--------------------------------------')
            print('The posterior has more than one entry')
            print('Run the simple option instead')
            print('--------------------------------------')
            print('--------------------------------------')
            exit()
    else:    
        print('Simple median search')
        ibest,like,paras=median_probable_model(list_files[0],debug=True)

print('Done!')
# %%
max_flux_obs=np.max(flux_obs)



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



if 'tmax_s' in prior_dict or 'temp_s' in prior_dict or 'tmax_s' in fixed_dict or 'temp_s' in fixed_dict:
    use_dust_emis=True
    if 'tmax_s' in prior_dict or 'tmax_s' in fixed_dict:
        sur_powerlaw=True
    else:
        sur_powerlaw=False
else:
    use_dust_emis=False
if 'tmax_abs' in prior_dict or 'temp_abs' in prior_dict or 'tmax_abs' in fixed_dict or 'temp_abs' in fixed_dict:
    use_dust_absorp=True
    if 'tmax_abs' in prior_dict or 'tmax_abs' in fixed_dict:
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


try:
    print(len(lam_obs))
    con_model.read_data(variables=init_dict,dust_species=init_abundance,
                        absorp_species=init_abundance_absorp,
                        slab_dict=load_in_slab_dict,slab_prefix=slab_prefix,
                        stellar_file=stellar_file,wavelength_points=lam_obs,slab_only_mode=fit_gas_only,
                        dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)
except NameError:
    con_model.read_data(variables=init_dict,dust_species=init_abundance,
                        absorp_species=init_abundance_absorp,
                        slab_dict=load_in_slab_dict,slab_prefix=slab_prefix,
                        stellar_file=stellar_file,slab_only_mode=fit_gas_only,
                        dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)

print('Data read in')
# %%
def input_to_model(cube,debug=False,timeit=False):
    if timeit:
        time_1=time()
    if sample_all:
        var_dict,abundance_dict,slab_dict,abundance_dict_absorp,sigma_dict=cube_to_dicts(cube,header_para=header_para,header_abund=header_abund,header_all=complete_header,header_absorp=header_absorp,scale_prior=scale_prior,fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)


    else:
        var_dict,slab_dict,sigma_dict=cube_to_dict(cube,header=list(header_para)+list(header_slab)+list(header_sigma),fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err,log_coldens=log_coldens)

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
  
    if not fit_gas_only:            
        var_dict['bb_star']=use_bb_star
    

    if sample_all:
        return var_dict,abundance_dict,abundance_dict_absorp,slab_dict
    else:
        return var_dict,slab_dict

print(paras)

# %%
if sample_all:
    var_dict,abundance_dict,abundance_dict_absorp,slab_dict=input_to_model(paras)
else:
    
    var_dict,slab_dict=input_to_model(paras,debug=False)

# %%

if sample_all:
    interp_flux=con_model.run_model_normalized(variables=var_dict,dust_species=abundance_dict,
                                                slab_dict=slab_dict,absorp_species=abundance_dict_absorp,max_flux_obs=max_flux_obs)


    var_dict['sc_ir']=scales[0]
    var_dict['sc_mid']=scales[1]
    i=2
    for key in abundance_dict:
        abundance_dict[key]=scales[i]
        i+=1
else:
    if fit_gas_only:
        
        interp_flux=con_model.run_model(variables=var_dict,dust_species=init_abundance,slab_dict=slab_dict,output_all=False,timeit=False)
        abundance_dict=init_abundance.copy()
        abundance_dict_absorp=init_abundance_absorp.copy()
    else:
        debug=True
        if debug:
            print('Used var dict', var_dict)
            print('Used dust emission dict', init_abundance)
            print('Used dust absorption dict', init_abundance_absorp)
            print('Used slab dict', slab_dict)
        interp_flux=con_model.run_fitted_to_obs(variables=var_dict,
                                                dust_species=init_abundance,
                                                absorp_species=init_abundance_absorp,
                                                slab_dict=slab_dict,
                                                flux_obs=flux_obs,lam_obs=lam_obs)       
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

# %%

mol_data=con_model.extract_emission_quantities(low_contribution=0.15,high_contribution=0.85,debug=True,close_plots=close_plots)


# %%


con_model.read_data(variables=var_dict,dust_species=abundance_dict,
                    slab_dict=slab_dict,slab_prefix=slab_prefix,
                    stellar_file=stellar_file,wavelength_points=lam_obs,slab_only_mode=fit_gas_only,
                    dust_path=dust_path,slab_folder=slab_folder,ext_model=ext_model)

# %%
con_model.plot_radial_structure(ylog=False,close_plots=close_plots)

# %%
emission_flux_individual_scaled={}
for key in con_model.slab_dict:
    emission_flux_individual_scaled[key]=con_model.emission_flux_individual[key]*slab_dict[key]['radius']**2
    con_model.emission_flux_individual_scaled[key]=emission_flux_individual_scaled[key]
# %%


# %%
def plot_molecule_minds_like(interp_flux,mol_fluxes,flux_obs=flux_obs,lam_obs=lam_obs,wave_range=[13.6,16.3],save_name='',debug=True):
    first=True
    for key in mol_fluxes:
        if first:
            first=False
            tot_mol_flux=mol_fluxes[key].copy()
        else:
            tot_mol_flux+=mol_fluxes[key]
    print(np.max(interp_flux),np.max(tot_mol_flux),np.shape(tot_mol_flux))
    print(np.shape(lam_obs),np.shape(np.zeros_like(lam_obs)),np.shape(mol_fluxes[key]))
    residual=flux_obs-(interp_flux-tot_mol_flux)

    plt.figure(figsize=(12,4))
    #plt.step(lam_obs,(interp_flux-tot_mol_flux)*1000,linewidth=0.5,color='black',linestyle='dashed',label='Continuum')

    added_mol_flux=np.zeros_like(mol_fluxes[key])
    idx_color=0
    for key in mol_fluxes:
        new_mol=mol_fluxes[key]*1000
        mol_color=key
        if '_comp' in mol_color:
            idx_mol=mol_color.find('_comp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {key} to {mol_color}')
        if '_absorp' in mol_color:
            idx_mol=mol_color.find('_absorp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {key} to {mol_color}')
        if debug:
            print(key,'Maximum',np.max(new_mol))
        plt.fill_between(lam_obs,(interp_flux-tot_mol_flux)*1000+new_mol+added_mol_flux,(interp_flux-tot_mol_flux)*1000+added_mol_flux,linewidth=0,label=molecular_names[mol_color],color=mol_colors_dict[mol_color],step='pre',zorder=100-idx_color)
        idx_color+=1
        added_mol_flux+=new_mol

    #plt.step(lam_obs,(interp_flux)*1000,linewidth=0.5,color='grey')
    #plt.step(lam_obs,flux_obs*1000,linewidth=0.5,color='black')
    #plt.plot(lam_obs,(interp_flux-tot_mol_flux)*1000,linewidth=0.5,color='black')
   
    #plt.ylim([-0.1,1])
    plt.xlim(wave_range)
    plt.legend(loc=(0,1),ncol=max(1,len(list(mol_fluxes.keys()))//2)).set_zorder(102)
    plt.ylabel('$F \, [\mathrm{mJy}]$')
    plt.xlabel('$\lambda [\mathrm{\mu m}]$')
    if close_plots:
        plt.close()
    else:
        plt.show()
    plt.figure(figsize=(12,4))
    
    #plt.step(lam_obs,(interp_flux-tot_mol_flux)*1000,linewidth=0.5,color='black',linestyle='dashed',label='Continuum',zorder=102)
    
    
    added_mol_flux=np.zeros_like(mol_fluxes[key])
    idx_color=0
    for key in mol_fluxes:
        new_mol=mol_fluxes[key]*1000
        
        mol_color=key
        if '_comp' in mol_color:
            idx_mol=mol_color.find('_comp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {key} to {mol_color}')
        if '_absorp' in mol_color:
            idx_mol=mol_color.find('_absorp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {key} to {mol_color}')
        plt.fill_between(lam_obs,(interp_flux-tot_mol_flux)*1000+new_mol+added_mol_flux,(interp_flux-tot_mol_flux)*1000+added_mol_flux,linewidth=0,label=molecular_names[mol_color],color=mol_colors_dict[mol_color],step='pre',zorder=100-idx_color)
        added_mol_flux+=new_mol
        idx_color+=1
    plt.step(lam_obs,flux_obs*1000,linewidth=0.5,color='black',zorder=101)
    #plt.plot(lam_obs,(interp_flux+tot_mol_flux)*1000,linewidth=0.5,color='black')
   
    #plt.ylim([-0.1,1])
    plt.xlim(wave_range)

    plt.legend(loc=(0,1),ncol=max(1,len(list(mol_fluxes.keys()))//2)).set_zorder(102)
    
    plt.ylabel('$F \, [\mathrm{mJy}]$')
    plt.xlabel('$\lambda [\mathrm{\mu m}]$')
    if save_name!='':
        plt.savefig(save_name,bbox_inches='tight')
    if close_plots:
        plt.close()
    else:
        plt.show()
    plt.figure(figsize=(12,4))
    plt.step(lam_obs,residual*1000,linewidth=0.5,color='black',zorder=101)
    added_mol_flux=np.zeros_like(mol_fluxes[key])
    idx_color=0
    for key in mol_fluxes:
        new_mol=mol_fluxes[key]*1000

            
        mol_color=key
        if '_comp' in mol_color:
            idx_mol=mol_color.find('_comp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {key} to {mol_color}')
        if '_absorp' in mol_color:
            idx_mol=mol_color.find('_absorp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {key} to {mol_color}')
        plt.fill_between(lam_obs,new_mol+added_mol_flux,added_mol_flux,label=molecular_names[mol_color],color=mol_colors_dict[mol_color],step='pre',linewidth=0,zorder=100-idx_color)
        added_mol_flux+=new_mol
        idx_color+=1
    #plt.ylim([-5,None])
    plt.xlim(wave_range)
    plt.legend(loc=(0,1),ncol=max(1,len(list(mol_fluxes.keys()))//2)).set_zorder(102)
    plt.ylabel('$F \, [\mathrm{mJy}]$')
    plt.xlabel('$\lambda [\mathrm{\mu m}]$')
    if close_plots:
        plt.close()
    else:
        plt.show()
    plt.figure(figsize=(12,4))
    plt.step(lam_obs,(flux_obs-(interp_flux))*1000,linewidth=0.5,color='black')
    plt.xlim(wave_range)
    plt.ylabel('$\Delta F \, [\mathrm{mJy}]$')
    plt.xlabel('$\lambda [\mathrm{\mu m}]$')
    if close_plots:
        plt.close()
    else:
        plt.show()
    plt.figure(figsize=(12,4))
    plt.step(lam_obs,(flux_obs-(interp_flux))/(interp_flux)*100,linewidth=0.5,color='black')
    plt.axhline(0,linewidth=0.5,color='tab:blue')
    plt.xlim(wave_range)
    plt.ylabel('Residual [%]')
    plt.xlabel('$\lambda [\mathrm{\mu m}]$')
    if close_plots:
        plt.close()
    else:
        plt.show()
    print_residual=np.mean(abs((flux_obs-(interp_flux))*1000))
    print('Mean difference: %5.5f mJy'%print_residual)
    

# %%


mol_colors=['#4363d8', '#e6194b', '#3cb44b', '#ffe119',  '#911eb4', '#46f0f0',
             '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324',  '#800000',
             '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000','#fffac8']

# %%
i=0
used_labs=[]
for key in mol_colors_dict:
    if molecular_names[key] not in used_labs:
        used_labs.append(molecular_names[key])
        plt.axvline(i,label=molecular_names[key],color=mol_colors_dict[key],lw=10)
        i+=1
plt.legend(loc=(1,0))
if close_plots:
    plt.close()
else:
    plt.show()


save_folder=f'{fold_string}figures/'

if not os.path.exists(save_folder):
    os.system(f'mkdir {save_folder}')
else:
    print(f'Folder {save_folder} exists')

# %%
prefix_fig=folder+f'/figures/test_{run_number}'
print(prefix_fig)
file_name=prefix_fig+'_mol_contribution_plot.pdf'
plot_molecule_minds_like(interp_flux,emission_flux_individual_scaled,wave_range=[np.min(lam_obs),np.max(lam_obs)],save_name=file_name)

# %%
wave_grid=[]
wave_range_plot=min(np.ptp(lam_obs),2)
i=np.min(lam_obs)
while i<=np.max(lam_obs):
    if i==np.min(lam_obs):
        wave_grid.append([i-0.1,i+wave_range_plot-0.1])
    else:
        wave_grid.append([i,i+wave_range_plot])
        
    i+=wave_range_plot


# %%


# %%
def plot_molecule_subplots(interp_flux,mol_fluxes,flux_obs=flux_obs,lam_obs=lam_obs,
                           y_median=[None],y_std=[],y_std_min=[],wave_new=[],wave_range=[[13.6,16.3]],save_name='',debug=True):
    first=True
    for key in mol_fluxes:
        if first:
            first=False
            tot_mol_flux=mol_fluxes[key].copy()
        else:
            tot_mol_flux+=mol_fluxes[key]
    print(np.max(interp_flux),np.max(tot_mol_flux),np.shape(tot_mol_flux))
    print(np.shape(lam_obs),np.shape(np.zeros_like(lam_obs)),np.shape(mol_fluxes[key]))
    residual=flux_obs-(interp_flux-tot_mol_flux)
    
    nrows=len(wave_range)
    fig,axs=plt.subplots(nrows=nrows,figsize=(14,2.8*nrows))
    
    #plt.step(lam_obs,(interp_flux-tot_mol_flux)*1000,linewidth=0.5,color='black',linestyle='dashed',label='Continuum',zorder=102)
    plot_post=False
    if y_median[0]!=None:
        plot_post=True
    
    for i in range(len(wave_range)):
        
        min_wave=wave_range[i][0]
        max_wave=wave_range[i][1]
        idx=np.where(lam_obs<=max_wave)[0]
        lam_obs_select=lam_obs[idx]
        idx2=np.where(lam_obs_select>=min_wave)[0]
        lam_obs_select=lam_obs_select[idx2]
        flux_obs_select=flux_obs[idx[idx2]]
        interp_flux_select=interp_flux[idx[idx2]]
        tot_mol_flux_select=tot_mol_flux[idx[idx2]]
        if plot_post:
            if np.max(wave_new)>max_wave:
                idx3=np.where(wave_new<=max_wave)[0]
                if np.min(wave_new)<min_wave: 
                    idx4=np.where(wave_new[idx3]>=min_wave)[0]
                    idx_tot=idx3[idx4]
                else:
                    idx_tot=idx3
            else:
                if np.min(wave_new)<min_wave: 
                    idx4=np.where(wave_new>=min_wave)[0]
                    idx_tot=idx4
                else:
                    idx_tot=np.arange(len(wave_new))
            
            y_med_plot=y_median[idx_tot]
            y_std_plot=y_std[idx_tot]
            y_std_min_plot=y_std_min[idx_tot]
            wave_new_plot=wave_new[idx_tot]
        
        added_mol_flux=np.zeros_like(lam_obs_select)
        idx_color=0
        for key in mol_fluxes:
            new_mol=mol_fluxes[key][idx[idx2]]*1000


            mol_color=key
            if '_comp' in mol_color:
                idx_mol=mol_color.find('_comp')
                
                mol_color=mol_color[:idx_mol]
                print(f'Changing {key} to {mol_color}')
            if '_absorp' in mol_color:
                idx_mol=mol_color.find('_absorp')
                
                mol_color=mol_color[:idx_mol]
                print(f'Changing {key} to {mol_color}')
            axs[i].fill_between(lam_obs_select,(interp_flux_select-tot_mol_flux_select)*1000+new_mol+added_mol_flux,(interp_flux_select-tot_mol_flux_select)*1000+added_mol_flux,linewidth=0,label=molecular_names[mol_color],color=mol_colors_dict[mol_color],step='pre',zorder=100-idx_color)
            added_mol_flux+=new_mol
            idx_color+=1
        axs[i].step(lam_obs_select,flux_obs_select*1000,linewidth=0.5,color='black',zorder=101)
        
        if plot_post:
            axs[i].fill_between(wave_new_plot,(y_std_min_plot)*1000,(y_std_plot)*1000,color='tab:blue',alpha=0.7,linewidth=0,step='pre', zorder=121)

            axs[i].step(wave_new_plot,y_med_plot*1000,linewidth=0.5,color='tab:blue',zorder=121)
        
        #plt.plot(lam_obs,(interp_flux+tot_mol_flux)*1000,linewidth=0.5,color='black')

        #plt.ylim([-0.1,1])
        axs[i].set_xlim(wave_range[i])
        axs[i].set_ylabel('$F \, [\mathrm{mJy}]$')

    axs[0].legend(loc=(0,1),ncol=max(1,len(list(mol_fluxes.keys()))//2)).set_zorder(102)
    
    axs[-1].set_xlabel('$\lambda [\mathrm{\mu m}]$')

    if save_name!='':
        plt.savefig(save_name,bbox_inches='tight')
    if close_plots:
        plt.close()
    else:
        plt.show()

zoom_file_name=prefix_fig+'_mol_contribution_zoom_in_plot.pdf'
plot_molecule_subplots(interp_flux,emission_flux_individual_scaled,wave_range=wave_grid,save_name=zoom_file_name)
# %%

# printing the slab dict of the median probable model
print('Slab dict:')
print(slab_dict)


# calculate the integrated fluxes for all molecules

print('---------------------------')
print('Integrated fluxes for all molecules over the fitted wavelength region:')

for mol in slab_dict:
    print(f'{mol}: {con_model.calc_integrated_flux(mol,wave_lims=[np.min(lam_obs),np.max(lam_obs)])} W/m^2')
print('Done!')
print('---------------------------')


if os.path.isfile(f'{prefix}array_flux.npy'):
    print('Loading posterior of fluxes')
    array_flux=np.load(f'{prefix}array_flux.npy')
    #creating new wavelength grid that combines the observational wavelength and the standard model points
    
    standard_wave=np.load('./standard_wave.npy')
    
        
    # stiching part
    
    above=np.unique(np.clip(standard_wave,a_min=np.max(con_model.xnew),a_max=None))
    below=np.unique(np.clip(standard_wave,a_max=np.min(con_model.xnew),a_min=None))
    out=np.append(above,below)
    wave_new=np.sort(np.unique(np.append(out,con_model.xnew)))
    y_model=array_flux
    print('Calculating the percentiles of the spectrum (takes a while)')
    y_median=np.median(y_model,axis=0)
    y_std=np.percentile(y_model,50+68/2,axis=0)
    y_2std=np.percentile(y_model,50+95/2,axis=0)
    y_3std=np.percentile(y_model,50+99.9/2,axis=0)
    y_std_min=np.percentile(y_model,50-68/2,axis=0)
    y_2std_min=np.percentile(y_model,50-95/2,axis=0)
    y_3std_min=np.percentile(y_model,50-99.9/2,axis=0)
    
    zoom_file_name=prefix_fig+'_mol_contribution_zoom_in_plot_with_post.pdf'
    plot_molecule_subplots(interp_flux,emission_flux_individual_scaled,y_median=y_median,y_std_min=y_std_min,y_std=y_std,wave_new=wave_new,wave_range=wave_grid,save_name=zoom_file_name)
else:
    print('---------------------------')
    print('---------------------------')
    print('For getting the posterior overplotted')
    print('You need to run the plotting rountine with the argument save_all')

    print('---------------------------')
    print('---------------------------')
print('Finished!')


