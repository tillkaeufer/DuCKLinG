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

from pymultinest.solve import solve, run
import matplotlib.pyplot as plt


import sys
import importlib

from spectres.spectral_resampling_numba import spectres_numba  as spectres
#from  matplotlib import colormaps as cmaps
#cm = cmaps['viridis']

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


try:
    log_coldens
    print('Log coldens')
    print(log_coldens)
except NameError:
    log_coldens=False
    print("log_coldens not set in input file")
    print('Default is false')
try:
    old_version
    print('Old version')
    print(old_version)
except NameError:
    old_version=True
    print("old_version not set in input file")
    print('Default is True')
try:
    continuum_penalty
    print('continuum_penalty')
    print(continuum_penalty)
except NameError:
    continuum_penalty=False
    print("continuum_penalty not set in input file")
    print('Default is False')
try:
    select_conti_like
    print('select_conti_like')
    print(select_conti_like)
except NameError:
    select_conti_like=False
    print("select_conti_like not set in input file")
    print('Default is False')
try:
    sum_sigma
    print('sum_sigma')
    print(sum_sigma)
except NameError:
    sum_sigma=False
    print("sum_sigma not set in input file")
    print('Default is False')
try:
    penalty_fact
    print('penalty_fact')
    print(penalty_fact)
except NameError:
    penalty_fact=1
    print("penalty_fact not set in input file")
    print('Default is 1')
try:
    radial_version
    print('radial_version')
    print(radial_version)
except NameError:
    radial_version=False
    print("radial_version not set in input file")
    print('Default is False')
try:
    coldens_restriction
    print('coldens_restriction')
    print(coldens_restriction)
except NameError:
    coldens_restriction=False
    print("coldens_restriction not set in input file")
    print('Default is False')
    
    
debug=False
# here you have to set the path where you want to save all the runs (bayesian_folder) and the dust_path where your Q-files are

def degree_to_cos(deg):
    return np.cos(np.pi/180*deg)
def temp_to_rad(rmin,t,q,tmax):
    return (t/tmax)**(1/q) *rmin 
def powerlaw_radius(rmin,r,q,tmax):
    return (r/rmin)**q *tmax 
def powerlaw_density_temp(tmax,tmin,t,sigma_tmin,sigma_tmax):
    return (t/tmax)**p *sigma_max 
def area(rmin,rmax):
    return np.pi*(rmax**2-rmin**2)
# In[2]:

def generate_grid(R=1,lambda_0=1,lambda_n=1):
    del_loglam=np.log10(1.0+1.0/R)
    N=1+int(np.log10(lambda_n/lambda_0)/del_loglam)
    mwlsline=np.logspace(np.log10(lambda_0),np.log10(lambda_n),N)
    return mwlsline


def spline(lam,flux,new_lam):

    #interpolation on a double logarithmic scale
    s=interpolate.InterpolatedUnivariateSpline(np.log10(lam),np.log10(flux))
    interp=10**(s(np.log10(new_lam)))
#    return interp #returning the nex SED
    return interp #for visualisation
class complete_model:
    
    #initializing the model
    #this needs to be done ones, so that all variables are known
    def __init__(self):
        self.variables=None
        self.abundance_dict=None
        self.data_dict={}
        #self.plotting=plotting
        #self.timeit=timeit
        #self.printing=printing
        self.star_spec=[]
        
        self.xnew=[]
        self.nwav=0
        self.freq=[]
        
        self.compo_ar=[]
        self.scaleparas=[]
        
        #slab data
        self.slab_wave=[]
        self.slab_data={}
        self.slab_parameters={}
        self.slab_temp_steps=0
        
        self.saved_continuum=[]
        
        #running the model without the continuum
        self.slab_only_mode=False
        
        #constants
        self.h = 6.6262e-27
        self.c = 2.997925e+10
        self.kboltz = 1.38e-16
        self.rsun = 6.9634e10
        self.parsec = 3.0857e18
        self.au =1.496e13
        self.trans_flux=1e23
        self.msun=1.99847e33
        self.mjup=1.89813e30
        self.mearth=5.9722e27

        self.scaled_stellar_flux=[]
        self.rim_flux=[]
        self.midplane_flux=[]
        self.surface_flux_tot=[]
        self.surface_flux_individual={}
        self.surface_flux_individual_scaled={}
        self.tot_flux=[]
        self.emission_flux=[]
        self.emission_flux_individual={}
        self.emission_flux_individual_scaled={}
        
        self.run_tzones=False
          
        self.bb_array=[]
        self.interp_bbody=False
        self.bb_wave_grid=[]
        
        self.bb_temp_steps=0
        self.bb_min_temp=1
        self.bb_max_temp=1000
        self.bb_temp_list=[]
        
        self.rim_powerlaw=False
        self.cosi=False
        self.single_slab=False
        self.radial_version=radial_version
        
    def __str__(self):
        #defining the output that will be printed
        #do print(model) if you want to see it
        #it will show the variable and dust dict
        string='-----------------\n'
        for key in self.variables:
            string+= key +': '
            string+=str(self.variables[key])+'\n'
        string+='\n'
        string+='-----------------\n'
        string+='Used data:\n'
        if self.data_dict!={}:
            for key in self.data_dict:
                string+=key+'\n'
        else:
            string+='Not loaded\n'
        
        string+='-----------------\n'
        return string    
    
    # defining how a black body
    # and black body with a powerlaw are calculated
    def bbody(self,t,freq=[],from_array=False):
        # there are two options
        
        # not from array:
        # calculating the black body useing the formular
        
        # from array:
        # using the precalucated arrays of black body to interpolate
        
        if from_array:
            idx=np.argmin(np.abs(self.bb_temp_list-t))
            if self.bb_temp_list[idx]<=t:
                idx_low=idx
                idx_up=idx+1
            else:
                idx_low=idx-1
                idx_up=idx
            if idx_up>=len(self.bb_temp_list):
                freq=self.c*1e4/self.xnew
                return (2*self.h*(freq*freq*freq)/(self.c*self.c))*(1.0/((np.exp((self.h*freq/(self.kboltz*t))) - 1.0)))
    
            
            
            black_body=(self.bb_array[idx_up]-self.bb_array[idx_low])*(t-self.bb_temp_list[idx_low])/self.bb_temp_steps+self.bb_array[idx_low]
            if self.interp_bbody:
                return np.interp(self.xnew,self.bb_wave_grid,black_body)
            else:
                return black_body
        else:
            return (2*self.h*(freq*freq*freq)/(self.c*self.c))*(1.0/((np.exp((self.h*freq/(self.kboltz*t))) - 1.0)))

    def bbody_temp_powerlaw(self,t,exp):
        # calculating a black body aith t**exp factor
        # Replaced with precomputation -1 because of starting
        idx=np.argmin(np.abs(self.bb_temp_list-t))
        if self.bb_temp_list[idx]<=t:
            idx_low=idx
            idx_up=idx+1
        else:
            idx_low=idx-1
            idx_up=idx


        black_body=(self.bb_array[idx_up]-self.bb_array[idx_low])*(t-self.bb_temp_list[idx_low])/self.bb_temp_steps+self.bb_array[idx_low]
        bb_exp=black_body*(t**(exp))
        if self.interp_bbody:
            return np.interp(self.xnew,self.bb_wave_grid,bb_exp)
        else:
            return bb_exp




    
    def read_data(self,variables={},dust_species={},slab_dict={},R=0,wavelength_points=[],
                  bb_temp_steps=10,bb_min_temp=10,bb_max_temp=10000,
                  stellar_file ='./MCMAXspec-hae.in',dust_path=dust_path,
                  slab_folder='./LineData/', slab_only_mode=False,
                  slab_prefix='1_',save_binned_data=True,load_binned_data=True,
                  q_files=True,interp_bbody=False,debug=False):
        # loading all the data the model needs
        # wavlength_points: array with the wavlength points (in micron) at which the model is evaluated
        # if it's left empty a predefined grid is used
        # stellar_file: path of the stellar spectrum (if you want to use it)
        # dust_path: folder with the dust files (already set in the cell above)
        # interp_bbody: leave it to false 
        
        
        #running the model without the continuum?
        self.slab_only_mode=slab_only_mode
        self.variables=variables
        
        if not slab_only_mode:
            self.abundance_dict=dust_species

            if 'incl' in self.variables:
                self.cosi=True

            self.interp_bbody=interp_bbody
        if len(wavelength_points)==0:
            #not saving the same data over and over again
            save_binned_data=False
            load_binned_data=False
            if R!=0:
                new_slab_wave=generate_grid(R,lambda_0=4,lambda_n=30)[1:-1] #in future maybe as input if needed
                self.slab_wave=new_slab_wave

        else:
            '''
            This does not work yet since slab_wave is not defined then
            '''
            
            
            self.xnew=np.array(wavelength_points)
            self.slab_wave=self.xnew

 
        '''
        Loading all the slab data
        taking the slab wavlength points to evaluate the slab data
        loading the density and temperature data
        
        '''
        slab_list=list(slab_dict.keys())
        
        ex_prebin=False
        if load_binned_data:
            slab_bin_folders=glob.glob(f'{slab_folder}/binned_data/*')
            slab_bin_folders.sort()
            print('Folder to be searched for pre-binned data:')
            print(slab_bin_folders)

            for folder in slab_bin_folders:
                ex_prebin=os.path.isfile(folder+'/wavelength.npy')
                print('Searching:')
                print(folder+'/wavelength.npy')
                print(ex_prebin)
                if ex_prebin:
                    wave_folder=np.load(folder+'/wavelength.npy')
                    if np.array_equal(self.xnew,wave_folder):
                        print('-----------------------------')
                        print('Found pre-binned data folder!')
                        print(folder)
                        print('-----------------------------')
                        save_slab_string=folder

                        break
            if not ex_prebin:
                print('-----------------------------')
                print('Found no pre-binned data folder!')
                if save_binned_data:
                    for i in range(100):
                        if not os.path.exists(f'{slab_folder}/binned_data/{str(i)}'):
                            print('Saving data in',f'{slab_folder}/binned_data/{str(i)}')
                            os.system(f'mkdir {slab_folder}/binned_data/{str(i)}')
                            np.save(f'{slab_folder}/binned_data/{str(i)}/wavelength',self.xnew)
                            save_slab_string=f'{slab_folder}/binned_data/{str(i)}'
                            break
                print('-----------------------------')
                
        for mol_name_init in slab_list:
            if '_comp' in mol_name_init:
                idx_comp=mol_name_init.find('_comp')
                mol_name=mol_name_init[:idx_comp]
                print(mol_name_init)
                print('is changed to')
                print(mol_name)
                
            else:
                mol_name=mol_name_init
            
            ex_mol=False
            if ex_prebin:
                ex_mol=os.path.isfile(f'{folder}/{slab_prefix}{mol_name}_convFlux.npy')
                if ex_mol:
                    print('Found data for',mol_name)
                    self.slab_data[mol_name_init]=np.load(f'{folder}/{slab_prefix}{mol_name}_convFlux.npy')
                
            if not ex_mol:
                if load_binned_data and ex_prebin:
                    print('No data found for',mol_name)
                slab_data_loaded=np.load(f'{slab_folder}{slab_prefix}{mol_name}_convFlux.npy')
                slab_wave=np.load(f'{slab_folder}{slab_prefix}{mol_name}_convWave.npy')

                if R==0 and len(wavelength_points)==0:
                    conti=False
                    if len(self.slab_wave)==0:
                        conti=True
                    if not conti:
                        for i in range(len(self.slab_wave)):
                            if self.slab_wave[i]!=slab_wave[i]:
                                conti=True
                                break
                    if conti:
                        print('Updateing Slab wave')
                        self.slab_wave=slab_wave

                    self.slab_data[mol_name_init]=slab_data_loaded
                else:
                    print('------------------')
                    print(f'Binning {mol_name}')
                    print('------------------')
                    slab_data_filled=np.zeros((np.shape(slab_data_loaded)[0],np.shape(slab_data_loaded)[1],len(self.slab_wave)))
                    count=0
                    for idx1 in range(np.shape(slab_data_loaded)[0]):
                        for idx2 in range(np.shape(slab_data_loaded)[1]):

                            slab_data_filled[idx1,idx2]= spectres(new_wavs=self.slab_wave, spec_wavs=np.flip(slab_wave),
                                                                  spec_fluxes=np.flip(slab_data_loaded[idx1,idx2]),fill=0.0,verbose=False)
                            if count%20==0:
                                print(f'{np.round(count/(np.shape(slab_data_loaded)[0]*np.shape(slab_data_loaded)[1])*100,1)} %',end='\r',flush=True)
                            count+=1
                            
                    self.slab_data[mol_name_init]=slab_data_filled    
                    if save_binned_data:
                        print('Saving:',mol_name,f'{save_slab_string}/{slab_prefix}{mol_name}_convFlux.npy')
                        np.save(f'{save_slab_string}/{slab_prefix}{mol_name}_convFlux',slab_data_filled)
                    print('                      ')
                    print('Done')
            else:
                print('Checking for molecular data')
        self.slab_parameters['col']=np.load(f'{slab_folder}{slab_prefix}parameter_col.npy')
        self.slab_parameters['temp']=np.load(f'{slab_folder}{slab_prefix}parameter_temp.npy')
        
        self.slab_temp_steps=self.slab_parameters['temp'][-1]-self.slab_parameters['temp'][-2]
        if not slab_only_mode:
            if len(wavelength_points)==0:
                #setting up the wavlength grid at which we calculate the model
                x1short = np.linspace(0.2,4,1000,endpoint='false')
                x1long = np.linspace(30,67.0,1000,endpoint='false')
                x2 = np.linspace(67.0,73.0,300,endpoint='false')
                x3= np.linspace(73.0,200.,500,endpoint='false')
                x4 = np.append(x1short,self.slab_wave)
                x5 = np.append(x4,x1long)
                x6 = np.append(x5,x2)
                self.xnew = np.sort(np.unique(np.append(x6,x3)))
        else:
            if len(wavelength_points)==0:
                self.xnew = self.slab_wave
        
        self.nwav = len(self.xnew)

        #converting it to frequencies
        self.freq = np.array((self.c*1e4)/(self.xnew))
        
        
        # pasting the slab data at the indices that corrispond to the slab wavelength
        # in the total grid         
        if not slab_only_mode:
            if len(wavelength_points)==0:
                #determining the indecies
                idx_slab=[]
                for idx_1 in range(len(self.slab_wave)):
                    idx_slab.append(np.where(self.xnew==self.slab_wave[idx_1])[0][0])
                # pasting the data at the right places
                for mol_name in slab_list:
                        mol_data=self.slab_data[mol_name].copy()
                        mol_data_all=np.zeros((np.shape(mol_data)[0],np.shape(mol_data)[1],self.nwav))
                        mol_data_all[:,:,np.array(idx_slab)]=mol_data
                        self.slab_data[mol_name]=mol_data_all
      

    
            

              
        if not slab_only_mode:

            # loading all dust species
            # they are linearly interpolated to the wavlength grid
            # they are saved in a dictonary (self.data_dict) under the name of the specie
            for key in self.abundance_dict:
                if debug: print('Load '+key+'...')
                if '/' in key:
                    print('Make sure not to mix up dust species from different opacity mechanisms!!')
                    print(key)

                if q_files:
                    wavelength,kabs = np.loadtxt(dust_path+key,skiprows=1,usecols=(0,1),unpack=True)  

                else:
                    wavelength,kabs = np.loadtxt(dust_path+key, comments="#", skiprows=0,
                                                          usecols=(0,1), unpack=True) 
                if debug:
                    print('wavelength min max',np.min(wavelength),np.max(wavelength))
                    print('absorbtion min max',np.min(kabs),np.max(kabs))

                # IS IT OKAY TO LIMIT IT T POSITIVE NUMBER??!?!?!?!
                kabs=np.clip(kabs,a_min=0.0,a_max=None)
                self.data_dict[key]=interpolate.interp1d(wavelength,kabs,kind='linear',bounds_error=False,fill_value=0.0)(self.xnew)
                if debug:
                    print('interp')
                    print('absorbtion min max',np.min(self.data_dict[key]),np.max(self.data_dict[key]))

            # loading the stellar spectrum if we don't assume a black body
            # we interplate it again linearly to the wavelength grid
            if not self.variables['bb_star']:
                wavestar,fluxstar = np.loadtxt(stellar_file, comments="#", skiprows=0,
                                                   usecols=(0,1), unpack=True)
                self.starspec = interpolate.interp1d(wavestar,fluxstar,kind='linear')(self.xnew)


            # in case we want to use tzones
            # we add tmin_s and t_max to the list of tzones
            # if they weren't there already
            # saveing the output as the new tzones

            # if no tzones are defined the minimum and maximum temperature are used 
            if 'tzones' in self.variables:
                self.run_tzones=True
                add=0
                if self.variables['tmin_s'] not in self.variables['tzones']:
                    add+=1
                    min_add=True
                if self.variables['tmax_s'] not in self.variables['tzones']:
                    add+=1
                    max_add=True
                if add!=0:
                    t_all=np.zeros(len(self.variables['tzones'])+add)
                    if min_add:
                        t_all[0]=self.variables['tmin_s']
                    if max_add:
                        t_all[-1]=self.variables['tmax_s']
                    if add==2:
                        t_all[1:-1]=self.variables['tzones']
                    else:
                        if max_add:
                            t_all[:-1]=self.variables['tzones']
                        else:
                            t_all[1:]=self.variables['tzones']
                else:
                    t_all=self.variables['tzones']
                self.variables['tzones']=np.sort(t_all)    


            #check if rim is a powerlaw or a single temperature
            if 't_rim' in self.variables:
                self.rim_powerlaw=False
            else:
                self.rim_powerlaw=True
                #this of course means you need to set
                # all other relevant variables
                list_rim=['tmax_rim','tmin_rim','q_rim']
                for var in list_rim:
                    if var not in self.variables:
                        print('-------')
                        print('There are not all parameters for the inner rim set!!!')
                        print(f'{var} is not defined')
                        print('-------')


            #precalculating black bodys for temperature grid

            print('Precalculating Black bodies')
            self.bb_temp_steps=bb_temp_steps
            self.bb_min_temp=bb_min_temp
            self.bb_max_temp=bb_max_temp
            t_min_bb=1 #minimum temperature
            t_max_bb=10000 # maximum temperature
            if self.interp_bbody:
                bb_wave_grid=np.logspace(np.log10(np.min(self.xnew)),np.log10(np.max(self.xnew)),num=1000)
            else:
                bb_wave_grid=self.xnew
            self.bb_wave_grid=bb_wave_grid
            bb_freq_grid= np.array((self.c*1e4)/(bb_wave_grid))
            num_temp_bb=int((self.bb_max_temp-self.bb_min_temp)/self.bb_temp_steps+1)
            self.bb_array=np.zeros((num_temp_bb,len(bb_freq_grid)))
            i=0
            self.bb_temp_list=[]
            for temp in range(self.bb_min_temp,self.bb_max_temp+self.bb_temp_steps,self.bb_temp_steps):
                if debug:
                    print(temp)
                self.bb_array[i]=self.bbody(t=temp,freq=bb_freq_grid,from_array=False)
                self.bb_temp_list.append(temp)
                i+=1
                if i%100==0:                
                    print(f'{np.round(i/num_temp_bb*100)}%',end='\r',flush=True)
            self.bb_temp_list=np.array(self.bb_temp_list)



    def set_midplane(self,use_as_inner_rim=False,new_midplane=True,timeit=False,small_window=False,debug=False):
        if timeit: time1=time()
        # choosing if the function uses the midplane of rim variables
        # both are using a temperature powerlaw and can be decribed by the same functions
        if not use_as_inner_rim:
            tmin_mp=self.variables["tmin_mp"] #minimum temperature midplane
            tmax_mp=self.variables["tmax_mp"] #maximum temperature midplane
            exp=self.variables['exp_midplane']
        else:
            tmin_mp=self.variables["tmin_rim"] #minimum temperature rim
            tmax_mp=self.variables["tmax_rim"] #maximum temperature rim
            exp=self.variables['exp_rim']
        
        
        if new_midplane:
            '''
            New implementation that is (hopefully) faster
            This implementation makes use of the precomputed black bodies.
            The idea:
                - selecting all black bodies within the temperature ranges
                - multiplying them with the temperature powerlaw
                - using the grid points in and outside of tmin_mp and tmax_mp
                  to calculate the Black bodies at tmin_mp and tmax_mp
                - multiplying these BBs with the temperature power law
                - summing up (times width of relevant temperature range) all
                  the components to have the integral over the full temperature range
                - the relevant ranges are for the inside points 1K
                - for the gridpoint inside the edges it 1/2 (pointing in from the temperature range)
                - the gridpoint inside the edges and the edge are equally contributing
                  to the area between them
                - This method reproduces the results of np.tapz with the gridpoints and tmax_mp and tmin_mp as points
                - If the upper and lower temperature limits are very close
                  more points are inserted and np.trapz used to solve the problem
                - This method needs to have 3 points inside the temperature range
                  therefore I added another version that jumps in if this is not the case 
                  (In future we can make this also more efficient)
            '''
            
            idx_min=np.argmin(np.abs(self.bb_temp_list-tmin_mp))
            if self.bb_temp_list[idx_min]<=tmin_mp:
                idx_tmin_mp=idx_min
            else:
                idx_tmin_mp=idx_min-1
            idx_max=np.argmin(np.abs(self.bb_temp_list-tmax_mp))
            if self.bb_temp_list[idx_max]<=tmax_mp:
                idx_tmax_mp=idx_max
            else:
                idx_tmax_mp=idx_max-1


            tmin_below=self.bb_temp_list[idx_tmin_mp]
            tmax_below=self.bb_temp_list[idx_tmax_mp]
            
            if debug:
                print('=====================')
                print('Midplane')
                print('=====================')
                print('Idx_min, t_min',idx_tmin_mp,tmin_below)
                print('Idx_max, t_max',idx_tmax_mp,tmax_below)
            
            if (idx_tmax_mp-idx_tmin_mp)<3:
                small_window==True
            if small_window:
                

                num_points=10

                #creating points on every gridpoint (or finer grid) in the temperature range and adding the limits to it
                ar_temp_mid=list(np.linspace(tmin_mp,tmax_mp,num_points))
                #ar_temp_mid.insert(0,tmin_mp)
                #ar_temp_mid.insert(len(ar_temp_mid),tmax_mp)
                ar_temp_mid=np.array(ar_temp_mid)
                
                #calculating the fluxes for every temperature
                fluxes=[]
                for t in ar_temp_mid:

                    fluxes.append(self.bbody(t,self.freq,from_array=True)*t**exp)
                fluxes=np.array(fluxes)
                # using a numpy function for the integration
                flux_midplane=np.trapz(fluxes,ar_temp_mid,axis=0)


            else:

                if timeit: time2=time()
                # bbs of all grid points in the temperature range excluding the outer two points
                bb_points_mp=self.bb_array[idx_tmin_mp+2:idx_tmax_mp].copy()

                if timeit: time3=time()

                # calculating the corisponding temperature**exp values
                bb_temps_mp=np.arange(tmin_below+2*self.bb_temp_steps,tmax_below,self.bb_temp_steps)**exp

                if timeit: time4=time()
                if debug:
                    print('Shapes must match',np.shape(bb_points_mp),np.shape(bb_temps_mp))
                #multiplying the BBs with the temperature powerlaws
                for i in range(len(bb_temps_mp)):
                    bb_points_mp[i]*=bb_temps_mp[i]

                if timeit: time5=time()

                # these are gridpoints that are still within the temperature range
                upper_bb_in=self.bb_array[idx_tmax_mp]
                lower_bb_in=self.bb_array[idx_tmin_mp+1]
                
                # they are outside of them
                upper_bb_out=self.bb_array[idx_tmax_mp+1]
                lower_bb_out=self.bb_array[idx_tmin_mp]
                
                
                #these are the BBs* temp**exp for the two edges (tmax and tmin)
                upper_edge=(upper_bb_in+(upper_bb_out-upper_bb_in)*(tmax_mp-tmax_below)/self.bb_temp_steps)*(tmax_mp)**exp
                lower_edge=(lower_bb_out+(lower_bb_in-lower_bb_out)*(tmin_mp-tmin_below)/self.bb_temp_steps)*(tmin_mp)**exp
                if debug:
                    print('Upper edge',np.sum(upper_edge))
                    print('Lower edge',np.sum(lower_edge))
                #these are the BBs just inside times temp**exp
                upper_bb_in_exp=self.bb_array[idx_tmax_mp]*(tmax_below)**exp
                lower_bb_in_exp=self.bb_array[idx_tmin_mp+1]*(tmin_below+self.bb_temp_steps)**exp
                if debug:
                    print('Upper bb in',np.sum(upper_bb_in_exp))
                    print('Lower bb in',np.sum(lower_bb_in_exp))

                
                #these are the number of grid points (not used anymore)
                num_points=int(tmax_mp)-int(tmin_mp)+1

                if timeit: time6=time()
                
                #all inner points have a temperature range of self.bb_temp_steps K
                inner_part=np.sum(bb_points_mp,axis=0)*self.bb_temp_steps
                #the two grid points inside have a weight of 1/2 from their value inside
                lower_in=lower_bb_in_exp*1/2*self.bb_temp_steps
                upper_in=upper_bb_in_exp*1/2*self.bb_temp_steps

                
                #these are the components between the last gridpoints and the limits
                lower_out=(lower_edge+lower_bb_in_exp)*(tmin_below+self.bb_temp_steps-tmin_mp)/2
                upper_out=(upper_bb_in_exp+upper_edge)*(tmax_mp-tmax_below)/2



                #summing eveything up
                flux_midplane=inner_part+lower_in+upper_in+upper_out+lower_out

                if timeit: 
                                time7=time()
                                print('--------------')
                                print('Midpplane time')
                                print('Init',time2-time1)
                                print('Call array',time3-time2)
                                print('BB times exp',time4-time3)
                                print('Multiplying',time5-time4)
                                print('First and last point',time6-time5)
                                print('Summing up',time7-time6)

                                print('--------------')


        else:
            #this is an old implementation that we only have for comparison
            #it shouldn't be used in multinest
            num_points=int((tmax_mp-tmin_mp))+1
            if num_points<10:
                num_points=10
            ar_temp_mid=np.linspace(tmin_mp+self.bb_temp_steps,tmax_mp,num_points,endpoint='False')
            flux_mid=np.zeros((num_points,len(self.xnew)))
            i=0
            for t in ar_temp_mid:
                if i==0 or i==len(ar_temp_mid)-1:
                    flux_mid[i]=self.bbody_temp_powerlaw(t,exp)*(tmax_mp-tmin_mp)/(num_points-1)/2*self.bb_temp_steps
                else:
                    flux_mid[i]=self.bbody_temp_powerlaw(t,exp)*(tmax_mp-tmin_mp)/(num_points-1)*self.bb_temp_steps
                i+=1
            flux_midplane=np.sum(flux_mid,axis=0)
            
        return flux_midplane



    def set_surface(self,one_output=False,new_surface=True,small_window=False,timeit=False):

        '''
        The surface is in general:
        A black body temperature powerlaw multiplyed by the dust opacity times dust abundance 

        New implementation that is faster
        This implementation makes use of the precomputed black bodies.
        The idea is the same as for the midplane.
        Additionally, the result is multiplyed by the dust opacity times abundance.
        Note that also the older version is a debugged version of the previous iteration.
        '''

        tmin,tmax=self.variables['tmin_s'],self.variables['tmax_s']
        if timeit: time1=time()
        if new_surface:
            idx_min=np.argmin(np.abs(self.bb_temp_list-tmin))
            if self.bb_temp_list[idx_min]<=tmin:
                idx_tmin=idx_min
            else:
                idx_tmin=idx_min-1
            idx_max=np.argmin(np.abs(self.bb_temp_list-tmax))
            if self.bb_temp_list[idx_max]<=tmax:
                idx_tmax=idx_max
            else:
                idx_tmax=idx_max-1


            tmin_below=self.bb_temp_list[idx_tmin]
            tmax_below=self.bb_temp_list[idx_tmax]

            if (int(tmax)-int(tmin))<3:
                small_window==True
            if small_window:

                num_points=int(tmax)-int(tmin)
                if num_points<10:
                    num_points=10

                    
                    
                ar_temp=list(np.linspace(tmin,tmax,num_points))
                #ar_temp.insert(0,tmin)
                #ar_temp.insert(len(ar_temp),tmax)
                ar_temp=np.array(ar_temp)
                fluxes=[]
                for t in ar_temp:

                    fluxes.append(self.bbody(t,self.freq,from_array=True)*t**self.variables['exp_surface'])
                fluxes=np.array(fluxes)
                tot_bb_exp=np.trapz(fluxes,ar_temp,axis=0)
                dust_abs=np.zeros_like(tot_bb_exp)

                for key in self.abundance_dict:
                    if one_output:
                        dust_abs+=self.abundance_dict[key]*self.data_dict[key] 
                    else:
                        self.surface_flux_individual[key]=self.data_dict[key]*tot_bb_exp
         
                if one_output:
                    flux_surface_tot=tot_bb_exp*dust_abs
                    return flux_surface_tot


            else:

                if timeit: time2=time()
                bb_points=self.bb_array[idx_tmin+2:idx_tmax].copy()

                if timeit: time3=time()


                bb_temps=np.arange(tmin_below+2*self.bb_temp_steps,tmax_below,self.bb_temp_steps)**self.variables['exp_surface']

                if timeit: time4=time()
                for i in range(len(bb_temps)):
                    bb_points[i]*=bb_temps[i]

                if timeit: time5=time()


                upper_bb_in=self.bb_array[idx_tmax]
                lower_bb_in=self.bb_array[idx_tmin+1]

                upper_bb_out=self.bb_array[idx_tmax+1]
                lower_bb_out=self.bb_array[idx_tmin]

                upper_edge=(upper_bb_in+(upper_bb_out-upper_bb_in)*(tmax-tmax_below)/self.bb_temp_steps)*(tmax)**self.variables['exp_surface']
                lower_edge=(lower_bb_out+(lower_bb_in-lower_bb_out)*(tmin-tmin_below)/self.bb_temp_steps)*(tmin)**self.variables['exp_surface']



                upper_bb_in_exp=self.bb_array[idx_tmax]*(tmax_below)**self.variables['exp_surface']
                lower_bb_in_exp=self.bb_array[idx_tmin+1]*(tmin_below+self.bb_temp_steps)**self.variables['exp_surface']



                num_points=int(tmax)-int(tmin)+1

                if timeit: time6=time()

                inner_part=np.sum(bb_points,axis=0)*self.bb_temp_steps
                lower_in=lower_bb_in_exp*1/2*self.bb_temp_steps
                upper_in=upper_bb_in_exp*1/2*self.bb_temp_steps

                lower_out=(lower_edge+lower_bb_in_exp)*(tmin_below+self.bb_temp_steps-tmin)/2
                upper_out=(upper_bb_in_exp+upper_edge)*(tmax-tmax_below)/2


                if timeit: time7=time()
                #print(inner_part)
                tot_bb_exp=(inner_part+lower_in+upper_in+upper_out+lower_out).copy()
                dust_abs=np.zeros_like(tot_bb_exp)

                for key in self.abundance_dict:
                    if one_output:
                        dust_abs+=self.abundance_dict[key]*self.data_dict[key] 
                    else:
                        self.surface_flux_individual[key]=self.data_dict[key]*tot_bb_exp
         
                if one_output:
                    flux_surface_tot=tot_bb_exp*dust_abs
                    return flux_surface_tot

                if timeit: 
                                time8=time()
                                print('--------------')
                                print('Surface time')
                                print('Init',time2-time1)
                                print('Call array',time3-time2)
                                print('BB times exp',time4-time3)
                                print('Multiplying',time5-time4)
                                print('First and last point',time6-time5)
                                print('Multiplying with dust opacity',time7-time6)
                                print('Summing up',time8-time7)

                                print('--------------')

        else:


            if one_output:
                flux_surface_tot = np.zeros((self.nwav))

            num_points=int((tmax-tmin))+1
            if num_points<10:
                num_points=10
            ar_temp=np.linspace(tmin,tmax,num_points,endpoint='False')
            flux=np.zeros((num_points,len(self.xnew)))
            i=0
            for t in ar_temp:
                if i==0 or i==len(ar_temp)-1:
                    flux[i]=self.bbody_temp_powerlaw(t,self.variables['exp_surface'])*(tmax-tmin)/(num_points-1)/2
                else:
                    flux[i]=self.bbody_temp_powerlaw(t,self.variables['exp_surface'])*(tmax-tmin)/(num_points-1)
                i+=1
            flux_sur=np.sum(flux,axis=0)


            for key in self.abundance_dict:
                dust_abs=self.data_dict[key] 
                dust_flux=dust_abs*flux_sur
                self.surface_flux_individual[key]=dust_flux
                if one_output:
                    flux_surface_tot+=self.abundance_dict[key]*self.surface_flux_individual[key]
            if one_output:
                return flux_surface_tot

    def set_emission_lines(self,LTE=True,one_output=False,scaled=True,output_quantities=False,debug=False,
                           fast_norm=True,debug_interp=False):
        if output_quantities:
            output_dict={}
        if self.radial_version:
            exp=(2-self.variables['exp_emission'])/self.variables['exp_emission']

        else:
            exp=self.variables['exp_emission']

        idxs=[]
        temps=[]

        emission_flux=np.zeros(self.nwav)


        for specie in self.slab_dict:
            if output_quantities:
                output_dict[specie]={}
            if 'ColDens' in list(self.slab_dict[specie]):
                col_range=False
            else:
                col_range=True
            
            if debug:                
                print('--------------')
                print(f'Slab of {specie}')
            
            #if we have a single temperature (temis)
            if 'temis' in self.slab_dict[specie]:
                
                temis=self.slab_dict[specie]['temis']
                dens=self.slab_dict[specie]['ColDens']
                if scaled:
                    numerator= 1e23*np.pi*((self.slab_dict[specie]['radius']*self.au)**2)/((self.variables['distance']*self.parsec)**2)
                    if self.cosi:
                        scale = numerator*degree_to_cos(self.variables['incl'])
                    else:
                        scale = numerator

                else:
                    numerator= 1e23*np.pi*((self.au)**2)/((self.variables['distance']*self.parsec)**2)
                    if self.cosi:
                        scale = numerator*degree_to_cos(self.variables['incl'])
                    else:
                        scale = numerator

                if debug:
                    print('Single slab version')
                    print('Temp,coldens',temis,dens)

                #now interpolating the column densities on a linear scale
                species_flux=np.zeros_like(self.slab_data[specie][0])



                # Doing a 2D interpolation for the edges at tmin tmax, and coldens_tmin,coldens_tmax
                # first we are selecting the important edges

                # lower edge

                idx_dens=[]

                arg_dens=np.argmin(abs(self.slab_parameters['col']-dens))
                if self.slab_parameters['col'][arg_dens]<dens:
                    idx_dens.append(arg_dens)
                    idx_dens.append(arg_dens+1)
                else:
                    idx_dens.append(arg_dens-1)
                    idx_dens.append(arg_dens)
                dens_lower=self.slab_parameters['col'][idx_dens[0]]
                dens_upper=self.slab_parameters['col'][idx_dens[1]]
                idx_temp=[]
                arg_temp=np.argmin(abs(self.slab_parameters['temp']-temis))
                if self.slab_parameters['temp'][arg_temp]<temis:
                    idx_temp.append(arg_temp)
                    idx_temp.append(arg_temp+1)
                else:
                    idx_temp.append(arg_temp-1)
                    idx_temp.append(arg_temp)

                temp_lower=self.slab_parameters['temp'][idx_temp[0]]
                temp_upper=self.slab_parameters['temp'][idx_temp[1]]
                if debug:
                    print('temp_lower',temp_lower)
                    print('temp_upper',temp_upper)
                    print('dens_lower',dens_lower)
                    print('dens_upper',dens_upper)


                # saving the respective spectra left right means temp and up down coldens
                a= self.slab_data[specie][idx_dens[0]][idx_temp[0]] # lower left
                b= self.slab_data[specie][idx_dens[0]][idx_temp[1]] # lower right
                c= self.slab_data[specie][idx_dens[1]][idx_temp[0]] # upper left
                d= self.slab_data[specie][idx_dens[1]][idx_temp[1]] # upper right

                #calculating the interpolation factors

                # delta is deltax*deltay
                delta_temp=temp_upper-temp_lower
                delta_col=dens_upper-dens_lower
                delta=delta_col*delta_temp

                fact_a=(dens_upper-dens)*(temp_upper-temis)
                fact_b=(dens_upper-dens)*(temis-temp_lower)
                fact_c=(dens-dens_lower)*(temp_upper-temis)
                fact_d=(dens-dens_lower)*(temis-temp_lower)

                # summing everthing up

                flux_species=scale*(fact_a*a+fact_b*b+fact_c*c+fact_d*d)/delta

                if not scaled:
                    self.emission_flux_individual[specie]=flux_species
                else:
                    self.emission_flux_individual_scaled[specie]=flux_species

                emission_flux+=flux_species
                if debug:
                    print('--------------')
                    print()

            
            
            
            
            
            
            # if we have a temperature range
            else:
                t_min,t_max=self.slab_dict[specie]['tmin'],self.slab_dict[specie]['tmax']

                #determining the scale factor
                if self.radial_version:
                    if scaled:
                        denominator=((self.variables['distance']*self.parsec)**2)*self.variables['exp_emission']*t_max**(2/self.variables['exp_emission'])
                        numerator= -1e23*2*np.pi*((self.slab_dict[specie]['radius']*self.au)**2)
                        if self.cosi:
                            scale = numerator/denominator*degree_to_cos(self.variables['incl'])
                        else:
                            scale = numerator/denominator

                    else:
                        denominator=((self.variables['distance']*self.parsec)**2)*self.variables['exp_emission']*t_max**(2/self.variables['exp_emission'])
                        numerator= -1e23*2*np.pi*((self.au)**2)
                        if self.cosi:
                            scale = numerator/denominator*degree_to_cos(self.variables['incl'])
                        else:
                            scale = numerator/denominator

                else:
                    if scaled:
                        scale = 1e23*np.pi*((self.au*self.slab_dict[specie]['radius'])**2)/((self.variables['distance']*self.parsec)**2)
                    else:
                        scale = 1e23*np.pi*((self.au)**2)/((self.variables['distance']*self.parsec)**2)


                if col_range:
                    interp_edge_first=True
                    # defining a column density at T min and at Tmax
                    # a exponential function is then assumed between those points
                    # the data is then interpolated (log normal) to get the column densities as all grid points
                    # the flux is then determined by having a linear interpolation between the column density at the grid points


                    dens_min,dens_max=self.slab_dict[specie]['ColDens_tmin'],self.slab_dict[specie]['ColDens_tmax']


                    # creating the law
                    dens_min_log=np.log10(dens_min)
                    dens_max_log=np.log10(dens_max)
                    if self.radial_version:
                        t_max_log=np.log10(t_max)
                        t_min_log=np.log10(t_min)
                        slope=(dens_max_log-dens_min_log)/(t_max_log-t_min_log)
                        temp_logs=np.log10(self.slab_parameters['temp'])
                    else:
                        slope=(dens_max_log-dens_min_log)/(t_max-t_min)

                    respective_cols=np.zeros_like(self.slab_parameters['temp'],'float')
                    for t in range(len(self.slab_parameters['temp'])):
                        if debug:
                            if self.radial_version:
                                print(dens_min_log,(temp_logs[t]-t_min_log)*slope)

                            else:
                                print(dens_min_log,(self.slab_parameters['temp'][t]-t_min)*slope)
                        if self.slab_parameters['temp'][t]<=t_max and self.slab_parameters['temp'][t]>=t_min: #+1*self.slab_temp_steps -1*self.slab_temp_steps
                            if self.radial_version:
                                respective_cols[t]=float(dens_min_log+(temp_logs[t]-t_min_log)*slope)
                            else:
                                respective_cols[t]=float(dens_min_log+(self.slab_parameters['temp'][t]-t_min)*slope)

                    if debug: print('respective_cols',respective_cols)
                    cols_ar=10**respective_cols
                    if debug and max(respective_cols)>100:
                        print(dens_min_log)
                        print(dens_max_log)
                        print(t_min)
                        print(t_max)
                        print('Max respctive_cols',max(respective_cols))
                    #print('cols_ar',respective_cols)
                    if output_quantities:
                        output_dict[specie]['ColDens_tmin']=t_min
                        output_dict[specie]['ColDens_slope']=slope
                        output_dict[specie]['logColDens_min']=dens_min_log

                    if debug:
                        print('Slope',slope)
                        print('respective cols',respective_cols)
                        print('Cols ar',cols_ar)
                        plt.plot(self.slab_parameters['temp'],cols_ar)
                        plt.scatter([t_min,t_max],[dens_min,dens_max])
                        T_g_tot,NHtot_tot=np.meshgrid(self.slab_parameters['temp'],self.slab_parameters['col'])
                        plt.scatter(T_g_tot,NHtot_tot,marker='+')
                        plt.yscale('log')
                        if self.radial_version:
                            plt.xscale('log')

                        plt.xlabel('Temperature')
                        plt.ylabel('ColDens')
                        plt.title('Temperature Column density relation')
                        plt.show()

                    #now interpolating the column densities on a linear scale
                    species_flux=np.zeros_like(self.slab_data[specie][0])
                    for i in range(len(cols_ar)):
                        #getting the two closest point in the array
                        idx_col_list=[]
                        dens=cols_ar[i]

                        arg_min_dens=np.argmin(abs(self.slab_parameters['col']-dens))
                        if self.slab_parameters['col'][arg_min_dens]<dens:
                            idx_col_list.append(arg_min_dens)
                            idx_col_list.append(arg_min_dens+1)
                        else:
                            idx_col_list.append(arg_min_dens-1)
                            idx_col_list.append(arg_min_dens)
                        #if the col dens are outside the trained range it is disregarded. 
                        in_array=True
                        if idx_col_list[0]<0 or idx_col_list[1]>=len(self.slab_parameters['col']):
                            in_array=False
                        if in_array:
                            if debug:
                                print(i,idx_col_list)
                                print('Dens, dens in array',dens,self.slab_parameters['col'][idx_col_list])
                            species_flux[i]=self.slab_data[specie][idx_col_list[0]][i]+(self.slab_data[specie][idx_col_list[1]][i]-self.slab_data[specie][idx_col_list[0]][i])*(cols_ar[i]-self.slab_parameters['col'][idx_col_list[0]])/(self.slab_parameters['col'][idx_col_list[1]]-self.slab_parameters['col'][idx_col_list[0]])  
                        elif dens==self.slab_parameters['col'][0]:
                            species_flux[i]=self.slab_data[specie][0][i]
                        elif dens==self.slab_parameters['col'][-1]:
                            species_flux[i]=self.slab_data[specie][-1][i]


                    # Doing a 2D interpolation for the edges at tmin tmax, and coldens_tmin,coldens_tmax
                    # first we are selecting the important edges

                    # lower edge

                    idx_dens_min=[]

                    arg_min_dens=np.argmin(abs(self.slab_parameters['col']-dens_min))
                    if self.slab_parameters['col'][arg_min_dens]<dens_min:
                        idx_dens_min.append(arg_min_dens)
                        idx_dens_min.append(arg_min_dens+1)
                    else:
                        idx_dens_min.append(arg_min_dens-1)
                        idx_dens_min.append(arg_min_dens)
                    dens_lower=self.slab_parameters['col'][idx_dens_min[0]]
                    dens_upper=self.slab_parameters['col'][idx_dens_min[1]]
                    idx_temp_min=[]
                    arg_min_temp=np.argmin(abs(self.slab_parameters['temp']-t_min))
                    if self.slab_parameters['temp'][arg_min_temp]<t_min:
                        idx_temp_min.append(arg_min_temp)
                        idx_temp_min.append(arg_min_temp+1)
                    else:
                        idx_temp_min.append(arg_min_temp-1)
                        idx_temp_min.append(arg_min_temp)

                    temp_lower=self.slab_parameters['temp'][idx_temp_min[0]]
                    temp_upper=self.slab_parameters['temp'][idx_temp_min[1]]
                    if debug_interp:
                        print('temp_lower',temp_lower)
                        print('temp_upper',temp_upper)
                        print('dens_lower',dens_lower)
                        print('dens_upper',dens_upper)


                    # saving the respective spectra left right means temp and up down coldens
                    a= self.slab_data[specie][idx_dens_min[0]][idx_temp_min[0]] # lower left
                    b= self.slab_data[specie][idx_dens_min[0]][idx_temp_min[1]] # lower right
                    c= self.slab_data[specie][idx_dens_min[1]][idx_temp_min[0]] # upper left
                    d= self.slab_data[specie][idx_dens_min[1]][idx_temp_min[1]] # upper right

                    #calculating the interpolation factors

                    # delta is deltax*deltay
                    delta_temp=temp_upper-temp_lower
                    delta_col=dens_upper-dens_lower
                    delta=delta_col*delta_temp

                    fact_a=(dens_upper-dens_min)*(temp_upper-t_min)
                    fact_b=(dens_upper-dens_min)*(t_min-temp_lower)
                    fact_c=(dens_min-dens_lower)*(temp_upper-t_min)
                    fact_d=(dens_min-dens_lower)*(t_min-temp_lower)

                    # summing everthing up

                    lower_edge=(fact_a*a+fact_b*b+fact_c*c+fact_d*d)/delta * t_min**exp



                    if debug or debug_interp:
                        print('delta',delta)
                        print('fact_a',fact_a/delta)
                        print('fact_b',fact_b/delta)
                        print('fact_c',fact_c/delta)
                        print('fact_d',fact_d/delta)
                        print('lower edge mean',np.mean(lower_edge))

                    # upper edge

                    idx_dens_max=[]
                    arg_max_dens=np.argmin(abs(self.slab_parameters['col']-dens_max))
                    if self.slab_parameters['col'][arg_max_dens]<dens_max:
                        idx_dens_max.append(arg_max_dens)
                        idx_dens_max.append(arg_max_dens+1)
                    else:
                        idx_dens_max.append(arg_max_dens-1)
                        idx_dens_max.append(arg_max_dens)

                    dens_lower=self.slab_parameters['col'][idx_dens_max[0]]
                    dens_upper=self.slab_parameters['col'][idx_dens_max[1]]
                    idx_temp_max=[]
                    arg_max_temp=np.argmin(abs(self.slab_parameters['temp']-t_max))
                    if self.slab_parameters['temp'][arg_max_temp]<t_max:
                        idx_temp_max.append(arg_max_temp)
                        idx_temp_max.append(arg_max_temp+1)
                    else:
                        idx_temp_max.append(arg_max_temp-1)
                        idx_temp_max.append(arg_max_temp)


                    temp_lower=self.slab_parameters['temp'][idx_temp_max[0]]
                    temp_upper=self.slab_parameters['temp'][idx_temp_max[1]]

                    if debug_interp:
                        print('temp_lower',temp_lower)
                        print('temp_upper',temp_upper)
                        print('dens_lower',dens_lower)
                        print('dens_upper',dens_upper)

                    # saving the respective spectra left right means temp and up down coldens
                    a= self.slab_data[specie][idx_dens_max[0]][idx_temp_max[0]] # lower left
                    b= self.slab_data[specie][idx_dens_max[0]][idx_temp_max[1]] # lower right
                    c= self.slab_data[specie][idx_dens_max[1]][idx_temp_max[0]] # upper left
                    d= self.slab_data[specie][idx_dens_max[1]][idx_temp_max[1]] # upper right

                    #calculating the interpolation factors

                    # delta is deltax*deltay
                    delta_temp=temp_upper-temp_lower
                    delta_col=dens_upper-dens_lower
                    delta=delta_col*delta_temp

                    fact_a=(dens_upper-dens_max)*(temp_upper-t_max)
                    fact_b=(dens_upper-dens_max)*(t_max-temp_lower)
                    fact_c=(dens_max-dens_lower)*(temp_upper-t_max)
                    fact_d=(dens_max-dens_lower)*(t_max-temp_lower)

                    # summing everthing up

                    upper_edge=(fact_a*a+fact_b*b+fact_c*c+fact_d*d)/delta* t_max**exp

                    if debug or debug_interp:
                        print('delta',delta)
                        print('fact_a',fact_a/delta)
                        print('fact_b',fact_b/delta)
                        print('fact_c',fact_c/delta)
                        print('fact_d',fact_d/delta)
                        print('Upper edge mean',np.mean(upper_edge))

                    idx_col_list=[0]

                else:
                    interp_edge_first=False
                    #evaluating where in the grid the giving column density is stored
                    # if it isn't part of the array the two closest densities are interpolated
                    #checking which indices corrispond to the temperature range
                    # future improvement is a density gradient and relation between temperature and density

                    dens=self.slab_dict[specie]['ColDens']
                    if output_quantities:
                        output_dict[specie]['ColDens_tmin']=t_min
                        output_dict[specie]['ColDens_slope']=0.0
                        output_dict[specie]['logColDens_min']=np.log10(dens)
                    idx_col_list=[]
                    if dens in self.slab_parameters['col']:
                        idx_col=np.where(self.slab_parameters['col']==dens)[0][0]
                        idx_col_list.append(idx_col)
                    else:
                        arg_min_dens=np.argmin(abs(self.slab_parameters['col']-dens))
                        if self.slab_parameters['col'][arg_min_dens]<dens:
                            idx_col_list.append(arg_min_dens)
                            idx_col_list.append(arg_min_dens+1)
                        else:
                            idx_col_list.append(arg_min_dens-1)
                            idx_col_list.append(arg_min_dens)

                min_found=False
                max_found=False
                for idx_t in range(len(self.slab_parameters['temp'])):
                    if (self.slab_parameters['temp'][idx_t]>=t_min) and not min_found:
                        idx_tmin=idx_t
                        min_found=True
                    if (self.slab_parameters['temp'][idx_t]>t_max) and not max_found:
                        idx_tmax=idx_t-1
                        max_found=True
                        break
                if debug:
                    print('Temp range', t_min,t_max)
                    print('Temp inside range',self.slab_parameters['temp'][idx_tmin],self.slab_parameters['temp'][idx_tmax])
                #deciding if the small temperature range version needs to be used
                grid_p=True
                t_range=idx_tmax-idx_tmin
                if t_range>=2:
                    #full version
                    # this means that there is an inner part consisting atleast of one temperature 
                    range_version=0
                elif t_range==1:
                    #getting rid of inner part
                    range_version=1                
                else:
                    #smallest version
                    # even the inner contribution of the edges should be gone

                    # if a grid points in in between the contribtutions
                    range_version=2
                    # if no grid point between contributions
                    if idx_tmax<idx_tmin:
                        grid_p=False
                    else:
                        grid_p=True
                if debug:
                    print('Range verion:',range_version)
                    print('Grid_p',grid_p)
                for idx_col in idx_col_list:
                    if output_quantities and len(idx_col_list)==2:
                        output_dict[specie][idx_col]={}
                    if not col_range:
                        species_flux=self.slab_data[specie][idx_col]

                    #summing the flux at every relevant grid point
                    if range_version==0:
                        slab_data_select=species_flux[idx_tmin+1:idx_tmax].copy()          
                        temp_paras=self.slab_parameters['temp'][idx_tmin+1:idx_tmax]**exp
                        if output_quantities:
                            output_dict[specie]['inner_part_temp']=self.slab_parameters['temp'][idx_tmin+1:idx_tmax]
                        #temp_paras=temp_paras/np.mean(temp_paras)
                        for i in range(len(temp_paras)):
                            slab_data_select[i]*=temp_paras[i]
                    else:
                        if debug:
                            print('SKIPPING INNER PART')
                    if debug:
                        if range_version==0:

                            print(np.shape(slab_data_select))
                            print('The inner part goes from/to',self.slab_parameters['temp'][idx_tmin+1],self.slab_parameters['temp'][idx_tmax-1])
                        else:
                            if debug:
                                print('SKIPPING INNER PART')

                        print('The slabs that are just inside are',self.slab_parameters['temp'][idx_tmin],self.slab_parameters['temp'][idx_tmax])
                        print('The slabs that are just outside are',self.slab_parameters['temp'][idx_tmin-1],self.slab_parameters['temp'][idx_tmax+1])

                    upper_slab_in=species_flux[idx_tmax]
                    lower_slab_in=species_flux[idx_tmin]


                    upper_slab_in_exp=upper_slab_in*(self.slab_parameters['temp'][idx_tmax])**exp
                    lower_slab_in_exp=lower_slab_in*(self.slab_parameters['temp'][idx_tmin])**exp
                    if debug:
                        print('Temperature powerlaw begin and end')
                        print(self.slab_parameters['temp'][idx_tmin],self.slab_parameters['temp'][idx_tmin]**exp)
                        print(self.slab_parameters['temp'][idx_tmax],self.slab_parameters['temp'][idx_tmax]**exp)

                    upper_slab_out=species_flux[idx_tmax+1]
                    lower_slab_out=species_flux[idx_tmin-1]

                    if (not interp_edge_first) or debug_interp:
                        upper_edge=(upper_slab_in+(upper_slab_out-upper_slab_in)*(t_max-self.slab_parameters['temp'][idx_tmax])/self.slab_temp_steps)*(t_max)**exp
                        lower_edge=(lower_slab_out+(lower_slab_in-lower_slab_out)*(t_min-self.slab_parameters['temp'][idx_tmin-1])/self.slab_temp_steps)*(t_min)**exp
                    if debug_interp:
                        print('Old lower edge mean',np.mean(lower_edge))
                        print('Old upper edge mean',np.mean(upper_edge))
                    if range_version==0:
                        if output_quantities:
                            if len(idx_col_list)==1:
                                output_dict[specie]['inner_part']=np.sum(slab_data_select,axis=1)*self.slab_temp_steps 
                            else:
                                output_dict[specie][idx_col]['inner_part']=np.sum(slab_data_select,axis=1)*self.slab_temp_steps 

                        inner_part=np.sum(slab_data_select,axis=0)*self.slab_temp_steps
                    else:
                        if debug:
                            print('SKIPPING INNER PART')
                    if range_version!=2:
                        lower_in=lower_slab_in_exp*self.slab_temp_steps/2
                        upper_in=upper_slab_in_exp*self.slab_temp_steps/2
                    else:
                        upper_in=np.zeros_like(upper_edge)
                        lower_in=np.zeros_like(lower_edge)
                    if output_quantities:
                        output_dict[specie]['upper_in_temp']=self.slab_parameters['temp'][idx_tmax]
                        output_dict[specie]['lower_in_temp']=self.slab_parameters['temp'][idx_tmin]

                        if len(idx_col_list)==1:
                            output_dict[specie]['upper_in']=np.sum(upper_in)
                            output_dict[specie]['lower_in']=np.sum(lower_in)
                        else:
                            output_dict[specie][idx_col]['upper_in']=np.sum(upper_in)
                            output_dict[specie][idx_col]['lower_in']=np.sum(lower_in)

                    if debug:
                        if range_version==0:
                            print('Comparing inner edges')
                            print('Flux ratio upper in and last in inner part',np.sum(upper_slab_in_exp)/np.sum(slab_data_select[-1]))

                            print('Len inner part',len(slab_data_select))
                            print('Sum inner part',np.sum(inner_part))
                            print('2*len*upper in',2*len(slab_data_select)*np.sum(upper_in))
                            print('Flux ratio lower in and first in inner part',np.sum(lower_slab_in_exp)/np.sum(slab_data_select[0]))
                            print('Len inner part',len(slab_data_select))
                            print('Sum inner part',np.sum(inner_part))
                            print('2*len*lower in',2*len(slab_data_select)*np.sum(lower_in))
                        else:
                            print('SKIPPING INNER PART')


                    if grid_p:
                        lower_out=(lower_edge+lower_slab_in_exp)/2*(self.slab_parameters['temp'][idx_tmin]-t_min)
                        upper_out=(upper_slab_in_exp+upper_edge)/2*(t_max-self.slab_parameters['temp'][idx_tmax])

                    else:
                        lower_out=np.zeros_like(lower_edge)
                        upper_out=(lower_edge+upper_edge)/2*(t_max-t_min)
                    if output_quantities:

                        output_dict[specie]['lower_out_temp']=t_min
                        output_dict[specie]['upper_out_temp']=t_max

                        if len(idx_col_list)==1:
                            output_dict[specie]['lower_out']=np.sum(lower_out)
                            output_dict[specie]['upper_out']=np.sum(upper_out)
                        else:
                            output_dict[specie][idx_col]['lower_out']=np.sum(lower_out)
                            output_dict[specie][idx_col]['upper_out']=np.sum(upper_out)

                    if debug:
                        print('-------------')
                        print('Comparing outer edges')
                        print('Ratio lower edge and lower_slab_in_exp',np.sum(lower_edge)/np.sum(lower_slab_in_exp))
                        print('Ratio upper edge and upper_slab_in_exp',np.sum(upper_edge)/np.sum(upper_slab_in_exp))


                    if debug:
                        if range_version==0:

                            sum_in=np.sum(inner_part)
                        else:
                            sum_in=0.0
                            if debug:
                                print('SKIPPING INNER PART')

                        sum_low_in=np.sum(lower_in)
                        sum_low_out=np.sum(lower_out)
                        sum_upper_in=np.sum(upper_in)
                        sum_upper_out=np.sum(upper_out)
                        tot_sum=sum_in+sum_low_in+sum_low_out+sum_upper_in+sum_upper_out
                        print('--------------')
                        print('Contributions: (tmin, tmax, percent)')
                        if range_version==0:
                            print('inner part',self.slab_parameters['temp'][idx_tmin+1]-self.slab_temp_steps/2,self.slab_parameters['temp'][idx_tmax-1]+self.slab_temp_steps/2,np.round(sum_in/tot_sum*100,1))
                        else:
                            if debug:
                                print('SKIPPING INNER PART')
                        if range_version==2:
                            print('lower in',self.slab_parameters['temp'][idx_tmin],self.slab_parameters['temp'][idx_tmin],np.round(sum_low_in/tot_sum*100,1))
                        else:
                            print('lower in',self.slab_parameters['temp'][idx_tmin],self.slab_parameters['temp'][idx_tmin]+self.slab_temp_steps/2,np.round(sum_low_in/tot_sum*100,1))

                        print('lower out',t_min,self.slab_parameters['temp'][idx_tmin],np.round(sum_low_out/tot_sum*100,1))
                        if range_version==2:
                            print('upper in',self.slab_parameters['temp'][idx_tmax],self.slab_parameters['temp'][idx_tmax],np.round(sum_upper_in/tot_sum*100,1))
                        else:
                            print('upper in',self.slab_parameters['temp'][idx_tmax]-self.slab_temp_steps/2,self.slab_parameters['temp'][idx_tmax],np.round(sum_upper_in/tot_sum*100,1))
                        if grid_p:

                            print('upper out',self.slab_parameters['temp'][idx_tmax],t_max,np.round(sum_upper_out/tot_sum*100,1))
                        else:
                            print('upper out',t_min,t_max,np.round(sum_upper_out/tot_sum*100,1))
                        print('--------------')
                    if range_version!=0:
                        if debug:
                            print('SKIPPING INNER PART')
                        inner_part=np.zeros_like(lower_out)

                    flux_species=(inner_part+lower_in+upper_in+upper_out+lower_out).copy()

                    '''
                    Normalization
                    only calculated in first density iteration
                    '''
                    if not self.radial_version:
                        if len(idx_col_list)==1 or idx_col==idx_col_list[0]:
                            if not fast_norm:
                                if range_version==0:

                                    power_in=np.sum(temp_paras)*self.slab_temp_steps
                                else:
                                    if debug:
                                        print('SKIPPING INNER PART')

                                    power_in=0.0
                                if range_version!=2:
                                    powers_edges_1=((self.slab_parameters['temp'][idx_tmax])**exp)*(self.slab_temp_steps/2+(t_max-self.slab_parameters['temp'][idx_tmax])/2)
                                    powers_edges_2=((self.slab_parameters['temp'][idx_tmin])**exp)*(self.slab_temp_steps/2+(self.slab_parameters['temp'][idx_tmin]-t_min)/2)
                                else:
                                    if grid_p:

                                        powers_edges_1=((self.slab_parameters['temp'][idx_tmax])**exp)*((t_max-self.slab_parameters['temp'][idx_tmax])/2)
                                        powers_edges_2=((self.slab_parameters['temp'][idx_tmin])**exp)*((self.slab_parameters['temp'][idx_tmin]-t_min)/2)
                                    else:
                                        powers_edges_1=0.0
                                        powers_edges_2=0.0
                                if grid_p:        
                                    power_edges_up=(t_max)**exp*(t_max-self.slab_parameters['temp'][idx_tmax])/2
                                    power_edges_low=(t_min)**exp*(self.slab_parameters['temp'][idx_tmin]-t_min)/2
                                else:
                                    power_edges_up=(t_max)**exp*(t_max-t_min)/2
                                    power_edges_low=(t_min)**exp*(t_max-t_min)/2

                                if debug:
                                    print('Contributions to normalization constant:')
                                    print('power_edges_1',powers_edges_1)
                                    print('power_edges_2',powers_edges_2)
                                    print('power_edges_low',power_edges_low)
                                    print('power_edges_up',power_edges_up)
                                    print('power_in',power_in)


                                norm=power_in+powers_edges_1+powers_edges_2+power_edges_low+power_edges_up
                            else:
                                #mathematical approach
                                norm=(t_max**(exp+1)-t_min**(exp+1))/(exp+1)
                                if debug:
                                    print('Short normalization approach')
                            if output_quantities:
                                output_dict[specie]['norm']=norm
                            if debug:
                                print('Normalization constant',norm)

                    if self.radial_version:
                        flux_species=flux_species*scale
                    else:
                        flux_species=flux_species*scale/norm
                    if len(idx_col_list)>1:
                        if debug:
                            print('Combining different column density slabs')
                            print('tot_flux of this iteration',np.sum(flux_species))
                        if idx_col==idx_col_list[0]:
                            flux_species_0=flux_species.copy()

                if len(idx_col_list)>1:
                    flux_species_1=flux_species.copy()
                    lower_dens=self.slab_parameters['col'][idx_col_list[0]]
                    higher_dens=self.slab_parameters['col'][idx_col_list[1]]
                    flux_species=flux_species_0+(flux_species_1-flux_species_0)*(dens-lower_dens)/(higher_dens-lower_dens)
                    fact=(dens-lower_dens)/(higher_dens-lower_dens)

                    if output_quantities:
                        key_out_list=['inner_part','upper_in','upper_out','lower_in','lower_out']
                        for key_out in key_out_list: 
                            if key_out in output_dict[specie][idx_col_list[0]]:
                                output_dict[specie][key_out]=output_dict[specie][idx_col_list[0]][key_out]+fact*(output_dict[specie][idx_col_list[1]][key_out]-output_dict[specie][idx_col_list[0]][key_out])
                        output_dict[specie].pop(idx_col_list[0])
                        output_dict[specie].pop(idx_col_list[1])
                    if debug:
                        print('---------------')
                        print('Density is',dens)
                        print('This is between',lower_dens,higher_dens)
                        print(f'Its {np.round((dens-lower_dens)/(higher_dens-lower_dens)*100)} % of the way')
                        print(f'The individual fluxes are {np.sum(flux_species_0)} and {np.sum(flux_species_1)}')
                        print('The new flux is',np.sum(flux_species))
                        print('---------------')


                if not scaled:
                    self.emission_flux_individual[specie]=flux_species
                else:
                    self.emission_flux_individual_scaled[specie]=flux_species
                emission_flux+=flux_species
                if debug:
                    print('--------------')
                    print()
        if output_quantities:
            return output_dict
        if one_output:
            return emission_flux

    def extract_emission_quantities(self,low_contribution=0.15,high_contribution=0.85,debug=False):
        '''
        This module extracts the important quantities from the emitting regio of the model.
        Output: the radius, temperature and (optionally) columndensity range in which 1-low_contribution-high_contribution
        of the flux is emitted
        Approach: 
        - Running set_emission without summing over the temperature range
        - selecting the lower and upper temperature at which low_contribution and high_contribution
          of the total flux are emitted
        - if a column density range is used: using ColDens_tmin and ColDens_tmax to calculate the column density
          at the new lower and upper temperature
        - using the normalzied powerlaw to calculate r*t^q/t_norm at the new upper and lower temperature
          this corrispond to the emitting radius of the specific temperatures
          important is that the radius is used in it's square therefore t**q/nrom has to be taken in the sqrt
        '''
        
        '''
        DOTO: make it possible for single coldens: meaning idx_col has to be included!!!
        
        '''
        #caling the emission_lines function to get all information necessary
        output_dict=self.set_emission_lines(one_output=False,scaled=False,output_quantities=True)
        if debug:
            for key in output_dict: 
                print('-----------')
                print(key)
                for key1 in output_dict[key]:
                    print(key1,output_dict[key][key1])
        
            print('-----------')
        exclude_list=['ColDens_slope','logColDens_min','ColDens_tmin','norm'] #all keys that are not added to the total flux
        results={}
        for species in output_dict:
            if debug:
                print('----------------')
                print('Keys in output dict')
                print(species)
                for key in output_dict[species]:
                    print(key)
                    print(output_dict[species][key])
            results[species]={}
            if debug:
                print('-----------')
                print(species)
                
                
                
            if 'temis' in self.slab_dict[species]:
                # for the simple slab model the extraction of the quantities doesn't make much sense
                # however, we want the analysing tools to run without problems
                # therefore we are inserting the corrisponding values quickly
                results[species]['tmin,tmax']=[self.slab_dict[species]['temis'],self.slab_dict[species]['temis']]
                results[species]['cmin,cmax']=[self.slab_dict[species]['ColDens'],self.slab_dict[species]['ColDens']]
                results[species]['rout,rin']=[self.slab_dict[species]['radius'],0] 
                results[species]['radius_eff']=self.slab_dict[species]['radius']
            else:
                tot_flux=0
                for key in output_dict[species]:
                    if key not in exclude_list and 'temp' not in key:
                        if debug:
                            print('Contributing to total flux:',key)
                        if 'inner_part'==key:
                            for i in range(len(output_dict[species]['inner_part'])):
                                tot_flux+=output_dict[species][key][i]

                        else:
                            tot_flux+=output_dict[species][key]
                if debug:
                    print('Total flux (w/o) radius and norm',tot_flux)
                order_tmin_to_tmax=['lower_out','lower_in','inner_part','upper_in','upper_out']
                summed_to_t=0
                if debug:
                    plt.figure()
                found_lower_lim=False
                found_upper_lim=False
                summed_to_t_previous=0

                if 'inner_part' in output_dict[species]:
                    for part in order_tmin_to_tmax:
                        if part=='inner_part':
                                for i in range(len(output_dict[species][part])):
                                    summed_to_t+=output_dict[species][part][i]

                                    if debug:
                                        plt.scatter(output_dict[species][part+'_temp'][i],summed_to_t/tot_flux)
                                    if summed_to_t<=low_contribution*tot_flux:
                                        t_at_min=output_dict[species][part+'_temp'][i]

                                    elif not found_lower_lim:
                                        found_lower_lim=True
                                        if debug:
                                            print('t_at_min on grid',t_at_min)
                                        target_dens=low_contribution*tot_flux
                                        t_up=output_dict[species][part+'_temp'][i]
                                        t_at_min=(target_dens-summed_to_t_previous)/(summed_to_t-summed_to_t_previous)*(t_up-t_previous)+t_previous

                                        if debug:
                                            print('Dens before and after',summed_to_t_previous,summed_to_t)
                                            print('t_at_min interpolate',t_at_min)
                                            print('target_dens',target_dens)
                                            print('t_up,t_previous',t_up,t_previous)

                                    if summed_to_t<=high_contribution*tot_flux:
                                        t_at_max=output_dict[species][part+'_temp'][i]



                                    elif not found_upper_lim:
                                        found_upper_lim=True

                                        if debug:
                                            print('t_at_max on grid',t_at_max)
                                        target_dens=high_contribution*tot_flux
                                        t_up=output_dict[species][part+'_temp'][i]
                                        t_at_max=(target_dens-summed_to_t_previous)/(summed_to_t-summed_to_t_previous)*(t_up-t_previous)+t_previous
                                        if debug:
                                            print('Dens before and after',summed_to_t_previous,summed_to_t)
                                            print('t_at_min interpolate',t_at_max)
                                            print('target_dens',target_dens)
                                            print('t_up,t_previous',t_up,t_previous)

                                    summed_to_t_previous+=output_dict[species][part][i]
                                    t_previous=output_dict[species][part+'_temp'][i]    
                        else:
                            summed_to_t+=output_dict[species][part]
                            if debug:
                                plt.scatter(output_dict[species][part+'_temp'],summed_to_t/tot_flux)
                            if summed_to_t<=low_contribution*tot_flux:
                                t_at_min=output_dict[species][part+'_temp']
                            elif not found_lower_lim: 
                                found_lower_lim=True
                                if part!='lower_out':

                                    if debug:
                                        print('t_at_min on grid',t_at_min)
                                    if part!='lower_out':
                                        target_dens=low_contribution*tot_flux
                                        t_up=output_dict[species][part+'_temp']
                                        t_at_min=(target_dens-summed_to_t_previous)/(summed_to_t-summed_to_t_previous)*(t_up-t_previous)+t_previous
                                        if debug:
                                            print('Dens before and after',summed_to_t_previous,summed_to_t)
                                            print('t_at_min interpolate',t_at_min)
                                            print('target_dens',target_dens)
                                            print('t_up,t_previous',t_up,t_previous)
                                else:
                                    t_at_min=self.slab_dict[species]['tmin']
                                    if debug:
                                        print('FIRST STEP IS ALREADY ABOVE THE SET LOWER LIM')
                            if summed_to_t<=high_contribution*tot_flux:
                                t_at_max=output_dict[species][part+'_temp']



                            elif not found_upper_lim: 
                                found_upper_lim=True
                                
                                if part!='lower_out':
                                    if debug:
                                        print('t_at_max on grid',t_at_max)
                                    target_dens=high_contribution*tot_flux
                                    t_up=output_dict[species][part+'_temp']
                                    t_at_max=(target_dens-summed_to_t_previous)/(summed_to_t-summed_to_t_previous)*(t_up-t_previous)+t_previous
                                    if debug:
                                        print('Dens before and after',summed_to_t_previous,summed_to_t)
                                        print('t_at_min interpolate',t_at_max)
                                        print('target_dens',target_dens)
                                        print('t_up,t_previous',t_up,t_previous)
                                else:
                                    # if the lower out is already above the upper threshold
                                    # there is no good way of defining flux<=high_contributino*tot_flux
                                    # therefore we are simply taking the first grid point above the lower limit
                                    t_at_max=output_dict[species]['lower_in_temp']

                            summed_to_t_previous+=output_dict[species][part]
                            t_previous=output_dict[species][part+'_temp']     

                else:
                    #if the temperature range is to small simply the minimal and maximal temperature are taken
                    t_at_max=self.slab_dict[species]['tmax']
                    t_at_min=self.slab_dict[species]['tmin']
                if debug:
                    print('t_at_min',t_at_min)
                    print('t_at_max',t_at_max)
                    plt.vlines(t_at_min,ymin=0,ymax=1)
                    plt.vlines(t_at_max,ymin=0,ymax=1)
                    plt.xlabel('T [K]')
                    plt.ylabel('Cumulative flux fraction')
                    plt.show()

                # translating temp at limits to col dens at limits   
                coldens_slope=output_dict[species]['ColDens_slope']
                coldens_t_min=output_dict[species]['ColDens_tmin']
                coldens_min=output_dict[species]['logColDens_min']
                if self.radial_version:
                    col_at_min=10**(coldens_min+(np.log10(t_at_min)-np.log10(coldens_t_min))*coldens_slope)
                    col_at_max=10**(coldens_min+(np.log10(t_at_max)-np.log10(coldens_t_min))*coldens_slope)
                else:
                    col_at_min=10**(coldens_min+(t_at_min-coldens_t_min)*coldens_slope)
                    col_at_max=10**(coldens_min+(t_at_max-coldens_t_min)*coldens_slope)

                if debug:
                    t_inbetween=np.linspace(t_at_min,t_at_max,1000,endpoint=True)
                    c_inbetween=[]
                    for t in t_inbetween:
                        if self.radial_version:
                            c_inbetween.append(10**(coldens_min+(np.log10(t)-np.log10(coldens_t_min))*coldens_slope))

                        else:
                            c_inbetween.append(10**(coldens_min+(t-coldens_t_min)*coldens_slope))
                    c_inbetween=np.array(c_inbetween)
                    plt.figure()
                    plt.plot(t_inbetween,c_inbetween)
                    plt.scatter(t_at_min,col_at_min)
                    plt.scatter(t_at_max,col_at_max)
                    plt.yscale('log')
                    if self.radial_version:
                        plt.xscale('log')

                    plt.xlabel('T [K]')
                    plt.ylabel(r'ColDens [$\rm cm^{-2}$]')
                    plt.show()



                if self.radial_version:
                    exp=(2-self.variables['exp_emission'])/(self.variables['exp_emission'])
                    if debug:
                        print('tmax,tmin',self.slab_dict[species]['tmax'],self.slab_dict[species]['tmin'])
                        print('t at max,t at min',t_at_max,t_at_min)


                    t_to_r=np.linspace(self.slab_dict[species]['tmin'],self.slab_dict[species]['tmax'],1000,endpoint=True)
                    t_to_r_inner=np.linspace(t_at_min,t_at_max,1000,endpoint=True)


                    radii=temp_to_rad(rmin=self.slab_dict[species]['radius'],t=t_to_r,q=self.variables['exp_emission'],tmax=self.slab_dict[species]['tmax'])
                    radii_inner=temp_to_rad(rmin=self.slab_dict[species]['radius'],t=t_to_r_inner,q=self.variables['exp_emission'],tmax=self.slab_dict[species]['tmax'])

                    r_at_min=temp_to_rad(rmin=self.slab_dict[species]['radius'],t=t_at_min,q=self.variables['exp_emission'],tmax=self.slab_dict[species]['tmax'])
                    r_at_max=temp_to_rad(rmin=self.slab_dict[species]['radius'],t=t_at_max,q=self.variables['exp_emission'],tmax=self.slab_dict[species]['tmax'])
                    if debug:
                        plt.figure()
                        plt.plot(radii,t_to_r)
                        plt.scatter(r_at_min,t_at_min)
                        plt.scatter(r_at_max,t_at_max)
                        plt.ylabel('T [K]')
                        plt.xlabel('R [au]')
                        plt.show() 

                        c_inbetween=[]
                        for t in t_to_r:
                            c_inbetween.append(10**(coldens_min+(np.log10(t)-np.log10(coldens_t_min))*coldens_slope))
                        c_inbetween=np.array(c_inbetween)
                        c_inner=[]
                        for t in t_to_r_inner:
                            c_inner.append(10**(coldens_min+(np.log10(t)-np.log10(coldens_t_min))*coldens_slope))
                        c_inner=np.array(c_inner)

                        plt.figure()
                        plt.plot(radii_inner,c_inner,lw=12,c='tab:red',zorder=0)

                        plt.scatter(radii,c_inbetween,c=t_to_r,vmin=min(t_to_r),vmax=max(t_to_r),s=35,cmap=cm)
                        #plt.scatter(r_at_min,col_at_min,c=t_at_min,vmin=min(t_to_r),vmax=max(t_to_r),s=42,cmap=cm,lw=2,edgecolors='tab:red')
                        #plt.scatter(r_at_max,col_at_max,c=t_at_max,vmin=min(t_to_r),vmax=max(t_to_r),s=45,cmap=cm,lw=2,edgecolors='tab:red')
                        plt.yscale('log')
                        plt.xscale('log')
                        plt.xlabel('R [au]')
                        plt.ylabel(r'ColDens [$\rm cm^{-2}$]')
                        plt.colorbar(label='T [K]')
                        plt.show()  
                    if self.cosi:
                        results[species]['radius_eff']=np.sqrt(r_at_min**2-r_at_max**2)*degree_to_cos(self.variables['incl'])

                    else:
                        results[species]['radius_eff']=np.sqrt(r_at_min**2-r_at_max**2)

                else:

                    #translating temperatures to radius
                    norm_total=output_dict[species]['norm']
                    exp=self.variables['exp_emission']
                    norm=(t_at_max**(exp+1)-t_at_min**(exp+1))/(exp+1)
                    norm=output_dict[species]['norm']

                    r_at_min=self.slab_dict[species]['radius']*(t_at_min**(exp/2)/np.sqrt(norm))
                    r_at_max=self.slab_dict[species]['radius']*(t_at_max**(exp/2)/np.sqrt(norm))


                    #calculate new total emission area
                    if debug:
                        print('tmax,tmin',self.slab_dict[species]['tmax'],self.slab_dict[species]['tmin'])
                        print('t at max,t at min',t_at_max,t_at_min)
                        print('exp_emission',exp)
                        print('tmax and min ** exp+1',t_at_max**(exp+1),t_at_min**(exp+1))
                        print('Norm total',norm_total)
                        print('New Norm',norm)

                    integral=(t_at_max**(exp+1)-t_at_min**(exp+1))/(self.slab_dict[species]['tmax']**(exp+1)-self.slab_dict[species]['tmin']**(exp+1))
                    new_effectiv_radius=np.sqrt(integral)*self.slab_dict[species]['radius']
                    results[species]['radius_eff']=new_effectiv_radius
                    if debug:
                        print('Ratio new limits to old limits',integral)
                        print('sqrt Ratio',np.sqrt(integral))
                        print('Old radius',self.slab_dict[species]['radius'])
                        print('New effectiv radius:',new_effectiv_radius)
                        t_inbetween=np.linspace(t_at_min,t_at_max,1000,endpoint=True)
                        r_inbetween=[]
                        for t in t_inbetween:
                            r_inbetween.append(self.slab_dict[species]['radius']*(t**(exp/2)/np.sqrt(norm)))
                        r_inbetween=np.array(r_inbetween)

                        #integrate over r/T to get r

                        integral_radius=np.sum(r_inbetween**2)*(t_inbetween[1]-t_inbetween[0])
                        integral_radius+=r_at_min**2*(t_inbetween[0]-t_at_min)
                        integral_radius+=r_at_max**2*(t_at_max-t_inbetween[-1])
                        print('Intregral over r/T from t_at_min to t_at_max:',np.sqrt(integral_radius))

                        plt.figure()
                        plt.plot(t_inbetween,r_inbetween)
                        plt.scatter(t_at_min,r_at_min)
                        plt.scatter(t_at_max,r_at_max)
                        plt.xlabel('T [K]')
                        plt.ylabel('R/T [au/K]')
                        plt.show()       
                        plt.figure()
                        plt.plot(r_inbetween,c_inbetween)
                        plt.scatter(r_at_min,col_at_min)
                        plt.scatter(r_at_max,col_at_max)
                        plt.yscale('log')
                        plt.xscale('log')
                        plt.xlabel('R/T [au/K]')
                        plt.ylabel(r'ColDens [$\rm cm^{-2}$]')
                        plt.show()       

                        plt.figure()
                        plt.scatter(r_inbetween,c_inbetween,c=t_inbetween,vmin=t_at_min,vmax=t_at_max,s=35,cmap=cm)
                        plt.scatter(r_at_min,col_at_min,c=t_at_min,vmin=t_at_min,vmax=t_at_max,s=35,cmap=cm)
                        plt.scatter(r_at_max,col_at_max,c=t_at_max,vmin=t_at_min,vmax=t_at_max,s=35,cmap=cm)
                        plt.yscale('log')
                        plt.xlabel('R/T [au/K]')
                        plt.ylabel(r'ColDens [$\rm cm^{-2}$]')
                        plt.colorbar(label='T [K]')
                        plt.show()  

                results[species]['tmin,tmax']=[t_at_min,t_at_max]
                results[species]['cmin,cmax']=[col_at_min,col_at_max]
                results[species]['rout,rin']=[r_at_min,r_at_max]    
        return results
       
        
        
        
    def run_model(self,variables,dust_species,slab_dict,output_all=False,scaled_emission=True,timeit=False):
        
        if timeit: time1=time()
        self.variables=variables
        if not self.slab_only_mode:
            self.abundance_dict=dust_species
            q=self.variables['q_mid']
            q_thin=self.variables['q_thin']
            self.variables['exp_midplane'] = (2.0 - q)/q

            self.variables['exp_surface'] = (2.0 - q_thin)/q_thin

        self.slab_dict=slab_dict
        
        q_emis=self.variables['q_emis']
        

        if old_version:
            self.variables['exp_emission']=(2-q_emis)/q_emis
        else:
            self.variables['exp_emission']=q_emis
        if not self.slab_only_mode:
 
            if self.rim_powerlaw:
                q_rim=self.variables['q_rim']
                self.variables['exp_rim'] = (2.0 - q_rim)/q_rim

            if timeit: time2=time()
            if self.variables['bb_star']:
                if self.variables['tstar']>9999:
                    from_array=False
                else:
                    from_array=True
                self.starspec=self.bbody(self.variables['tstar'],from_array=from_array)

                scale = 1e23*np.pi*((self.rsun*self.variables['rstar'])**2)/((self.variables['distance']*self.parsec)**2)
            else:
                scale = 1.0

            flux_star=self.starspec
            if timeit: time3=time()
        if output_all:
            if not self.slab_only_mode:

                if self.rim_powerlaw:
                    rim_flux=self.set_midplane(use_as_inner_rim=True)
                else:
                    rim_flux=self.bbody(self.variables['t_rim'],from_array=True)


                midplane_flux=self.set_midplane()


                self.set_surface(timeit=timeit)

            self.set_emission_lines(one_output=False,scaled=False)
            
            if not self.slab_only_mode:

        
                self.scaled_stellar_flux=scale*flux_star
                if old_version:
                    self.rim_flux=scale*rim_flux
                    self.midplane_flux=scale*midplane_flux
                else:
                    self.rim_flux=self.trans_flux*rim_flux
                    self.midplane_flux=self.trans_flux*midplane_flux

                #self.emission_flux=emission_flux #without the scaling because this is already applied


                for key in self.surface_flux_individual:
                    if old_version:
                        self.surface_flux_individual_scaled[key]=scale*self.surface_flux_individual[key]
                    else:
                        self.surface_flux_individual_scaled[key]=self.trans_flux*self.surface_flux_individual[key]


                return self.scaled_stellar_flux, self.rim_flux, self.midplane_flux, self.surface_flux_individual_scaled,self.emission_flux_individual.copy()
            else:
                return self.emission_flux_individual.copy()
                
        else:
            
            if not self.slab_only_mode:
                if self.rim_powerlaw:
                    unscaled_rim=self.set_midplane(use_as_inner_rim=True)
                else:
                    unscaled_rim=self.bbody(self.variables['t_rim'],from_array=True)

                rim_flux=self.variables['sc_ir']*unscaled_rim
                if timeit: time4=time()

                midplane_flux=self.variables['sc_mid']*self.set_midplane()
                if timeit: time5=time()
                surface_flux=self.set_surface(one_output=True)
                if timeit: time6=time()

            emission_flux=self.set_emission_lines(one_output=True,scaled=True)
            
            if self.slab_only_mode:
                tot_flux=emission_flux
                self.emission_flux=emission_flux
                self.tot_flux=tot_flux
            else:
                
                if timeit: time7=time()

                if old_version:
                    tot_flux=flux_star+rim_flux+midplane_flux+surface_flux
                    tot_flux=scale*tot_flux+emission_flux
                else:
                    tot_flux=scale*flux_star+self.trans_flux*(rim_flux+midplane_flux+surface_flux)+emission_flux

                self.scaled_stellar_flux=scale*flux_star
                if old_version:
                    self.rim_flux=scale*rim_flux
                    self.midplane_flux=scale*midplane_flux
                    self.surface_flux_tot=scale*surface_flux
                else:
                    self.rim_flux=self.trans_flux*rim_flux
                    self.midplane_flux=self.trans_flux*midplane_flux
                    self.surface_flux_tot=self.trans_flux*surface_flux

                self.emission_flux=emission_flux #without the scaling because this is already applied
                self.tot_flux=tot_flux
                if timeit:
                    time8=time()
                    print('Init',time2-time1)
                    print('Star',time3-time2)
                    print('Inner rim',time4-time3)
                    print('Midplane',time5-time4)
                    print('Surface',time6-time5)
                    print('Emission',time7-time6)
                    print('Summing up',time8-time7)


            return tot_flux
        
        
        
        

    def run_model_normalized(self,variables,dust_species,slab_dict,max_flux_obs,translate_scales=False,debug=False,timeit=False,save_continuum=continuum_penalty):
        '''
        This functions runs the model, but the scaling factors are between 0 and 1
        This is done by using the maximum value of every component and the maximum of the observation
        The emission from the different molecules is not rescales since it will not vary by so many
        Orders of magnitude, most likely 10**-2 - 10**2 is sufficient
        '''        
        if timeit: time1=time()
        self.variables=variables
        self.abundance_dict=dust_species
        self.slab_dict=slab_dict
        
        q=self.variables['q_mid']
        q_thin=self.variables['q_thin']
        q_emis=self.variables['q_emis']
        
        self.variables['exp_midplane'] = (2.0 - q)/q

        self.variables['exp_surface'] = (2.0 - q_thin)/q_thin
        
        if old_version:
            self.variables['exp_emission']=(2-q_emis)/q_emis
        else:
            self.variables['exp_emission']=q_emis
        if self.rim_powerlaw:
            q_rim=self.variables['q_rim']
            self.variables['exp_rim'] = (2.0 - q_rim)/q_rim
            
        if timeit: time2=time()
        if self.variables['bb_star']:
            if self.variables['tstar']>9999:
                from_array=False
            else:
                from_array=True
            self.starspec=self.bbody(self.variables['tstar'],from_array=from_array)
           
            scale = 1e23*np.pi*((self.rsun*self.variables['rstar'])**2)/((self.variables['distance']*self.parsec)**2)
        else:
            scale = 1.0
        flux_star=self.starspec
        if timeit: time3=time()
            
            
        if self.rim_powerlaw:
            unscaled_rim=self.set_midplane(use_as_inner_rim=True)
        else:
            unscaled_rim=self.bbody(self.variables['t_rim'],from_array=True)

        rim_flux=unscaled_rim
        if timeit: time4=time()


        midplane_flux=self.set_midplane()
        if timeit: time5=time()

        self.set_surface()
        if timeit: time6=time()

        self.set_emission_lines(one_output=False,scaled=True)
        if timeit: time7=time()

            
            
        self.scaled_stellar_flux=scale*flux_star
        self.rim_flux=self.variables['sc_ir']*rim_flux/max(rim_flux)*max_flux_obs
        self.midplane_flux=self.variables['sc_mid']*midplane_flux*max_flux_obs/max(midplane_flux)
        

        emission_flux=self.set_emission_lines(one_output=True,scaled=True)        
        
        tot_flux=self.scaled_stellar_flux+self.rim_flux+self.midplane_flux + emission_flux
        
 
        if debug:
            print(scale)
            print(np.max(rim_flux))
            print(np.max(midplane_flux))
        
        if translate_scales:       
            scale_paras_list=[]
            if old_version:
                scale_paras_list.append(self.variables['sc_ir']/scale*max_flux_obs/max(rim_flux))
                scale_paras_list.append(self.variables['sc_mid']/scale*max_flux_obs/max(midplane_flux))
            else:
                scale_paras_list.append(self.variables['sc_ir']/self.trans_flux*max_flux_obs/max(rim_flux))
                scale_paras_list.append(self.variables['sc_mid']/self.trans_flux*max_flux_obs/max(midplane_flux))
                
        surface_flux_tot=np.zeros_like(self.xnew)
        for key in self.surface_flux_individual:
            self.surface_flux_individual_scaled[key]=self.abundance_dict[key]*self.surface_flux_individual[key]*max_flux_obs/max(self.surface_flux_individual[key])
            surface_flux_tot+=self.surface_flux_individual_scaled[key]
            tot_flux+=self.surface_flux_individual_scaled[key]    
            if translate_scales:
                if old_version:
                    scale_paras_list.append(self.abundance_dict[key]/scale*max_flux_obs/max(self.surface_flux_individual[key]))
                else:
                    scale_paras_list.append(self.abundance_dict[key]/self.trans_flux*max_flux_obs/max(self.surface_flux_individual[key]))

        
        
        if save_continuum:
            self.saved_continuum=tot_flux.copy()-emission_flux.copy() 
        self.surface_flux_tot=surface_flux_tot
        self.tot_flux=tot_flux
        
        if timeit:
            time8=time()
            print('Init',time2-time1)
            print('Star',time3-time2)
            print('Inner rim',time4-time3)
            print('Midplane',time5-time4)
            print('Surface',time6-time5)
            print('Emission',time7-time6)
            print('Summing up',time8-time7)
        
        if translate_scales:
            return tot_flux,np.array(scale_paras_list)
        else:
            return tot_flux

    def run_fitted_to_obs(self,variables,dust_species,slab_dict,flux_obs,lam_obs,
                          interp=False,scipy=True,debug=False,save_continuum=continuum_penalty):
        stellar_flux, rim_flux, midplane_flux, surface_flux_dict, emission_flux_dict= self.run_model(variables=variables,dust_species=dust_species,slab_dict=slab_dict,output_all=True)
        


        #to make the algorithm find the best fit
        # I need to increase the dust component so that
        # their fluxes are of comparable strength
        # to make this most efficient I am normalizing all fluxes
        # and adjusting the scalefactors afterwards
        num_species=len(list(emission_flux_dict.keys()))
        num_dust=len(list(surface_flux_dict.keys()))    
        
        max_values=[]
        max_values.append(max(rim_flux))
        max_values.append(max(midplane_flux))
        for key in surface_flux_dict:
            max_values.append(max(surface_flux_dict[key]))
        for key in emission_flux_dict:
            max_values.append(max(emission_flux_dict[key]))
            
            
        max_values=np.array(max_values)
        
        #to not devide by 0.0 we set the max value to 1.0 if it is 0.0
        # since the whole component is 0.0 is does't matter
        for i in range(len(max_values)):
            if max_values[i]==0.0:
                max_values[i]=1.0
        if interp:
            stellar_flux=spline(self.xnew,stellar_flux,lam_obs)
            rim_flux=spline(self.xnew,rim_flux,lam_obs)
            midplane_flux=spline(self.xnew,midplane_flux,lam_obs)

        compo_ar=np.zeros((len(stellar_flux),2+num_dust+num_species))
        compo_ar[:,0]=rim_flux/max_values[0]
        compo_ar[:,1]=midplane_flux/max_values[1]
        i=2
        
        for key in surface_flux_dict:
            if interp:
                surface_flux_dict[key]=spline(self.xnew,surface_flux_dict[key],lam_obs)
            compo_ar[:,i]=surface_flux_dict[key]/max_values[i]
            i+=1
        for key in emission_flux_dict:
            if interp:
                emission_flux_dict[key]=spline(self.xnew,emission_flux_dict[key],lam_obs)
            compo_ar[:,i]=emission_flux_dict[key]/max_values[i]
            i+=1
            

        self.compo_ar=compo_ar
        
        if scipy:
  
            scaleparas=nnls(compo_ar,flux_obs-stellar_flux)[0]
            
        else:
            scaleparas=np.linalg.lstsq(compo_ar,flux_obs-stellar_flux,rcond=None)[0]
        if debug:
            print('before norm',scaleparas)
        scaleparas=scaleparas/max_values #changing the scalefactors by the same factor
        if debug:
            print('compo array min and max values')
            print(np.min(compo_ar,axis=0))
            print(np.max(compo_ar,axis=0))
            
        self.scaleparas=scaleparas
        tot_flux=stellar_flux+scaleparas[0]*rim_flux+scaleparas[1]*midplane_flux

        i=2
        for key in surface_flux_dict:
            tot_flux+=scaleparas[i]*surface_flux_dict[key]
            i+=1
        if save_continuum:
            self.saved_continuum=tot_flux.copy()            
        for key in emission_flux_dict:
            if debug:
                print(i,scaleparas[i],key)
                plt.title('Emission flux contribution')
                plt.plot(self.xnew,scaleparas[i]*self.emission_flux_individual[key],label=key,alpha=0.7)
                plt.plot(self.xnew,scaleparas[i]*emission_flux_dict[key],label=key,alpha=0.7)

            tot_flux+=scaleparas[i]*emission_flux_dict[key]
            i+=1

        if debug:
            plt.xscale('log')
            plt.legend()
            plt.xlabel(r'$\lambda [\rm \mu m]$')
            plt.show()
            
    
        return tot_flux

    def run_fitted_to_obs_slab(self,variables,slab_dict,flux_obs,lam_obs,
                          interp=False,scipy=True,debug=False,save_continuum=continuum_penalty):
        emission_flux_dict= self.run_model(variables=variables,dust_species={},slab_dict=slab_dict,output_all=True)
        

        #to make the algorithm find the best fit
        # I need to increase the dust component so that
        # their fluxes are of comparable strength
        # to make this most efficient I am normalizing all fluxes
        # and adjusting the scalefactors afterwards
        num_species=len(list(emission_flux_dict.keys()))
        
        max_values=[]
        for key in emission_flux_dict:
            max_values.append(max(emission_flux_dict[key]))
            
            
        max_values=np.array(max_values)
        
        #to not devide by 0.0 we set the max value to 1.0 if it is 0.0
        # since the whole component is 0.0 is does't matter
        for i in range(len(max_values)):
            if max_values[i]==0.0:
                max_values[i]=1.0
        compo_ar=np.zeros((len(flux_obs),num_species))
        i=0
        for key in emission_flux_dict:
            if interp:
                emission_flux_dict[key]=spline(self.xnew,emission_flux_dict[key],lam_obs)
            compo_ar[:,i]=emission_flux_dict[key]/max_values[i]
            i+=1
            

        self.compo_ar=compo_ar
        
        if scipy:
  
            scaleparas=nnls(compo_ar,flux_obs)[0]
            
        else:
            scaleparas=np.linalg.lstsq(compo_ar,flux_obs,rcond=None)[0]
        if debug:
            print('before norm',scaleparas)
        scaleparas=scaleparas/max_values #changing the scalefactors by the same factor
        if debug:
            print('compo array min and max values')
            print(np.min(compo_ar,axis=0))
            print(np.max(compo_ar,axis=0))
            
        self.scaleparas=scaleparas
        tot_flux=np.zeros_like(flux_obs)

        i=0
        for key in emission_flux_dict:
            if debug:
                print(i,scaleparas[i],key)
                plt.title('Emission flux contribution')
                plt.plot(self.xnew,scaleparas[i]*self.emission_flux_individual[key],label=key,alpha=0.7)
                plt.plot(self.xnew,scaleparas[i]*emission_flux_dict[key],label=key,alpha=0.7)

            tot_flux+=scaleparas[i]*emission_flux_dict[key]
            i+=1

        if debug:
            plt.xscale('log')
            plt.legend()
            plt.xlabel(r'$\lambda [\rm \mu m]$')
            plt.show()
            
    
        return tot_flux

    
    def get_q_translation_factors(self):
        factor_dict={}
        for key in self.abundance_dict:

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
        return factor_dict
    
    def calc_dust_masses(self,unit='msun',q_curve=True):
        '''
        This only works with the radial version
        The idea is that pi*r^2*N=Mass
        What we have is C * Q_curve.
        So, C=-2 cosi pi (R_min)^2 * N / (d^2 q Tmax^(2/q))/q_tranlation_fact

        M=pi R^2*N = - 1/2 /cosi * C *d^2 * q * T_max^(2/q) *q_translation_fact * r_in^2((tmin/tmax)^2/q-1)
        
        R^2 =r_in^2((tmin/tmax)^2/q-1)

        Unit options: msun, mjup,mearth
        '''
        if unit=='msun':
            unit_val=self.msun
        elif unit=='mjup':
            unit_val=self.mjup
        elif unit=='mearth':
            unit_val=self.mearth
            
            
        if q_curve:
            
            fact_dict=self.get_q_translation_factors()
        mass_dict={}
        for key in self.abundance_dict:
            #print(self.abundance_dict[key])
            M= -1/2.0 /degree_to_cos(self.variables['incl']) * self.abundance_dict[key] *(self.parsec*self.variables['distance'])**2 * self.variables['q_thin'] * self.variables['tmax_s']**(2/self.variables['q_thin'])*4/3
            if q_curve:
                M*=fact_dict[key]
            M*=((self.variables['tmin_s']/self.variables['tmax_s'])**(2/self.variables['q_thin'])-1)/unit_val
            mass_dict[key]=M
        return mass_dict
    def plot_radial_structure(self,low_contribution=0.15,high_contribution=0.85,ylog=True):
        '''
        This function plots the radial molecular structure of the model.
        The molecular data is loaded using exctract_emission_quantities.
        '''
                
        mol_dict=self.extract_emission_quantities(low_contribution=low_contribution,high_contribution=high_contribution,debug=False)
        
        plt.figure()
        for mol in mol_dict:
            rout,rin=mol_dict[mol]['rout,rin']
            tmin,tmax=mol_dict[mol]['tmin,tmax']
            slope=(np.log10(tmax)-np.log10(tmin))/(np.log10(rin)-np.log10(rout))
            r=10**np.linspace(np.log10(rin),np.log10(rout),1000)
            ts=tmax*(r/rin)**(slope)
            plt.plot(r,ts,label=mol)
            plt.scatter([rin,rout],[tmax,tmin])
        plt.legend()
        plt.xlabel(r'$R$ [$\rm au$]')
        plt.ylabel(r'$T$ [K]')
        plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.show()
            
    def plot(self, plot_midplane=False):
        if plot_midplane:
            plt.figure()
            plt.loglog(self.xnew,self.midplane_flux, label="midplane")
            plt.ylim(1e-5*max(self.midplane_flux),max(self.midplane_flux)*10)
            plt.title('mid-plane flux contribution')
            plt.xlabel(r'$\lambda$ [$\rm \mu m$]')
            plt.ylabel(r'$F_\nu$ [Jy]')
            plt.show()
        
        if self.run_tzones:
            plt.figure()
            for i in range(len(self.surface_flux_individual)):
                plt.loglog(self.xnew,self.surface_flux_individual[i],label=self.variables['tzones'][i])
            #plt.loglog(self.xnew,self.surface_flux_tot,label='Total flux')
            max_val=np.max(self.surface_flux_individual)
            plt.ylim(1e-5*max_val,max_val*10)
            plt.legend(loc='best')
            plt.title('disk surface layer')
            plt.xlabel('wavelength ($\mu$m')
            plt.ylabel(r'$F_\nu$ [Jy]')
            plt.show()
        
        plt.title('Emission flux contribution')
        #plot emission lines
        max_em=0
        for key in self.emission_flux_individual_scaled:
            plt.plot(self.xnew,self.emission_flux_individual_scaled[key],label=key,alpha=0.8,zorder=2)
            max_em=max(max_em,np.max(self.emission_flux_individual_scaled[key]))
        plt.ylim(top=max_em*1.1)
        plt.fill_between(x=[np.min(self.slab_wave),np.max(self.slab_wave)],y1=np.array([max_em*1.4,max_em*1.4]),y2=np.array([max_em*1.05,max_em*1.05]),alpha=0.3,label='Slab range',zorder=0)
        #plt.xscale('log')
        plt.legend()
        plt.xlabel(r'$\lambda$ [$\rm \mu$m]')
        plt.ylabel(r'$F_\nu$ [Jy]')
        plt.show()

        plt.figure() 
        plt.ylim(min(self.tot_flux),max(self.tot_flux)*1.1)
        if not self.slab_only_mode:
            plt.loglog(self.xnew,self.scaled_stellar_flux,label='star')
            plt.loglog(self.xnew,self.rim_flux,label='inner rim')
            plt.loglog(self.xnew,self.midplane_flux,label='midplane')
            plt.loglog(self.xnew,self.surface_flux_tot,label='surface')

        plt.loglog(self.xnew,self.emission_flux,label='molecular emission')
        plt.loglog(self.xnew,self.tot_flux,label='total')
        plt.legend()
        #plt.title('full model')
        plt.xlabel(r'$\lambda$ [$\rm \mu$m]')
        plt.ylabel(r'$F_\nu$ [Jy]')
        plt.show()



def create_header(var_dict,abundance_dict,slab_dict,fit_obs_err,fit_conti_err):
    header=[]
    header_para=[]
    header_abund=[]
    header_slab=[]
    header_sigma=[]
    for key in var_dict:
        
        if key!='bb_star' and key not in fixed_dict:
            header.append(key)
            header_para.append(key)

    for key in slab_dict:
        for key1 in slab_dict[key]:
            head_slab=key+':'+key1
            header.append(head_slab)
            header_slab.append(head_slab)
    for key in abundance_dict:
        header.append(key)
        header_abund.append(key)            

    if fit_obs_err:
        
        if 'log_sigma_obs' in prior_dict:
            header.append('log_sigma_obs')
            header_sigma.append('log_sigma_obs')
        else:
            header.append('sigma_obs')
            header_sigma.append('sigma_obs')
        
    if fit_conti_err:
        if 'log_sigma_conti' in prior_dict:
            header.append('log_sigma_conti')
            header_sigma.append('log_sigma_conti')

        else:
            header.append('sigma_conti')
            header_sigma.append('sigma_conti')

    header=np.array(header)
    header_para=np.array(header_para)
    header_slab=np.array(header_slab)
    header_sigma=np.array(header_sigma)

    header_abund=np.array(header_abund)
    if fit_obs_err or fit_conti_err:
        
        return header,header_para,header_abund,header_slab,header_sigma
    else:
        return header,header_para,header_abund,header_slab


def cube_to_dict(data,header,fit_obs_err=False,fit_conti_err=False,log_coldens=log_coldens):
    var_dict={}
    slab_dict={}
    sigma_dict={}
    i=0
    for key in header:
        #print('cube to dict',key)
        #print(data[i])
        if key=='sigma_obs' or key=='sigma_conti':
            sigma_dict[key]=data[i]
        elif key=='log_sigma_obs' or key=='log_sigma_conti':
            sigma_dict[key[4:]]=10**data[i]
        elif ':' in key:
            idx=key.find(':')
            if key[:idx] not in slab_dict:
                
                slab_dict[key[:idx]]={}
            if log_coldens and 'ColDens' in key[idx+1:]:
                slab_dict[key[:idx]][key[idx+1:]]=10**data[i]

            else:
                if key[idx+1:]=='logradius':
                    slab_dict[key[:idx]]['radius']=10**data[i]

                else:
                    slab_dict[key[:idx]][key[idx+1:]]=data[i]
        else:
            var_dict[key]=data[i]
        i+=1
    if fit_conti_err or fit_obs_err:
        return var_dict,slab_dict,sigma_dict
    else:
        return var_dict,slab_dict

def cube_to_dicts(data,header_para,header_abund,header_all,scale_prior,fit_obs_err=False,fit_conti_err=False,debug=False):
    var_dict={}
    slab_dict={}
    abund_dict={}
    sigma_dict={}
    i=0
    for key in header_all:
        if debug: 
            print(key)
            print(i)
            print(data[i])
        if key=='sigma_obs' or key=='sigma_conti':
            sigma_dict[key]=data[i]
        elif key=='log_sigma_obs' or key=='log_sigma_conti':
            sigma_dict[key[4:]]=10**data[i]
            
        elif ':' in key:
            if debug: print('In Slab')
            idx=key.find(':')
            if key[:idx] not in slab_dict:
                
                slab_dict[key[:idx]]={}
            if log_coldens and 'ColDens' in key[idx+1:]:
                slab_dict[key[:idx]][key[idx+1:]]=10**data[i]

            else:
                if key[idx+1:]=='logradius':
                    slab_dict[key[:idx]]['radius']=10**data[i]

                else:
                    slab_dict[key[:idx]][key[idx+1:]]=data[i]
        else:
            if key in header_para:    
                if debug: print('In Para')

                var_dict[key]=data[i]

            elif key in header_abund:
                if debug: print('In Abundance')

                if prior_on_log:
                    abund_dict[key]=10**data[i] # so the prior of the abundances should be in log10 scale
                else:
                    abund_dict[key]=data[i] #so the prior of the abundances should be in log10 scale
            elif key in scale_prior:
                if prior_on_log:
                    var_dict[key]=10**data[i]
                else:
                    var_dict[key]=data[i]
        i+=1


    if fit_conti_err or fit_obs_err:
        return var_dict,abund_dict,slab_dict,sigma_dict
    else:
        return var_dict,abund_dict,slab_dict

    
def loglike(cube,debug=False,timeit=False,return_model=False):
    sigma_dict={}
    if timeit:
        time_1=time()
    if sample_all:
        
        if fit_conti_err or fit_obs_err:
            var_dict,abundance_dict,slab_dict,sigma_dict=cube_to_dicts(cube,header_para=header_para,header_abund=header_abund,header_all=complete_header,scale_prior=scale_prior,fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)
        else:
            var_dict,abundance_dict,slab_dict=cube_to_dicts(cube,header_para=header_para,header_abund=header_abund,header_all=complete_header,scale_prior=scale_prior)

    else:    
        if fit_conti_err or fit_obs_err:
            var_dict,slab_dict,sigma_dict=cube_to_dict(cube,header=list(header_para)+list(header_slab)+list(header_sigma),fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)
        else:
            var_dict,slab_dict=cube_to_dict(cube,header=list(header_para)+list(header_slab))
  
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
            else:
                idx=key.find(':')
                if key[:idx] not in slab_dict:
                    slab_dict[key[:idx]]={}
                if debug:
                    print('..added to slab_dict')

                slab_dict[key[:idx]][key[idx+1:]]=fixed_dict[key]
  
                
    var_dict['bb_star']=use_bb_star
    
    #checking if the physics works out
    penalty=float(-10**20.0)
    if var_dict['tmin_s']>=var_dict['tmax_s']:
        return penalty
    
    if var_dict['tmin_mp']>=var_dict['tmax_mp']:
        return penalty
    
    if 't_rim' not in var_dict.keys():
        if var_dict['tmin_rim']>=var_dict['tmax_rim']:
            return penalty
    
    for key in slab_dict:
        if 'tmin' in slab_dict[key]: 
            if slab_dict[key]['tmin']>=slab_dict[key]['tmax']:
                return penalty
        if coldens_restriction:
            if 'ColDens_tmin' in slab_dict[key]: 
                if slab_dict[key]['ColDens_tmin']>slab_dict[key]['ColDens_tmax']:
                    return penalty

    
    if timeit:
        time_2=time()    
    if sample_all:
        interp_flux=con_model.run_model_normalized(variables=var_dict,dust_species=abundance_dict,
                                                slab_dict=slab_dict,max_flux_obs=max_flux_obs)


        
    else:
        interp_flux=con_model.run_fitted_to_obs(variables=var_dict,
                                                dust_species=init_abundance,
                                                slab_dict=slab_dict,
                                                flux_obs=flux_obs,lam_obs=lam_obs)

    if timeit:
        time_3=time()

    if fit_obs_err or 'sigma_obs' in sigma_dict:
        sigma=sigma_dict['sigma_obs']*flux_obs
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
    if return_model:
        return con_model
    #print(loglikelihood)
    if (not np.isfinite(loglikelihood)) or (np.isnan(loglikelihood)):
        return penalty
    else:
        return loglikelihood
    
def loglike_run(cube,ndim,nparams,debug=False,timeit=False):
    sigma_dict={}
    if timeit:
        time_1=time()
    if sample_all:
        
        if fit_conti_err or fit_obs_err:
            var_dict,abundance_dict,slab_dict,sigma_dict=cube_to_dicts(cube,header_para=header_para,header_abund=header_abund,header_all=complete_header,scale_prior=scale_prior,fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)
        else:
            var_dict,abundance_dict,slab_dict=cube_to_dicts(cube,header_para=header_para,header_abund=header_abund,header_all=complete_header,scale_prior=scale_prior)

    else:    
        if fit_conti_err or fit_obs_err:
            var_dict,slab_dict,sigma_dict=cube_to_dict(cube,header=list(header_para)+list(header_slab)+list(header_sigma),fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)
        else:
            var_dict,slab_dict=cube_to_dict(cube,header=list(header_para)+list(header_slab))
  
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
            else:
                idx=key.find(':')
                if key[:idx] not in slab_dict:
                    slab_dict[key[:idx]]={}
                if debug:
                    print('..added to slab_dict')

                slab_dict[key[:idx]][key[idx+1:]]=fixed_dict[key]
  
                
    var_dict['bb_star']=use_bb_star
    
    #checking if the physics works out
    penalty=float(-10**20.0)
    if var_dict['tmin_s']>=var_dict['tmax_s']:
        return penalty
    
    if var_dict['tmin_mp']>=var_dict['tmax_mp']:
        return penalty
    
    if 't_rim' not in var_dict.keys():
        if var_dict['tmin_rim']>=var_dict['tmax_rim']:
            return penalty
    
    for key in slab_dict:
        if 'tmin' in slab_dict[key]: 
            if slab_dict[key]['tmin']>=slab_dict[key]['tmax']:
                return penalty
        if coldens_restriction:
            if 'ColDens_tmin' in slab_dict[key]: 
                if slab_dict[key]['ColDens_tmin']>slab_dict[key]['ColDens_tmax']:
                    return penalty
    
    if timeit:
        time_2=time()    
    if sample_all:
        interp_flux=con_model.run_model_normalized(variables=var_dict,dust_species=abundance_dict,
                                                slab_dict=slab_dict,max_flux_obs=max_flux_obs)


        
    else:
        interp_flux=con_model.run_fitted_to_obs(variables=var_dict,
                                                dust_species=init_abundance,
                                                slab_dict=slab_dict,
                                                flux_obs=flux_obs,lam_obs=lam_obs)

    if timeit:
        time_3=time()

    if fit_obs_err or 'sigma_obs' in sigma_dict:
        sigma=sigma_dict['sigma_obs']*flux_obs
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
    if return_model:
        return con_model
    #print(loglikelihood)
    if (not np.isfinite(loglikelihood)) or (np.isnan(loglikelihood)):
        return penalty
    else:
        return loglikelihood
    
def prior_fast(cube):
    new_cube=(cube)*(upper_lim-lower_lim)+lower_lim
    return new_cube
def prior_run_fast(cube,ndim,nparams):
    
    cube=(cube)*(upper_lim-lower_lim)+lower_lim
    #return new_cube


# In[9]:



def return_init_dict(use_bb_star,rin_powerlaw):
    var_dict={'tmin_s':None,
             'tmax_s':None,
             'tmin_mp':None,
             'tmax_mp':None,
             #'sc_ir':None,
             #'sc_mid':None,
             'q_mid':None,
             'q_thin':None,
             'q_emis':None}
    if 'incl' in prior_dict or 'incl' in fixed_dict:
        var_dict['incl']=None
    if use_bb_star:
        var_dict['distance']=None
        var_dict['tstar']=None
        var_dict['rstar']=None
        var_dict['bb_star']=True
    else:
        var_dict['bb_star']=False
    if rin_powerlaw:
        var_dict['tmax_rim']=None
        var_dict['tmin_rim']=None
        var_dict['q_rim']=None
    else:
        var_dict['t_rim']=None
    return var_dict


# # here starts the part where you have to adjust things
# 

# put in the observation that you want to fit
# - lam_obs: array with the wavelength points in micron
# - flux_obs: array with the corresponding fluxes in Jy
# - sig_obs: array with the corresponding uncertainties of the observations

# In[23]:


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
    #print(prior_dict_dust)


# setting up the dictonaries and headers that will be used

# In[19]:
init_dict=return_init_dict(use_bb_star=use_bb_star,rin_powerlaw=rin_powerlaw)


if 'log_sigma_obs' in prior_dict:
    fit_obs_err=True
else:
    fit_obs_err=False
if 'log_sigma_conti' in prior_dict:
    fit_conti_err=True
else:
    fit_conti_err=False
if fit_conti_err or fit_obs_err:
    header,header_para,header_abund,header_slab,header_sigma=create_header(var_dict=init_dict,
                                                              abundance_dict=init_abundance,
                                                              slab_dict=slab_prior_dict,
                                                              fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)

else:
    header,header_para,header_abund,header_slab=create_header(var_dict=init_dict,
                                                              abundance_dict=init_abundance,
                                                              slab_dict=slab_prior_dict,
                                                              fit_conti_err=fit_conti_err,fit_obs_err=fit_obs_err)
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

# initializing the model and reading in the data

# In[20]:


con_model=complete_model()


con_model.read_data(variables=init_dict,dust_species=init_abundance,
                    slab_dict=slab_prior_dict,slab_prefix=slab_prefix,
                    stellar_file=stellar_file,wavelength_points=lam_obs)

if __name__ == "__main__":
    print(con_model)


# # Let's run

# In[25]:

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

if debug:
    print('N dims',len(upper_lim))
    test_vals=[]
    for i in range(len(upper_lim)):
        test_vals.append((upper_lim[i]+lower_lim[i])/2)
    print(len(test_vals))
    print(test_vals)
    loglike(test_vals,debug=True,timeit=False)
          
if __name__ == "__main__":
    if not os.path.isfile(f'{prefix}start.time'):
        os.system(f'date > {prefix}start.time')
    
    if running:
        result = run(LogLikelihood=loglike_run, Prior=prior_run_fast, 
                   n_dims=len(upper_lim), outputfiles_basename=prefix, verbose=True,n_live_points = n_live_points,evidence_tolerance = evidence_tolerance ,sampling_efficiency = sampling_efficiency,importance_nested_sampling=False)
    else:

        result = solve(LogLikelihood=loglike, Prior=prior_fast, 
                       n_dims=len(upper_lim), outputfiles_basename=prefix, verbose=True,n_live_points = n_live_points,evidence_tolerance = evidence_tolerance ,sampling_efficiency = sampling_efficiency, importance_nested_sampling=False)
    if not os.path.isfile(f'{prefix}end.time'):
        os.system(f'date > {prefix}end.time')


