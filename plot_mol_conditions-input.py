import glob
import numpy as np
import time
import os
from scipy import interpolate
from scipy.optimize import nnls
import json
import uuid
import multiprocessing as mp

from matplotlib.lines import Line2D
from PyAstronomy import pyasl
import corner
from ast import literal_eval

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.legend_handler import HandlerTuple

import sys
import importlib
from time import sleep

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



# print license
print('------------------------------------------------')
print('------------------------------------------------')
print('   DuCKLinG  Copyright (C) 2025 Till Kaeufer    ')  
print('------------------------------------------------')
print('This program comes with ABSOLUTELY NO WARRANTY  ')
print('This is free software, and you are welcome to   ')
print('redistribute it under certain conditions        ')
print('detailed in the LICENSE file.                   ')
print('------------------------------------------------')
print('------------------------------------------------')


print('Load inputs...')

no_sigma=False
close_plots=True
debug=False
temp_range=[25,1500]
coldens_range=[14,24]
radial_range=[None,None]
nbins=100
log_t_first=False
log_t_second=True
log_r=True
npoints_per_model=1000
reduce_posterior=False

# %%
if __name__ == "__main__":
    input_file=sys.argv[1]



    if len(sys.argv)>2:
       
        arg_list=sys.argv[1:]
        
        for i in range(len(arg_list)):
            argument=arg_list[i]
            if argument=='log_t:True':
                log_t_first=True
                log_t_second=True            
            elif argument=='log_t:False':
                log_t_first=False
                log_t_second=False
            elif argument=='log_t_first:True':
                log_t_first=True            
            elif argument=='log_t_second:False':
                log_t_second=False
            elif argument=='nbins':
                nbins=int(arg_list[i+1])
            elif argument=='temp_range':
                temp_range=np.array(literal_eval(arg_list[int(i+1)]),dtype='float64')
            elif argument=='radial_range':
                radial_range=np.array(literal_eval(arg_list[int(i+1)]),dtype='float64')
                
            elif argument=='coldens_range':
                coldens_range=np.array(literal_eval(arg_list[int(i+1)]),dtype='float64')
            elif argument=='npoints':
                npoints_per_model=int(arg_list[i+1])  
            elif argument=='close':
                close_plots=True
            elif argument=='reduce_post':
                reduce_posterior=True
            elif argument=='no_sigma':
                no_sigma=True

                


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


print('Temp range: ',temp_range)
print('ColDens range: ',coldens_range)
print('Radial range: ', radial_range)
print('Nbins: ', nbins)
print('Npoints per model: ',npoints_per_model)
print('Log T first plot: ',log_t_first)
print('Log T second plot: ',log_t_second)


print('..Done!')
print('----------------------------')

try:
    lam_obs_full
    print('Masking used')
except NameError:
    print('There is no masking used')
    lam_obs_full=lam_obs

if reduce_posterior:
    reduce_str='_reduced'
else:
    reduce_str=''
#sleep(5)


fold_string=bayesian_folder
if subfold!='':
    fold_string=fold_string+subfold
prefix = fold_string+'test_'+str(run_number)

print('Loading posterior...')

folder=bayesian_folder+subfold
prefix_fig=folder+f'/figures/{run_number}'
list_complete_post=glob.glob(folder+f'*_{run_number}complete_posterior{reduce_str}.npy')

list_complete_post.sort()

header=np.load(folder+f'test_{run_number}header_complete_posterior.npy')


paras=np.load(list_complete_post[0])

print('Shape header:',np.shape(header))
print('Shape parameters:',np.shape(paras))
print('...Done!')



print('Plotting all median, std and minus std')
posterior_dict={}
for i in range(len(header)):
    print(header[i])
    samp=paras[:,i]
    median=np.percentile(samp,50)
    std_plus=np.percentile(samp,50+68/2)
    std_minus=np.percentile(samp,50-68/2)
    posterior_dict[header[i]]=[median,std_plus,std_minus]
    print(median,std_plus,std_minus)


print('Creating unique colour map for all molecules..')

dict_cmaps={}
for mol in mol_colors_dict:
    
    color=mpl.colors.to_rgba(mol_colors_dict[mol])
    n_colors=100
    list_colors=[]
    for i in range(n_colors):
        adjust_color=list(color)
        adjust_color[-1]=float(i/n_colors)
        list_colors.append(tuple(adjust_color))
    dict_cmaps[mol]=mpl.colors.ListedColormap(list_colors,'Colormap')

print('...Done!')

print('Creating bins and functions for first plot...')

xbins=np.linspace(temp_range[0],temp_range[1],nbins)
ybins=np.linspace(coldens_range[0],coldens_range[1],nbins)
if log_t_first:
    
    xbins=np.linspace(np.log10(temp_range[0]),np.log10(temp_range[1]),nbins)
    ybins=ybins=np.linspace(coldens_range[0],coldens_range[1],nbins)


def emission_points_from_boundries(t_in,t_out,coldens_in,coldens_out,debug=False):
    slope=(np.log10(coldens_out)-np.log10(coldens_in))/(np.log10(t_out)-np.log10(t_in))
    if debug:
        print(np.median(slope))
        
    t_points=np.linspace(t_in,t_out,npoints_per_model)
    #t_points=np.logspace(np.log10(t_in),np.log10(t_out),1000)
    #n_points=np.zeros_like(t_points)
    if debug:
        print(np.shape(t_points))
    #for i in range(len(t_points)):
    #    n_points[i]=coldens_in*(t_points[i]/t_in)**(slope)
    n_points=coldens_in*(t_points/t_in)**(slope)
    
    if debug:
        print('t range',np.min(t_points),np.max(t_points))
    
        print('N range',np.min(n_points),np.max(n_points))
    
    return t_points.flatten(),n_points.flatten()


def xticks_on_log(xbins_linear,xbins,bin_label=[30,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500],debug=False):
    label_pos=[]
    for val in bin_label:
        if debug:
            print('Value')
            print(val)
        for i,val_xbins in enumerate(xbins):
            #print(i)
            if val_xbins>=val:
                if debug:
                    print('larger')
                    print(val_xbins)
                fact=(val-xbins[i])/(xbins[i+1]-xbins[i])
                break
        label_pos.append(xbins_linear[i]+(fact)*(xbins_linear[i+1]-xbins_linear[i]))              
    return label_pos
print('...Done!')

print('Plotting column density and temperature for all molecules')

for full_range in [False,True]:
    mol_list=list(slab_prior_dict.keys())
    zorder=np.arange(len(mol_list))

    custom_lines=[]
    custom_labels=[]
    plt.figure(figsize=(9,6))
    i=0
    for mol in mol_list:
        print(mol)
        powerlaw=True 
        if f'{mol}:temis' in header or f'{mol}:temis' in fixed_dict:
            powerlaw=False
            if f'{mol}:temis' in header:

                idx_t=np.where(header==f'{mol}:temis')[0][0]
                t_points= paras[:,idx_t]
            else:

                t_points= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:temis']                
            if f'{mol}:ColDens' not in fixed_dict:
                idx_coldens=np.where(header==f'{mol}:ColDens')[0][0]

                n_points= paras[:,idx_coldens]
            else:
                n_points= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens']



        else:
            if full_range:
                if 'ColDens' in slab_prior_dict[mol]:

                    idx_coldens_out=np.where(header==f'{mol}:ColDens')[0][0]
                
                    coldens_out= paras[:,idx_coldens_out]
                    coldens_in= coldens_out
                if f'{mol}:ColDens' in fixed_dict:

                    coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens']
                    coldens_out=coldens_in

                if 'ColDens_tmax' in slab_prior_dict[mol]:
                    idx_coldens_in=np.where(header==f'{mol}:ColDens_tmax')[0][0]
                    coldens_in= paras[:,idx_coldens_in]
                if f'{mol}:ColDens_tmax' in fixed_dict:
                    coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmax']

                if 'ColDens_tmin' in slab_prior_dict[mol]:
                    idx_coldens_out=np.where(header==f'{mol}:ColDens_tmin')[0][0]
                    coldens_out= paras[:,idx_coldens_out]
                if f'{mol}:ColDens_tmin' in fixed_dict:
                    coldens_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmin']

                if 'tmin' in slab_prior_dict[mol]:
                    idx_t_out=np.where(header==f'{mol}:tmin')[0][0]
                    t_out= paras[:,idx_t_out]
                if f'{mol}:tmin' in fixed_dict:
                    t_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmin'] 

                if 'tmax' in slab_prior_dict[mol]:
                    idx_t_in=np.where(header==f'{mol}:tmax')[0][0]
                    t_in= paras[:,idx_t_in]
                if f'{mol}:tmax' in fixed_dict:
                    t_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmax'] 

            else:
                idx_coldens_out=np.where(header==f'{mol}:\nlogDensCol at 0.15')[0][0]
                idx_coldens_in=np.where(header==f'{mol}:\nlogDensCol at 0.85')[0][0]

                idx_t_out=np.where(header==f'{mol}:\nt at 0.15')[0][0]
                idx_t_in=np.where(header==f'{mol}:\nt at 0.85')[0][0]

                coldens_out= paras[:,idx_coldens_out]
                coldens_in= paras[:,idx_coldens_in]
                t_in= paras[:,idx_t_in]
                t_out= paras[:,idx_t_out]
 
                print(idx_t_in,idx_t_out)
                print(np.shape(paras),np.shape(header))

        
        print('Used a powerlaw?',powerlaw)
        if powerlaw:

            t_points,n_points=emission_points_from_boundries(t_in,t_out,coldens_in,coldens_out)


        if log_t_first:
            t_points=np.log10(t_points)
        heatmap, xedges, yedges = np.histogram2d(t_points, n_points, bins=[xbins,ybins])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        mol_color=mol
        if '_comp' in mol_color:
            idx_mol=mol_color.find('_comp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {mol} to {mol_color}')
        if '_absorp' in mol_color:
            idx_mol=mol_color.find('_absorp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {mol} to {mol_color}')
        plt.imshow(heatmap.T, extent=extent,origin='lower',aspect='auto',cmap=dict_cmaps[mol_color],zorder=zorder[i])
        
        custom_lines.append(Line2D([0], [0], color=mol_colors_dict[mol_color], lw=4))
        custom_labels.append(molecular_names[mol_color])
        i+=1

    plt.ylabel('$\log_{10} \Sigma$ [cm$^{-2}$]')
    if no_sigma:
        plt.ylabel('$\log_{10} N$ [cm$^{-2}$]')
    
    plt.xlabel('$T$ [K]')    
    if log_t_first:

        ticks_labels=[30,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
        ticks_labels_print=[30,100,200,300,'',500,'',700,'','',1000,'','','','',1500]

        plt.xticks(np.log10(ticks_labels),ticks_labels_print)
        plt.legend(custom_lines,custom_labels,loc=(-0.07,1.05),ncol=max(len(custom_lines)//2-2,1),handler_map={tuple: HandlerTuple(ndivide=None)})
        
        plt.xlim([np.log10(temp_range[0]),np.log10(temp_range[1])]) 
    else:
        plt.legend(custom_lines,custom_labels,loc=(-0.07,1.05),ncol=max(len(custom_lines)//2-2,1),handler_map={tuple: HandlerTuple(ndivide=None)})
        plt.xlim([temp_range[0],temp_range[1]]) 
    if full_range:
        plt.savefig(prefix_fig+'_molecular_conditions_full_range.pdf',bbox_inches='tight')
    else:
        plt.savefig(prefix_fig+'_molecular_conditions.pdf',bbox_inches='tight')

    if close_plots:
        plt.close()
    else:
        plt.show()


print('Plotting the full range with an emission contour') 

mol_list=list(slab_prior_dict.keys())
zorder=np.arange(len(mol_list))

plt.figure(figsize=(9,6))

for full_range in [False,True]:
    i=0

    custom_lines=[]
    custom_labels=[]
    for mol in mol_list:
        print(mol)
        powerlaw=True 
        if f'{mol}:temis' in header or f'{mol}:temis' in fixed_dict:
            powerlaw=False
            if f'{mol}:temis' in header:

                idx_t=np.where(header==f'{mol}:temis')[0][0]
                t_points= paras[:,idx_t]
            else:

                t_points= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:temis']                
            if f'{mol}:ColDens' not in fixed_dict:
                idx_coldens=np.where(header==f'{mol}:ColDens')[0][0]

                n_points= paras[:,idx_coldens]
            else:
                n_points= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens']



        else:
            if full_range:
                if 'ColDens' in slab_prior_dict[mol]:

                    idx_coldens_out=np.where(header==f'{mol}:ColDens')[0][0]
                
                    coldens_out= paras[:,idx_coldens_out]
                    coldens_in= coldens_out
                if f'{mol}:ColDens' in fixed_dict:

                    coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens']
                    coldens_out=coldens_in

                if 'ColDens_tmax' in slab_prior_dict[mol]:
                    idx_coldens_in=np.where(header==f'{mol}:ColDens_tmax')[0][0]
                    coldens_in= paras[:,idx_coldens_in]
                if f'{mol}:ColDens_tmax' in fixed_dict:
                    coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmax']

                if 'ColDens_tmin' in slab_prior_dict[mol]:
                    idx_coldens_out=np.where(header==f'{mol}:ColDens_tmin')[0][0]
                    coldens_out= paras[:,idx_coldens_out]
                if f'{mol}:ColDens_tmin' in fixed_dict:
                    coldens_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmin']

                if 'tmin' in slab_prior_dict[mol]:
                    idx_t_out=np.where(header==f'{mol}:tmin')[0][0]
                    t_out= paras[:,idx_t_out]
                if f'{mol}:tmin' in fixed_dict:
                    t_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmin'] 

                if 'tmax' in slab_prior_dict[mol]:
                    idx_t_in=np.where(header==f'{mol}:tmax')[0][0]
                    t_in= paras[:,idx_t_in]
                if f'{mol}:tmax' in fixed_dict:
                    t_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmax'] 

            else:
                idx_coldens_out=np.where(header==f'{mol}:\nlogDensCol at 0.15')[0][0]
                idx_coldens_in=np.where(header==f'{mol}:\nlogDensCol at 0.85')[0][0]

                idx_t_out=np.where(header==f'{mol}:\nt at 0.15')[0][0]
                idx_t_in=np.where(header==f'{mol}:\nt at 0.85')[0][0]

                coldens_out= paras[:,idx_coldens_out]
                coldens_in= paras[:,idx_coldens_in]
                t_in= paras[:,idx_t_in]
                t_out= paras[:,idx_t_out]
 
                print(idx_t_in,idx_t_out)
                print(np.shape(paras),np.shape(header))

        
        print('Used a powerlaw?',powerlaw)
        if powerlaw:

            t_points,n_points=emission_points_from_boundries(t_in,t_out,coldens_in,coldens_out)
        else:
            n_points= paras[:,idx_coldens]

            t_points= paras[:,idx_t]
        if log_t_first:
            t_points=np.log10(t_points)
        heatmap, xedges, yedges = np.histogram2d(t_points, n_points, bins=[xbins,ybins])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        mol_color=mol
        if '_comp' in mol_color:
            idx_mol=mol_color.find('_comp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {mol} to {mol_color}')
        if '_absorp' in mol_color:
            idx_mol=mol_color.find('_absorp')
            
            mol_color=mol_color[:idx_mol]
            print(f'Changing {mol} to {mol_color}')
        if full_range:
            plt.imshow(heatmap.T, extent=extent,origin='lower',aspect='auto',cmap=dict_cmaps[mol_color],zorder=zorder[i])
        
            custom_lines.append(Line2D([0], [0], color=mol_colors_dict[mol_color], lw=4))
            custom_labels.append(molecular_names[mol_color])
        if not full_range:

            xpoints=(xbins[:-1]+xbins[1:])/2.0
            ypoints=(ybins[:-1]+ybins[1:])/2.0

            yall,xall=np.meshgrid(ypoints,xpoints)
            xall=xall.flatten()
            yall=yall.flatten()
            zall=heatmap.flatten()
            plt.tricontour(xall,yall,zall,levels=[np.max(zall)/10.0],colors='grey',zorder=10000)

        i+=1

plt.ylabel('$\log_{10} \Sigma$ [cm$^{-2}$]')

if no_sigma:
    plt.ylabel('$\log_{10} N$ [cm$^{-2}$]')
plt.xlabel('$T$ [K]')  
custom_lines.append(Line2D([0], [0], color='grey', lw=2))
custom_labels.append(r'$70\,\mathrm{\%}$ Emission')  
if log_t_first:

    ticks_labels=[30,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
    ticks_labels_print=[30,100,200,300,'',500,'',700,'','',1000,'','','','',1500]

    plt.xticks(np.log10(ticks_labels),ticks_labels_print)
    plt.legend(custom_lines,custom_labels,loc=(-0.07,1.05),ncol=max(len(custom_lines)//2-2,1),handler_map={tuple: HandlerTuple(ndivide=None)})
    
    plt.xlim([np.log10(temp_range[0]),np.log10(temp_range[1])]) 
else:
    plt.legend(custom_lines,custom_labels,loc=(-0.07,1.05),ncol=max(len(custom_lines)//2-2,1),handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.xlim([temp_range[0],temp_range[1]]) 
plt.savefig(prefix_fig+'_molecular_conditions_contour_version.pdf',bbox_inches='tight')

if close_plots:
    plt.close()
else:
    plt.show()

print('...Done!')



def radius_temperature_relation(t_in,t_out,r_in,r_out,debug=False,log_r=False):
    slope=(np.log10(t_out)-np.log10(t_in))/(np.log10(r_out)-np.log10(r_in))
    if debug:
        print('Median slope',np.median(slope))
    r_points=np.linspace(r_in,r_out,npoints_per_model)
    if log_r: 
        r_points=10**np.linspace(np.log10(r_in),np.log10(r_out),npoints_per_model)
        
    t_points=np.zeros_like(r_points)
    if debug:
        print(np.shape(t_points))
    for i in range(len(r_points)):
        t_points[i]=t_in*(r_points[i]/r_in)**(slope)
    if debug:
        print('t range',np.min(t_points),np.max(t_points))
    
        print('r range',np.min(r_points),np.max(r_points))

    return r_points.flatten(),t_points.flatten()

print('Loading functions for second plot...')
tot_r=[]
tot_t=[]
tot_c=[]
for full_range in [True,False]:


    for mol in mol_list:
        print(mol)
        powerlaw=True 
        if f'{mol}:temis' in header or f'{mol}:temis' in fixed_dict:
            powerlaw=False
            if f'{mol}:temis' in header:

                idx_t=np.where(header==f'{mol}:temis')[0][0]
                t_points= paras[:,idx_t]
                powerlaw=False
            else:

                t_points= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:temis']                
            if f'{mol}:ColDens' not in fixed_dict:
                idx_coldens=np.where(header==f'{mol}:ColDens')[0][0]

                n_points= paras[:,idx_coldens]
            else:
                n_points= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens']


        if full_range:
            if 'ColDens' in slab_prior_dict[mol]:

                idx_coldens_out=np.where(header==f'{mol}:ColDens')[0][0]
            
                coldens_out= paras[:,idx_coldens_out]
                coldens_in= coldens_out
            if f'{mol}:ColDens' in fixed_dict:

                coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens']
                coldens_out=coldens_in

            if 'ColDens_tmax' in slab_prior_dict[mol]:
                idx_coldens_in=np.where(header==f'{mol}:ColDens_tmax')[0][0]
                coldens_in= paras[:,idx_coldens_in]
            if f'{mol}:ColDens_tmax' in fixed_dict:
                coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmax']

            if 'ColDens_tmin' in slab_prior_dict[mol]:
                idx_coldens_out=np.where(header==f'{mol}:ColDens_tmin')[0][0]
                coldens_out= paras[:,idx_coldens_out]
            if f'{mol}:ColDens_tmin' in fixed_dict:
                coldens_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmin']

            if 'tmin' in slab_prior_dict[mol]:
                idx_t_out=np.where(header==f'{mol}:tmin')[0][0]
                t_out= paras[:,idx_t_out]
            if f'{mol}:tmin' in fixed_dict:
                t_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmin'] 

            if 'tmax' in slab_prior_dict[mol]:
                idx_t_in=np.where(header==f'{mol}:tmax')[0][0]
                t_in= paras[:,idx_t_in]
            if f'{mol}:tmax' in fixed_dict:
                t_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmax'] 

            if f'{mol}: logradius' in header:
                
                idx_rin=np.where(header==f'{mol}: logradius')[0][0]


                r_in= 10**paras[:,idx_rin]
            if f'{mol}:logradius' in fixed_dict:

                r_in= np.ones(len(paras[:,0]))*fixed_dict[f'{mol}:logradius']
                    
            if f'{mol}: radius' in header:
                idx_rin=np.where(header==f'{mol}: radius')[0][0]
            
                r_in= paras[:,idx_rin]
            if f'{mol}:radius' in fixed_dict:

                r_in= np.ones(len(paras[:,0]))*fixed_dict[f'{mol}:radius']

            if 'q_emis' in prior_dict:

                idx_qemis=np.where(header==f'q_emis')[0][0]
                qemis=paras[:,idx_qemis]


            r_out= r_in*(t_out/t_in)**(1/qemis)
        else:
            idx_coldens_out=np.where(header==f'{mol}:\nlogDensCol at 0.15')[0][0]
            idx_coldens_in=np.where(header==f'{mol}:\nlogDensCol at 0.85')[0][0]
            idx_rin=np.where(header==f'{mol}:\nrin')[0][0]
            idx_rout=np.where(header==f'{mol}:\nrout')[0][0]

            idx_t_out=np.where(header==f'{mol}:\nt at 0.15')[0][0]
            idx_t_in=np.where(header==f'{mol}:\nt at 0.85')[0][0]
            coldens_out= paras[:,idx_coldens_out]
            coldens_in= paras[:,idx_coldens_in]
            r_out= paras[:,idx_rout]
            r_in= paras[:,idx_rin]
            t_out= paras[:,idx_t_out]
            t_in= paras[:,idx_t_in]  

        print('Powerlaw:',powerlaw)
        if powerlaw:
            print('Filtering out 0.0 radii')
            print('From', len(r_in))
            idx_ok=np.where(r_in!=0.0)[0]

            coldens_out=coldens_out[idx_ok]
            coldens_in=coldens_in[idx_ok]
            r_out=r_out[idx_ok]
            r_in=r_in[idx_ok]            
            t_out=t_out[idx_ok]
            t_in=t_in[idx_ok]
            print('To',len(r_in))
            tot_r.append(r_in)
            tot_r.append(r_out)
            tot_t.append(t_in)
            tot_t.append(t_out)
            tot_c.append(coldens_out)
            tot_c.append(coldens_in)

if len(tot_r)==0:
    print('No molecule emits over a temperature powerlaw.')
    print('Therefore, we cannot print the radial structure')
else:
    flat_r=[]
    for entry in tot_r:
        for val in entry:
            flat_r.append(val)
            
    flat_t=[]
    for entry in tot_t:
        for val in entry:
            flat_t.append(val)
    flat_c=[]
    for entry in tot_c:
        for val in entry:
            flat_c.append(val)
            
    flat_r=np.array(flat_r)
    flat_t=np.array(flat_t)
    flat_c=np.array(flat_c)
    
    print('Minimum temperature')
    print(np.min(flat_t))
    print('Maximum temperature')
    print(np.max(flat_t))

    print('Minimum log coldens')
    print(np.log10(np.min(flat_c)))
    print('Maximum log coldens')
    print(np.log10(np.max(flat_c)))
    print('Minimum radius (lin/log)')
    print(np.min(flat_r),np.log10(np.min(flat_r)))
    print('Maximum radius (lin/log)')
    print(np.max(flat_r),np.log10(np.max(flat_r)))
    
    if radial_range[0]==None:
        if log_r:
            radial_range[0]=np.log10(np.min(flat_r))
        else:
            radial_range[0]=np.min(flat_r)
            
    if radial_range[1]==None:
        if log_r:
            radial_range[1]=np.log10(np.max(flat_r))
        else:
            radial_range[1]=np.max(flat_r)
    if log_r:
        xbins_linear=np.linspace(10**float(radial_range[0]),10**float(radial_range[1]),nbins)
    else:
        xbins_linear=np.linspace(radial_range[0],radial_range[1],nbins)
    rad_range_x=np.ptp(radial_range)
        
    xbins=np.linspace(radial_range[0]-0.05*rad_range_x,radial_range[1]+0.05*rad_range_x,nbins)
    
    if log_t_second:
        ybins=np.linspace(np.log10(temp_range[0]),np.log10(temp_range[1]),nbins)
    else:
        ybins=np.linspace(temp_range[0],temp_range[1],nbins)
    

    print('...Done!')
    print('Plotting radial temperature distribution..')
    for full_range in [True,False]:
            heatmap_dict={}
            mol_list=list(slab_prior_dict.keys())
            zorder=np.arange(len(mol_list))
            
            custom_lines=[]
            custom_labels=[]
            plt.figure(figsize=(9,6))
            i=0
            for mol in mol_list:
                print(mol)
                powerlaw=True 
                if f'{mol}:temis' in header or f'{mol}:temis' in fixed_dict:
                    powerlaw=False


                if full_range:

                    if 'tmin' in slab_prior_dict[mol]:
                        idx_t_out=np.where(header==f'{mol}:tmin')[0][0]
                        t_out= paras[:,idx_t_out]
                    if f'{mol}:tmin' in fixed_dict:
                        t_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmin'] 

                    if 'tmax' in slab_prior_dict[mol]:
                        idx_t_in=np.where(header==f'{mol}:tmax')[0][0]
                        t_in= paras[:,idx_t_in]
                    if f'{mol}:tmax' in fixed_dict:
                        t_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmax'] 

                    if f'{mol}: logradius' in header:
                        
                        idx_rin=np.where(header==f'{mol}: logradius')[0][0]


                        r_in= 10**paras[:,idx_rin]
                            
                    if f'{mol}: radius' in header:
                        idx_rin=np.where(header==f'{mol}: radius')[0][0]
                    
                        r_in= paras[:,idx_rin]
                    if f'{mol}:radius' in fixed_dict:
                        r_in=np.ones_like(paras[:,0])*fixed_dict[f'{mol}:radius']
                    if f'{mol}:logradius' in fixed_dict:
                        r_in=np.ones_like(paras[:,0])*10**fixed_dict[f'{mol}:logradius']

                    if 'q_emis' in prior_dict:

                        idx_qemis=np.where(header==f'q_emis')[0][0]
                        qemis=paras[:,idx_qemis]


                    r_out= r_in*(t_out/t_in)**(1/qemis)
                else:
                    idx_rout=np.where(header==f'{mol}:\nrout')[0][0]
                    idx_rin=np.where(header==f'{mol}:\nrin')[0][0]
            
                    idx_t_out=np.where(header==f'{mol}:\nt at 0.15')[0][0]
                    idx_t_in=np.where(header==f'{mol}:\nt at 0.85')[0][0]

                    r_in=paras[:,idx_rin]
                    r_out=paras[:,idx_rout]
                    t_in=paras[:,idx_t_in]
                    t_out=paras[:,idx_t_out]
                if powerlaw:
                    if debug:
                        print('Median t_in, t_out')
                        print(np.median(t_in),np.median(t_out))
                        print('Median r_in, r_out')
                        print(np.median(r_in),np.median(r_out))

                    
                    idx_ok=np.where(r_in!=0.0)[0]
                    r_out=r_out[idx_ok]
                    r_in=r_in[idx_ok]            
                    t_out=t_out[idx_ok]
                    t_in=t_in[idx_ok]
                    r_points,t_points=radius_temperature_relation(t_in=t_in,t_out=t_out,r_in=r_in,r_out=r_out,log_r=log_r)

                    if log_t_second:
                        t_points=np.log10(t_points)
                    if log_r:
                        r_points=np.log10(r_points)
                    heatmap, xedges, yedges = np.histogram2d(r_points, t_points, bins=[xbins,ybins])
                    heatmap_dict[mol]=heatmap
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    mol_color=mol
                    if '_comp' in mol_color:
                        idx_mol=mol_color.find('_comp')
                        
                        mol_color=mol_color[:idx_mol]
                        print(f'Changing {mol} to {mol_color}')
                    if '_absorp' in mol_color:
                        idx_mol=mol_color.find('_absorp')
                        
                        mol_color=mol_color[:idx_mol]
                        print(f'Changing {mol} to {mol_color}')
                    plt.imshow(heatmap.T, extent=extent,origin='lower',aspect='auto',cmap=dict_cmaps[mol_color],zorder=zorder[i])
                        
                    custom_lines.append(Line2D([0], [0], color=mol_colors_dict[mol_color], lw=4))
                    custom_labels.append(molecular_names[mol_color])
                    i+=1
            
            plt.xlabel('$\log_{10} R$ [au]')
            plt.ylabel('$T$ [K]')    
            if log_t_second:
            
            
                ticks_labels=[30,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
                ticks_labels_print=[30,100,200,300,'',500,'',700,'','',1000,'','','','',1500]
            
                plt.yticks(np.log10(ticks_labels),ticks_labels_print)
                
                plt.ylim([np.log10(temp_range[0]),np.log10(temp_range[1])]) 
            else:
                plt.ylim([temp_range[0],temp_range[1]]) 
            plt.legend(custom_lines,custom_labels,loc=(-0.1,1.05),ncol=max(len(custom_lines)//2,1))
            if full_range:
                plt.savefig(prefix_fig+'_molecular_conditions_temp_by_radius_full_range.pdf',bbox_inches='tight')
            else:
                plt.savefig(prefix_fig+'_molecular_conditions_temp_by_radius.pdf',bbox_inches='tight')
            if close_plots:
                plt.close()
            else:
                plt.show()
    print('Contour version')

    plt.figure(figsize=(9,6))
    for full_range in [False,True]:
        heatmap_dict={}
        mol_list=list(slab_prior_dict.keys())
        zorder=np.arange(len(mol_list))
        
        custom_lines=[]
        custom_labels=[]
        i=0
        for mol in mol_list:
            print(mol)
            powerlaw=True 
            if f'{mol}:temis' in header or f'{mol}:temis' in fixed_dict:
                powerlaw=False


            if full_range:

                if 'tmin' in slab_prior_dict[mol]:
                    idx_t_out=np.where(header==f'{mol}:tmin')[0][0]
                    t_out= paras[:,idx_t_out]
                if f'{mol}:tmin' in fixed_dict:
                    t_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmin'] 

                if 'tmax' in slab_prior_dict[mol]:
                    idx_t_in=np.where(header==f'{mol}:tmax')[0][0]
                    t_in= paras[:,idx_t_in]
                if f'{mol}:tmax' in fixed_dict:
                    t_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:tmax'] 

                if f'{mol}: logradius' in header:
                    
                    idx_rin=np.where(header==f'{mol}: logradius')[0][0]


                    r_in= 10**paras[:,idx_rin]
                        
                if f'{mol}: radius' in header:
                    idx_rin=np.where(header==f'{mol}: radius')[0][0]
                
                    r_in= paras[:,idx_rin]
                if f'{mol}:radius' in fixed_dict:
                    r_in=np.ones_like(paras[:,0])*fixed_dict[f'{mol}:radius']
                if f'{mol}:logradius' in fixed_dict:
                    r_in=np.ones_like(paras[:,0])*10**fixed_dict[f'{mol}:logradius']

                if 'q_emis' in prior_dict:

                    idx_qemis=np.where(header==f'q_emis')[0][0]
                    qemis=paras[:,idx_qemis]


                r_out= r_in*(t_out/t_in)**(1/qemis)
            else:
                idx_rout=np.where(header==f'{mol}:\nrout')[0][0]
                idx_rin=np.where(header==f'{mol}:\nrin')[0][0]
        
                idx_t_out=np.where(header==f'{mol}:\nt at 0.15')[0][0]
                idx_t_in=np.where(header==f'{mol}:\nt at 0.85')[0][0]

                r_in=paras[:,idx_rin]
                r_out=paras[:,idx_rout]
                t_in=paras[:,idx_t_in]
                t_out=paras[:,idx_t_out]
            if powerlaw:
                if debug:
                    print('Median t_in, t_out')
                    print(np.median(t_in),np.median(t_out))
                    print('Median r_in, r_out')
                    print(np.median(r_in),np.median(r_out))

                
                idx_ok=np.where(r_in!=0.0)[0]
                r_out=r_out[idx_ok]
                r_in=r_in[idx_ok]            
                t_out=t_out[idx_ok]
                t_in=t_in[idx_ok]
                r_points,t_points=radius_temperature_relation(t_in=t_in,t_out=t_out,r_in=r_in,r_out=r_out,log_r=log_r)
        
                if log_t_second:
                    t_points=np.log10(t_points)
                if log_r:
                    r_points=np.log10(r_points)
                heatmap, xedges, yedges = np.histogram2d(r_points, t_points, bins=[xbins,ybins])
                heatmap_dict[mol]=heatmap
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                mol_color=mol
                if '_comp' in mol_color:
                    idx_mol=mol_color.find('_comp')
                    
                    mol_color=mol_color[:idx_mol]
                    print(f'Changing {mol} to {mol_color}')
                if '_absorp' in mol_color:
                    idx_mol=mol_color.find('_absorp')
                    
                    mol_color=mol_color[:idx_mol]
                    print(f'Changing {mol} to {mol_color}')
                if full_range:
                    plt.imshow(heatmap.T, extent=extent,origin='lower',aspect='auto',cmap=dict_cmaps[mol_color],zorder=zorder[i])
                        
                    custom_lines.append(Line2D([0], [0], color=mol_colors_dict[mol_color], lw=4))
                    custom_labels.append(molecular_names[mol_color])
                else:

                    xpoints=(xbins[:-1]+xbins[1:])/2.0
                    ypoints=(ybins[:-1]+ybins[1:])/2.0

                    yall,xall=np.meshgrid(ypoints,xpoints)
                    xall=xall.flatten()
                    yall=yall.flatten()
                    zall=heatmap.flatten()
                    plt.tricontour(xall,yall,zall,levels=[np.max(zall)/10.0],colors='grey',zorder=10000)
                i+=1
    
    plt.xlabel('$\log_{10} R$ [au]')
    plt.ylabel('$T$ [K]')    
    if log_t_second:
    
    
        ticks_labels=[30,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
        ticks_labels_print=[30,100,200,300,'',500,'',700,'','',1000,'','','','',1500]
    
        plt.yticks(np.log10(ticks_labels),ticks_labels_print)
        
        plt.ylim([np.log10(temp_range[0]),np.log10(temp_range[1])]) 
    else:
        plt.ylim([temp_range[0],temp_range[1]]) 
    plt.legend(custom_lines,custom_labels,loc=(-0.1,1.05),ncol=max(len(custom_lines)//2,1))
    plt.savefig(prefix_fig+'_molecular_conditions_temp_by_radius_contour_version.pdf',bbox_inches='tight')

    if close_plots:
        plt.close()
    else:
        plt.show()

    print('Done!')
    print('Plot radial column density distribution..')


    ybins=np.linspace(coldens_range[0],coldens_range[1],nbins)


    print('Plotting radial temperature distribution..')
    for full_range in [True,False]:
            heatmap_dict={}
            mol_list=list(slab_prior_dict.keys())
            zorder=np.arange(len(mol_list))
            
            custom_lines=[]
            custom_labels=[]
            plt.figure(figsize=(9,6))
            i=0
            for mol in mol_list:
                print(mol)
                powerlaw=True 
                if f'{mol}:temis' in header or f'{mol}:temis' in fixed_dict:
                    powerlaw=False


                if full_range:
                    if 'ColDens' in slab_prior_dict[mol]:

                        idx_coldens_out=np.where(header==f'{mol}:ColDens')[0][0]
                    
                        coldens_out= paras[:,idx_coldens_out]
                        coldens_in= coldens_out
                    if f'{mol}:ColDens' in fixed_dict:

                        coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens']
                        coldens_out=coldens_in

                    if 'ColDens_tmax' in slab_prior_dict[mol]:
                        idx_coldens_in=np.where(header==f'{mol}:ColDens_tmax')[0][0]
                        coldens_in= paras[:,idx_coldens_in]
                    if f'{mol}:ColDens_tmax' in fixed_dict:
                        coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmax']

                    if 'ColDens_tmin' in slab_prior_dict[mol]:
                        idx_coldens_out=np.where(header==f'{mol}:ColDens_tmin')[0][0]
                        coldens_out= paras[:,idx_coldens_out]
                    if f'{mol}:ColDens_tmin' in fixed_dict:
                        coldens_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmin']

                    if f'{mol}: logradius' in header:
                        
                        idx_rin=np.where(header==f'{mol}: logradius')[0][0]


                        r_in= 10**paras[:,idx_rin]
                            
                    if f'{mol}: radius' in header:
                        idx_rin=np.where(header==f'{mol}: radius')[0][0]
                    
                        r_in= paras[:,idx_rin]
                    if f'{mol}:radius' in fixed_dict:
                        r_in=np.ones_like(paras[:,0])*fixed_dict[f'{mol}:radius']
                    if f'{mol}:logradius' in fixed_dict:
                        r_in=np.ones_like(paras[:,0])*10**fixed_dict[f'{mol}:logradius']

                    if 'q_emis' in prior_dict:

                        idx_qemis=np.where(header==f'q_emis')[0][0]
                        qemis=paras[:,idx_qemis]


                    r_out= r_in*(t_out/t_in)**(1/qemis)
                else:
                    idx_coldens_out=np.where(header==f'{mol}:\nlogDensCol at 0.15')[0][0]
                    idx_coldens_in=np.where(header==f'{mol}:\nlogDensCol at 0.85')[0][0]
                    idx_rout=np.where(header==f'{mol}:\nrout')[0][0]
                    idx_rin=np.where(header==f'{mol}:\nrin')[0][0]
            

                    coldens_in=paras[:,idx_coldens_in]
                    coldens_out=paras[:,idx_coldens_out]
                    r_in=paras[:,idx_rin]
                    r_out=paras[:,idx_rout]
                if powerlaw:
                    if debug:
                        print('Median coldens_in, coldens_out')
                        print(np.median(coldens_in),np.median(coldens_out))
                        print('Median r_in, r_out')
                        print(np.median(r_in),np.median(r_out))

                    
                    idx_ok=np.where(r_in!=0.0)[0]
                    coldens_out=coldens_out[idx_ok]
                    coldens_in=coldens_in[idx_ok]
                    r_out=r_out[idx_ok]
                    r_in=r_in[idx_ok]       
                    r_points,n_points=radius_temperature_relation(t_in=10**coldens_in,t_out=10**coldens_out,r_in=r_in,r_out=r_out,log_r=log_r)
            

            
                    n_points=np.log10(n_points)
                    if log_r:
                        r_points=np.log10(r_points)
                    heatmap, xedges, yedges = np.histogram2d(r_points, n_points, bins=[xbins,ybins])
                    heatmap_dict[mol]=heatmap
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    mol_color=mol
                    if '_comp' in mol_color:
                        idx_mol=mol_color.find('_comp')
                        
                        mol_color=mol_color[:idx_mol]
                        print(f'Changing {mol} to {mol_color}')
                    if '_absorp' in mol_color:
                        idx_mol=mol_color.find('_absorp')
                        
                        mol_color=mol_color[:idx_mol]
                        print(f'Changing {mol} to {mol_color}')
                    plt.imshow(heatmap.T, extent=extent,origin='lower',aspect='auto',cmap=dict_cmaps[mol_color],zorder=zorder[i])
                        
                    custom_lines.append(Line2D([0], [0], color=mol_colors_dict[mol_color], lw=4))
                    custom_labels.append(molecular_names[mol_color])
                    i+=1
            
            plt.xlabel('$\log_{10} R$ [au]')
            plt.ylabel('$\log_{10} \Sigma$ [cm$^{-2}$]') 

            if no_sigma:
                plt.ylabel('$\log_{10} N$ [cm$^{-2}$]')
            plt.ylim([coldens_range[0],coldens_range[1]]) 
            plt.legend(custom_lines,custom_labels,loc=(-0.1,1.05),ncol=max(len(custom_lines)//2,1))
            if full_range:
                plt.savefig(prefix_fig+'_molecular_conditions_coldens_by_radius_full_range.pdf',bbox_inches='tight')
            else:
                plt.savefig(prefix_fig+'_molecular_conditions_coldens_by_radius.pdf',bbox_inches='tight')
            if close_plots:
                plt.close()
            else:
                plt.show()
    print('Contour version')

    plt.figure(figsize=(9,6))
    for full_range in [False,True]:
        heatmap_dict={}
        mol_list=list(slab_prior_dict.keys())
        zorder=np.arange(len(mol_list))
        
        custom_lines=[]
        custom_labels=[]
        i=0
        for mol in mol_list:
            print(mol)
            powerlaw=True 
            if f'{mol}:temis' in header or f'{mol}:temis' in fixed_dict:
                powerlaw=False


            if full_range:
                if 'ColDens' in slab_prior_dict[mol]:

                    idx_coldens_out=np.where(header==f'{mol}:ColDens')[0][0]
                
                    coldens_out= paras[:,idx_coldens_out]
                    coldens_in= coldens_out
                if f'{mol}:ColDens' in fixed_dict:

                    coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens']
                    coldens_out=coldens_in

                if 'ColDens_tmax' in slab_prior_dict[mol]:
                    idx_coldens_in=np.where(header==f'{mol}:ColDens_tmax')[0][0]
                    coldens_in= paras[:,idx_coldens_in]
                if f'{mol}:ColDens_tmax' in fixed_dict:
                    coldens_in= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmax']

                if 'ColDens_tmin' in slab_prior_dict[mol]:
                    idx_coldens_out=np.where(header==f'{mol}:ColDens_tmin')[0][0]
                    coldens_out= paras[:,idx_coldens_out]
                if f'{mol}:ColDens_tmin' in fixed_dict:
                    coldens_out= np.ones(np.shape(paras[:,0]))*fixed_dict[f'{mol}:ColDens_tmin']
                if f'{mol}: logradius' in header:
                    
                    idx_rin=np.where(header==f'{mol}: logradius')[0][0]


                    r_in= 10**paras[:,idx_rin]
                        
                if f'{mol}: radius' in header:
                    idx_rin=np.where(header==f'{mol}: radius')[0][0]
                
                    r_in= paras[:,idx_rin]
                if f'{mol}:radius' in fixed_dict:
                    r_in=np.ones_like(paras[:,0])*fixed_dict[f'{mol}:radius']
                if f'{mol}:logradius' in fixed_dict:
                    r_in=np.ones_like(paras[:,0])*10**fixed_dict[f'{mol}:logradius']

                if 'q_emis' in prior_dict:

                    idx_qemis=np.where(header==f'q_emis')[0][0]
                    qemis=paras[:,idx_qemis]


                r_out= r_in*(t_out/t_in)**(1/qemis)
            else:
                idx_coldens_out=np.where(header==f'{mol}:\nlogDensCol at 0.15')[0][0]
                idx_coldens_in=np.where(header==f'{mol}:\nlogDensCol at 0.85')[0][0]
                idx_rout=np.where(header==f'{mol}:\nrout')[0][0]
                idx_rin=np.where(header==f'{mol}:\nrin')[0][0]
        

                coldens_in=paras[:,idx_coldens_in]
                coldens_out=paras[:,idx_coldens_out]
                r_in=paras[:,idx_rin]
                r_out=paras[:,idx_rout]
            if powerlaw:
                if debug:
                    print('Median coldens_in, coldens_out')
                    print(np.median(coldens_in),np.median(coldens_out))
                    print('Median r_in, r_out')
                    print(np.median(r_in),np.median(r_out))

                
                idx_ok=np.where(r_in!=0.0)[0]
                coldens_out=coldens_out[idx_ok]
                coldens_in=coldens_in[idx_ok]
                r_out=r_out[idx_ok]
                r_in=r_in[idx_ok]          
                r_points,n_points=radius_temperature_relation(t_in=10**coldens_in,t_out=10**coldens_out,r_in=r_in,r_out=r_out,log_r=log_r)
        

        
                n_points=np.log10(n_points)
                if log_r:
                    r_points=np.log10(r_points)
                heatmap, xedges, yedges = np.histogram2d(r_points, n_points, bins=[xbins,ybins])
                heatmap_dict[mol]=heatmap
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                mol_color=mol
                if '_comp' in mol_color:
                    idx_mol=mol_color.find('_comp')
                    
                    mol_color=mol_color[:idx_mol]
                    print(f'Changing {mol} to {mol_color}')
                if '_absorp' in mol_color:
                    idx_mol=mol_color.find('_absorp')
                    
                    mol_color=mol_color[:idx_mol]
                    print(f'Changing {mol} to {mol_color}')
                if full_range:
                    plt.imshow(heatmap.T, extent=extent,origin='lower',aspect='auto',cmap=dict_cmaps[mol_color],zorder=zorder[i])
                        
                    custom_lines.append(Line2D([0], [0], color=mol_colors_dict[mol_color], lw=4))
                    custom_labels.append(molecular_names[mol_color])
                else:

                    xpoints=(xbins[:-1]+xbins[1:])/2.0
                    ypoints=(ybins[:-1]+ybins[1:])/2.0

                    yall,xall=np.meshgrid(ypoints,xpoints)
                    xall=xall.flatten()
                    yall=yall.flatten()
                    zall=heatmap.flatten()
                    plt.tricontour(xall,yall,zall,levels=[np.max(zall)/10.0],colors='grey',zorder=10000)
                i+=1
    
    plt.xlabel('$\log_{10} R$ [au]')
    plt.ylabel('$\log_{10} \Sigma$ [cm$^{-2}$]')  

    if no_sigma:
        plt.ylabel('$\log_{10} N$ [cm$^{-2}$]')
    plt.ylim([coldens_range[0],coldens_range[1]]) 
    plt.legend(custom_lines,custom_labels,loc=(-0.1,1.05),ncol=max(len(custom_lines)//2,1))
    plt.savefig(prefix_fig+'_molecular_conditions_coldens_by_radius_contour_version.pdf',bbox_inches='tight')

    if close_plots:
        plt.close()
    else:
        plt.show()

    print('Done!')

print('..Finished!!')