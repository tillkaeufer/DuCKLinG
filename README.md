# Dust Continuum Kit with Line emission from Gas (DuCKLinG)

<img src="./DuCKLinG_logo.png" width="250" />

This repository gives you all the files to run the DuCKLinG model.  
It can be run as a forward model or be used for retrievals with [MultiNest](https://github.com/JohannesBuchner/MultiNest) or [UltraNest](https://github.com/JohannesBuchner/UltraNest).

A description of the model can be seen in [Kaeufer et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...687A.209K/abstract). If the code is useful for your research, please get in touch (t.kaeufer@exeter.ac.uk). I'm very happy to discuss your project and will hopefully be able to help you with any problems regarding the code.

Please have a look at the LICENSE file to understand the usage and distribution conditions.  
This code uses a GPLv3 license which includes the kind request that you cite [Kaeufer et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...687A.209K/abstract) in scientific publications that use the code. Please also consider citing [Arabhavi et al. 2024](https://ui.adsabs.harvard.edu/abs/2024Sci...384.1086A/abstract) who introduced the slab models that form the basis of the gas component of the model and [Juhász et al. 2009](https://ui.adsabs.harvard.edu/abs/2009ApJ...695.1024J/abstract) who first introduced the general concept that underpins the dust components. For citations of the dust opacities see references in [Kaeufer et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...687A.209K/abstract) and [Jang et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...691A.148J/abstract).

## Installation

DuCKLinG does not need any installation it is a Python object.
There are several Python packages that need to be installed in your environment. Many of them might be installed already.  
Here is a list of all packages that are loaded in at the beginning of the scripts:

- numpy
- matplotlib
- pickleshare 
- glob2 
- scipy
- [spectres](https://pypi.org/project/spectres/)
- json5
- uuid
- [multiprocess](https://pypi.org/project/multiprocess/)
- PyAstronomy
- corner
- pymultinest (only needed if MultiNest is run)
- argparse
- ultranest (if you want to run ultranest)
- numba
- h5py
- dust_extinction

You can install them individually or run:
`pip install -r requirements.txt`  
This will install all of them at once.
  
Additionally, it is recommended to install [OpenMPI](https://www.open-mpi.org/) and mpi4py to run the retrieval in parallel.

If you run the multinest retrieval you also need to install multinest for example [here](https://github.com/JohannesBuchner/MultiNest) or [here](https://github.com/farhanferoz/MultiNest).

If you run the ultranest retrieval (recommended for high dimensional parameter spaces) install [UltraNest](https://johannesbuchner.github.io/UltraNest/installation.html)

Generally, my experience is that ultranest is the only reliable option for very complex, high-dimensional parameter spaces, but that MultiNest is much faster for simpler cases. So, if you are able to install both, it's best to run MultiNest retrievals and switch to UltraNest when they become unfeasible.

### Download gas files

If you are only interested in fitting the dust component you do not need to download the slab files. Instead you can create an empty folder caleld 'LineData' in the DuCKLinG directory and everything should be working.

The molecular emission in DuCKLinG is calculated from a grid of 0D ProDiMo-slab models calculated by [Arabhavi et al. (2024)](https://doi.org/10.1126/science.adi8147).  
The files are available in this [folder](https://drive.google.com/drive/folders/1XAEgrss9oupWWGo_nNQyHwQFsfMxEiKq?usp=sharing).  
Download the folder and add it in DuCKLinG/LineData to have the example work without changing the paths.  
Be aware that this requires about 15 GB of storage. You can also choose to download only the data for the molecules you are interested in.
If you want to download the folder using the terminal you can do this by installing gdown (`pip install gdown`) and executing the following command: `gdown --folder https://drive.google.com/drive/folders/1XAEgrss9oupWWGo_nNQyHwQFsfMxEiKq --remaining-ok`

Please note that there are two versions of water slab grid. 12_H2O runs on HITRAN data (recommended) and 12_LamH2O uses LAMDA data.


## Getting started

### Hands-on session

If you are new to the program, it might be a good idea to check out the [DuCKLinG hands-on session](https://github.com/tillkaeufer/duckling_hands-on_session) to get started. It should be self-explanatory and good start to get things running.  
Otherwise, there are also a few retrieval examples in the repository you are currently at. Below you see a quick introduction.

### Forward model

The notebook forward_model gives a quick introduction to the different functionalities of the model.  
You can run it and see how different molecular conditions and dust species change the resulting output.

### Retrieval example

There is an example input file in the Example folder.
You can use this as a test ground if everything works.

Run one of the following commands  
> python retrieval-input.py ./Input_files/example_input-wo_extinction.txt  
> python retrieval-input.py ./Input_files/example_input-wo_extinction_multinest.txt   
> python retrieval-input.py ./Input_files/example_input-extinction.txt  
> python retrieval-input.py ./Input_files/example_input-absorption.txt  
> python retrieval-input.py ./Input_files/example_input-dust_only.txt   

to start the retrieval.

If you want to run it on multiple (N) cores you can use

> mpiexec -n N python retrieval-input.py ./path/to/inputfile  



but first run it once on a single core (see warning below).

The example should be reasonable quick in a single core, but for real world problems multiple cores are highly recommended.

After is (hopefully) finished, you can run:

> python plot_retrieval_results.py ./path/to/inputfile

and enjoy some plots.


### Exploring the slab grid

If you just want to have a look at the different slab models that are used as input for the model you can use the analysing_slab_grid notebook.

## How to run

The idea is that you create an input file that defines all the settings, priors, and observation that you want to use and then you simply execute it the same way as shown for the *retrieval example*.

| :exclamation:  Even if you run the retrieval in parallel make sure to first run everything in a single core till the retrieval part of the script starts. This makes sure that the slab grids are already binned to your observation and this is not done N times. |
|-----------------------------------------|

Below you find a detailed describtion of all the settings specified in the input file. I recommend that you copy the example input file and adjust it to your needs.  
Start simple, see if it works and then you can make it more complicated.

### Input file

The input file provides all the information for the retrieval. You don't need to change anything in the python scripts.

There are different sections in the input files that govern different things.

The README.md file in the Input_files folder explains the meaning of all parameters that the model uses.

#### Settings
This is where the settings for background data are provided:

- bayesian_folder: general folder where you want to save the output of your retrieval
- subfold: The retrieval will be saved in bayesian_folder+subfold
- run_number: Unique ID of the run you are about to start. All the files will be saved as 'test_'+run_numer
  If bayesian_folder+subfold+'test_'+run_nubmer already exists, the retrieval will be continued at the point where you stopped it the last time
- dust_path: path to your dust opacity files
- slab_folder: path to the slab grid
- slab_prefix: the number given to the slab grid. Number 12 is the one provided with this repository. This is the Slab grid by [Arabhavi et al. (2024)](https://doi.org/10.1126/science.adi8147) binned to a spectral resolution of $R=3500,3000,2500,1500$ for channels 1 to 4 of MIRI, respectively.

 

Optional settings to investigate specific behaviours:  
You can limit the integrated flux of a molecule to a value in $W/m^-2$.  
This might be useful to eliminate quasi-continua of certain molecules.
For doing so you have to set *limit_integrated_flux=True* and then for example *limit_flux_dict={'C4H2':4.5e-19}*  
It is also possible to limit the flux only over a certain wavelength window. The idea is that if you choose this window where flux is only popping up in case of strong quasi-continua you can limit the flux without penalising fits of the main features. If is done by creating a two layer dictionary (e.g. *limit_flux_dict={'C4H2':{'wave':\[10.0,15.0\],'flux':4.5e-19}}*). The wavelengths should be provided in micron.  
To get to know the integrated fluxes you can run a first fit and use the plot_mol_contributions-input.py rountine which will print the integrated fluxes over the whole spectrum for all modelled species. To get the flux in a certain wavelength window you can now insert a fake limit_flux_dict in the input file with your wished wavelengths limit and plot_mol_contributions-input.py will now also print the integrated flux in that window.

Similarly, you can limit the maximum emitting area of a molecule.  
This is done with *limit_radial_extent=True* and for example *limit_radius=2* (limiting radius in au)


#### Parameters

Here we define the setup of the model we want to use.

- sample_all: If False, we are using the method described in the DuCKLinG paper to reduce the dimensionality of the parameter space. If you have a lot of spare time, feel free to set it to True and see what happens. This will sample also the linear parameters in a Bayesian way (and take a lot of time).
- use_bb_star: If True a black body is used as the star. If False we load in the stellar spectrum set in the next variable.
- stellar_file : path to the stellar file. For the file format have a look a the example file
- rin_powerlaw: The inner rim is described by a black body. However, you can also opt to sample it as a temperature power law (be aware that you then need to set the parameters for that).
- dust_species_list: This is a list with all the dust opacity files that you want to use. The files have to be located in dust_path. If you don't want to use this option just delete the list and delete all the surface layer parameters from the dictionaries as well.
- absorp_species_list: If you want to recreated dust absorption features instead of emission you can use this list to provide the dust opacity files that you want to use. The files have to be located in dust_path. Make sure that you use the absorption parameters then as well (e.g. tmax_abs, tmin_abs, and q_abs or 'temp_abs'). If you leave this list empty dust absorption will be ignored. 
- if you are using extinction you can add a line that loads the extinction model you are using. An example is provided in the example input files. If you want to use extinction you have to define Rv and E(B-V) in either the fixed_dict or the prior_dict.
- if you want a change of dust composition between the hot and cold region of the dust emission/absorption layer you can follow all species names in dust_species_list you want in the hot part with '_hot' and the ones for the cold part by '_cold' (e.g. 'MgOlivine0.1.Combined.Kappa_rv0.1.dat_hot'). Additionally, you need to include the t_change_s/t_change_abs parameter that determines the transition temperature between hot and cold (as a value between 0 and 1). An example of this can be found in the example input files folder.  
- similarly you can choose for the cold dust to have a different slope in temperature power law by setting the parameter q_thin_cold.  
- if you additionally want a gap in the temperature profile you can define tmin_hot_s and tmax_cold_s (if you include 
no_overlap_t=True you will additionally constrain that the two power laws do not overlap).  

#### fixed parameters
Setting the fixed parameters.  
For some parameters, you might want to fix the value in the retrieval (e.g. distance).
For doing to sett fixed_paras=True and provide a fixed_dict dictionary with the parameter names and corresponding values.

### Priors

Finally, we arrived at the heart of the input file.  
In this section, you are setting the prior ranges for all the parameters you want to retrieve.    
For the name of the molecules have a look at the slab data that you downloaded. It is important that you are using the names that the files have.  
Files with molecular_name+'_I' or molecular_name+'_II' use a mix of the 12C and 13C isotopologues (see [Arabhavi et al. 2024](https://doi.org/10.1126/science.adi8147)).   
If you want multiple components of the same molecule you can simply provide multiple priors for e.g. 'H2O', 'H2O_comp2', 'H2O_comp3'...   
If you are using molecular absorption just call the molecules '_absorp' (e.g. 'H2O_absorp'). You can combine molecular emission and absorption to your liking. If you are using temperature power laws for the absorption the exponent will be the same as the emission exponent. If you need an independent parameter, please let me know.  
Multiple absorption components of the same molecule can be introduced by combining '_absorp' and '_comp' (e.g. 'H2O_absorp_comp2').  


- prior_dict: Prior dictionary for all parameters that are not related to the slab grid (except for q_emis)
- slab_prior_dict: Prior dictionary for all parameters related to the slab grid. If log_coldens=True to column densities are provided as their logarithm.
  - If the molecule is supposed to emit along a temperature range, set tmin and tmin for that molecule. If you want to emit at a single temperature set t_emis. If all molecules are emitting without temperature ranges you can delete q_emis from your prior_dict. 
  - If the molecule is supposed to emit along a column density power law, set ColDens_tmin and ColDens_tmax, otherwise use ColDens.

| :exclamation: Note that column density ranges are only possible if a temperature range is used, not for single temperatures. |
|-----------------------------------------|

#### code to load observations

Here you provide a short Python script to load your observation.  
In the end, it is important that flux_obs provided the fluxes in Jansky at the wavelength points lam_obs (in microns).  
You can also provide the corresponding uncertainties as sig_obs.

By default the likelihood function is as follows:  

$L=\sum_{i} -\frac{1}{2}\times \left[\log({2\pi\times \sigma_i^2})+\frac{(f_{\rm model,i}-f_{\rm obs,i})^2}{\sigma_i^2}\right]$

With $L$ being the likelihood and $i$ the index of the the wavelength point of the observation. $\sigma_i$, $f_{\rm model,i}$, and $f_{\rm obs,i}$ are the uncertainty, model prediction, and observed flux at the $i$-th wavelengths point.  

Due to the change of spectral resolving power of MIRI over wavelengths, it means that there are more points at shorter wavelengths which makes this part of the spectrum dominating the likelihood function.  If you want every wavelengths interval to have the same weight you can do so by adding the following line to the input file:

weights_obs=calc_weights(lam_obs)

This will activate this likelihood functions:  
$L=\sum_{i} -\frac{w_i}{2}\times \left[\log({2\pi\times (\sigma_i)^2})+\frac{ (f_{\rm model,i}-f_{\rm obs,i})^2}{(\sigma_i)^2}\right]$   
or (if weight_scale_sigma=True in the input file):  
$L=\sum_{i} -\frac{1}{2}\times \left[\log({2\pi\times (\sigma_i/w_i)^2})+\frac{ (f_{\rm model,i}-f_{\rm obs,i})^2}{(\sigma_i/w_i)^2}\right]$.  

with $w_i$ being the weights (which are normalized to the number of wavelength points).  
Alternatively, you can also define your own custom weights.  As long as their have the same lengths as lam_obs, everything should work. You can also set to_wave=False in the calc_weights function to get a equal weighting by spectral resolution and not wavelength intervall. If the uncertainties are treated as a free parameter, weighting should only be applied very carefully; checking the retrieved uncertainties. It can results in very long retrievals.  

It is also possible to mask some wavelength regions. You can for example fit only the unblended water lines by creating a lam_obs file that only contains points close to the lines. However, this will technically effect the binning of the data (e.g. very large bins between the lines).  
Therefore, it is possible to provide a lam_obs_full (and flux_obs_full) array that contains all the wavelengths points before masking.  
The program will then bin the slab data to lam_obs_full and select the wavelengths points that are present in lam_obs.  
On top of that, the plotting will plot the full spectrum and highlight the selected regions. If you are using uncertainties or weights, please also create a weights_obs_full and sig_obs_full array.


#### settings for the retrieval  
- use_ultranest: Set it to True if you are running a ultranest retrieval and to False if you run multinest (it is needed for the plotting routines to know how the output format looks like)
  
**For UltraNest:**

- slice_sampler: Set to True to use slice sampling, otherwise nested sampling is applied
- length_ultra: If you are using slice_sample for ultranest, you need to provide how many steps you are taking. The integer here is multiplied by the number of parameters that you are using to derive the number of steps. 2 works fine in my case. If you want to check if everything converged, double the number, run it again, and see if anything changed.
- n_live_points: Number of live points. Typically 400, but some cases might require more.
- evidence_tolerance: The change in evidence at which the fit is considered converged. 0.5 is a good values if you want to calculate Bayes factors, for fast retrievals 5 should be fine as well.
- frac_remain: Stop criterion if the fraction of the integral is left in the remainder. For values like 0.001 will make sure you cover all peaks, larger numbers like 0.5 are okay if the posterior is simple.
- dlogz: Evidence uncertainty that should be achieved for convergence. A good value is 0.5.
- dKL: posterior uncertainty for convergence. A good value is 0.5.
- max_iter: for Multinest it is possible to provide a maximum number of iterations after which the fitting is stopped. This parameter should be used very carefully. By default it is negative and therefore does not affect the retrieval.  
- imp_nest_samp: using importance_nested_sampling for Multinest (default is False).  

**For MultiNest:**

There are two predefined settings that can be selected with 
- fast_retrival: If True the settings are n_live_points = 400, evidence_tolerance = 5.0, and sampling_efficiency = 0.8, otherwise the settings are n_live_points = 1000, evidence_tolerance = 0.5, and sampling_efficiency = 0.3

Alternatively, you can set the settings yourself, by setting n_live_points, evidence_tolerance, and sampling_efficiency in the input file (remove the # in front of them)

## How to plot

After running a retrieval you might want to have a look at the results.  
The plot_retrieval_results program does that for you.

You can run it with:  

> python plot_retrieval_results.py ./path/to/inputfile [option]

Additionally, it accepts a large range of options to specify your plotting needs. These options are simply added after the path to the input file with a space between every argument.

- save: This saves the full posterior (retrieved parameters and linear parameters) as a numpy array
- save_all: Saving the full posterior and the posterior of fluxes. 
- custom_list: The output will create figures of the model fluxes compared to the observation. It will do one for the full wavelength region from $0.2\rm \mu m$ to $200 \rm \mu m$. Additionally, zoom_ins will be created to more specific wavelength regions (default is  $2\rm \mu m$ to $40\rm \mu m$). You can specify the regions in a list. use the option custom_list and add a list after that (separated by a space) e.g. [[4,20],[4,10],[10,20]]
- plot_dust: This will plot all the different dust opacity curves individually. If you provide an integer after this (e.g. plot_dust 5) it will only plot the five dust species with the highest peak fluxes. This might make the plotting easier to understand. Since all dust curves are saved individually this option requires a lot of memory and might not work for very large retrievals. In that case you can use the reduce_post option to make it still work.  
- reduce_post: If you want to have a quick look if the retrieval works and do not care about the exact retrieved values (or want to plot the dust individually but do not have the memory) you can provide reduce_post followed by an integer i (separated by a space). This will select i models for the posterior and calculate everything only for these models
- no_spectrum: For only plotting the parameter posterior and ignoring the (memory-intensive and time-consuming) step of plotting the spectra add this option.
- close: this will avoid that all the plots are displayed when they are done. They will simply be saved.
- all: this option will automatically run the other two plotting routines as well
- all_plus: this option also runs the two other plotting routines and overplots the posterior in plot_mol_contributions-input.py as well.  
- savetxt: this option will save all the model components (median fluxes with standard deviations) in txt files. If plot_dust=True it will also save all the individual dust components.  

Hopefully, this will plot/save you all the data you are interested in.  

Next you can plot the contributions of all molecules to the median probable model.  
This can be done by running:

> python plot_mol_contributions-input.py ./path/to/inputfile

This function requires you to run the plotting routine before with the save option (save or save_all).  
The save_all option will allow you to also overplot the posterior of models to the figures.  
For details on the median probable model have a look at [Kaeufer et al. 2024](https://www.aanda.org/articles/aa/pdf/2024/07/aa49936-24.pdf).

The options for this routine are as follows:

- preliminary: this can be used if your run just started and the posterior file has only one entry. It is a great way to check if the fit is doing what you want.
- simple: this selects the median probable model on the multinest posterior and not the full posterior. The advantage is that you can run it without running any plotting routine before and before the fit is finished. The difference to preliminary is that this can be used when the posterior file has multiple entries already.
- close: Closing the plots instead of showing them to the user.
- width: this phrase should be followed by a float. It specifies how large the wavelengths windows are that are plotted. Default is 2.0.
- no_comp: the latest version of the code will plot different components of the same molecules in different colours (alpha=1/component_number). If you want the old behaviour that all components of the same molecule are plotted in the same colour add this argument.
- cold_water: For all molecules the temperature contibutions to every part of the spectrum is plotted (if the argument no_temp is not used). If cold_water is added as an argument this means, that the colorscale for water will be ranging from blue to red with the middle temperature at 400K, which is often considered the splitting temperature between warm and cold water.  

If you want to plot the molecular column densities, temperatures and radii this is possible with the following function:

> python plot_mol_conditions-input.py ./path/to/inputfile

This function requires you to run the plotting routine before with the save option (save or save_all).  
This function has a couple of options that allow you to change the plotting. They are called by adding them after the input file.

- nbins: Number of bins for the colour maps (e.g. nbins 100)
- npoints: Number of points that are calculated along the temperature power laws (e.g. npoints 1000)
- log_t:True : This enforces that both plots use a logarithmic scale for the temperature.
- log_t:False : This enforces that both plots use a linear scale for the temperature.
- log_t_first:True : This enforces that the first plot (temperature vs. column density) uses a logarithmic scale for the temperature.
- log_t_second:False : This enforces that the second plot (temperature vs. radius) uses a linear scale for the temperature.
- radial_range: This argument sets the limits of the radial range that is plotted (on a log scale, e.g. radial_range [-2,0]). If you leave it blank, it is automatically set to the radial profiles that are retrieved.
- coldens_range: This argument sets the limits of the column density range that is plotted (on a log scale, e.g. coldens_range [14,24])
- temp_range: This argument sets the limits of the temperature range that is plotted (on a linear scale, e.g. temp_range [100,1500])  
- close: Closing the plots instead of showing them to the user.

## Troubleshooting
- It is important to delete MultiNest files when you re-run a model that was interrupted due to any kinds of errors.
- If you have your own sigma of your data, comment out 'log_sigma_obs' and define 'sig_obs'.
- Similarly, if you are not using one of the components (e.g. dust absorption or emission) it is important to get rid of all the associated parameters (e.g. if dust absorption is used the emission parameters tmax_s, tmin_s, and q_thin should be deleted from your prior_dict/fixed_dict).
- If the path to output directory is too long, Multinest cannot save its output file names properly due to the limits on fortran.
- If you are running the plotting routine on a mac, there might be an error if you are using the custom_list argument (e.g python plot_retrieval_results.py custom_list [[5,40]]), you can fix it by adding 'noglob' to the line (e.g. noglob python plot_retrieval_results.py custom_list [[5,40]])
- If you get errors that some functions are unknown, make sure that you are in the DuCKLinG directory when running your code. This is necessary because functions are loaded from utils.py.
- If a stellar spectrum is provided, it has to be longer than the observational data. Otherwise, you might get "ValueError: A value (---) in x_new is above the interpolation range's maximum value (---)."
  
## Paper that use the model

- Introduction of the model and application to a TTauri star: [Kaeufer et al. 2024a](https://ui.adsabs.harvard.edu/abs/2024A%26A...687A.209K/abstract)
- Application to a very low-mass star to disentangle the dust and gas contributions: [Kaeufer et al. 2024b](https://ui.adsabs.harvard.edu/abs/2024A%26A...690A.100K/abstract)
- Using the dust component of the model to determine the dust composition of PDS70: [Jang et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...691A.148J/abstract)
- Determine the dust composition of highly irradiated very distant disk around XUE 1: [Portilla-Revelo et al. 2025](https://doi.org/10.48550/arXiv.2504.00841)

