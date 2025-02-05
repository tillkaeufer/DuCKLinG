# Dust Continuum Kit with Line emission from Gas (DuCKLinG)

<img src="./DuCKLinG_logo.png" width="250" />

This repository gives you all the files to run the DuCKLinG model.  
It can be run as a forward model or in retrieval with MultiNest or UltraNest.

A description of the model can be seen in [Kaeufer et al. 2024](https://www.aanda.org/articles/aa/pdf/2024/07/aa49936-24.pdf)

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
  
Additionally, it is recommended to install [OpenMPI](https://www.open-mpi.org/) and mpi4py to run the retrieval in parallel.

If you run the multinest retrievel you also need to install multinest for example [here](https://github.com/JohannesBuchner/MultiNest) or [here](https://github.com/farhanferoz/MultiNest).

If you run the ultranest retrieval (recommended for high dimensional parameter spaces) install [UltraNest](https://johannesbuchner.github.io/UltraNest/installation.html)

### Download gas files

The molecular emission in DuCKLinG is calculated from a grid of 0D ProDiMo-slab models calculated by [Arabhavi et al. (2024)](https://doi.org/10.1126/science.adi8147).  
The files are available in this [folder](https://drive.google.com/drive/folders/1XAEgrss9oupWWGo_nNQyHwQFsfMxEiKq?usp=sharing).  
Download the folder and add it in DuCKLinG/LineData to have the example work without changing the paths.  
Be aware that this requires about 15 GB of storage. You can also choose to download only the data for the molecules you are interested in.
If you want to download the folder using the terminal you can do this by installing gdown (`pip install gdown`) and executing the following command: `gdown --folder https://drive.google.com/drive/folders/1XAEgrss9oupWWGo_nNQyHwQFsfMxEiKq --remaining-ok`

| :exclamation:  On January 22, I uploaded a new water slab file that is using HITRAN. Previously the slab data was based on Lamda. If you are running water fits, I highly recommend downloading the HITRAN data. However, you can still access the LAMDA data, which is now saved as 12_LamH2O. |
|-----------------------------------------|


## Getting started

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

The idea is that you create an input file that defines all the settings, priors, and observation that you want to use and then you simply execute it the same way as shown for the *retreival example*.

| :exclamation:  Even if you run the retrieval in parallel make sure to first run everything in a single core till the retrieval part of the script starts. This makes sure that the slab grids are already binned to your observation and this is not done N times. |
|-----------------------------------------|

Below you find a detailed describtion of all the settings specified in the input file. I recommend that you copy the example input file and adjust it to your needs.  
Start simple, see if it works and then you can make it more complicated.

### Input file

The input file provides all the information for the retrieval. You don't need to change anything in the python scripts.

There are different sections in the input files that govern different things.

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

#### settings for the retrieval  
- use_ultranest: Set it to True if you are running a ultranest retrieval and to False if you run multinest (it is needed for the plotting routines to know how the output format looks like)
  
For UltraNest: 
- length_ultra: We are using slice_sample for ultanest. Therefore, you need to provide how many steps you are taking. The integer here is multiplied by the number of parameters that you are using to derive the number of steps. 2 works fine in my case. If you want to check if everything converged, double the number, run it again, and see if anything changed.

For MultiNest:

There are two predefined settings that can be selected with 
- fast_retrival: If True the settings are n_live_points = 1000, evidence_tolerance = 5.0, and sampling_efficiency = 0.8, otherwise the settings are n_live_points = 1000, evidence_tolerance = 0.5, and sampling_efficiency = 0.3

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
- plot_dust: This will plot all the different dust opacity curves individually. This requires a lot of memory and does not work for large retrievals.
- reduce_post: If you want to have a quick look if the retrieval works and do not care about the exact retrieved values (or want to plot the dust individually but do not have the memory) you can provide reduce_post followed by an integer i (separated by a space). This will select i models for the posterior and calculate everything only for these models
- no_spectrum: For only plotting the parameter posterior and ignoring the (memory-intensive and time-consuming) step of plotting the spectra add this option.
- close: this will avoid that all the plots are displayed when they are done. They will simply be saved.
- all: this option will automatically run the other two plotting routines as well
- all_plus: this option also runs the two other plotting routines and overplots the posterior in plot_mol_contributions-input.py as well.

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

If you want to plot the molecular column densities, temperatures and radii this is possible with the follwoing function:

> python plot_mol_conditions-input.py ./path/to/inputfile

This function requires you to run the plotting routine before with the save option (save or save_all).  
This function has a couple of options that allow you to change the plotting. They are called by adding them after the input file.

- nbins: Number of bins for the colour maps (e.g. nbins 100)
- npoints: Number of points that are calculated along the temperature power laws (e.g. npoints 1000)
- log_t:True : This envokes that both plots use a logarithmic scale for the temperature.
- log_t:False : This envokes that both plots use a linear scale for the temperature.
- log_t_first:True : This envokes that the first plot (temperature vs. column density) uses a logarithmic scale for the temperature.
- log_t_second:False : This envokes that the second plot (temperature vs. radius) uses a linear scale for the temperature.
- radial_range: This argument sets the limits of the radial range that is plotted (on a log scale, e.g. radial_range [-2,0]). If you leave it blank, it is automatically set to the radial profiles that are retrieved.
- coldens_range: This argument sets the limits of the column density range that is plotted (on a log scale, e.g. coldens_range [14,24])
- temp_range: This argument sets the limits of the temperature range that is plotted (on a linear scale, e.g. temp_range [100,1500])  
- close: Closing the plots instead of showing them to the user.




## Troubleshooting
- It is important to delete MultiNest files when you re-run a model that was interrupted due to any kinds of errors.
- If you have your own sigma of your data, comment out 'log_sigma_obs' and define 'sig_obs'.
- If the path to output directory is too long, Multinest cannot save its output file names properly due to the limits on fortran.
- If you are running the plotting rountine on a mac, there might be an error if you are using the custom_list argument (e.g python plot_retrieval_results.py custom_list [[5,40]]), you can fix it by adding 'noglob' to the line (e.g. noglob python plot_retrieval_results.py custom_list [[5,40]])
- If you get errors that some functions are unknown, make sure that you are in the DuCKLinG directory when running your code. This is necessary because functions are loaded from utils.py.
  
## Licence
Tbd
