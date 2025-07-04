
# Variable dictionary

| Name              | Description                                                                             | Component | Typical range |
| ----------------- | --------------------------------------------------------------------------------------- | --------- | ------------- |
| **General**       |                                                                                         |           |               |
| dist              | Distance to the object                                                                  |           |               |
| tstar             | Stellar temperature (K)                                                                 | Star      |               |
| rstar             | Stellar radius (R_sun)                                                                  | Star      |               |
| bb_star           | Should a black body be used for the star, otherwise a stellar spectrum has be to loaded | Star      | True, False   |
| incl              | Inclination of the system                                                               |           | (0.0,90.0)    |
| E(B-V)            | First extinction parameter                                                              |           | (0.1,0.9)     |
| Rv                | Second extinction parameter                                                             |           | (3.0,4.0)     |
| **Uncertainty**   |                                                                                         |           |               |
| sigma_obs         | Observational uncertainty as a factor of flux                                           |           |               |
| log_sigma_obs     | Observational uncertainty as a factor of flux (log scale)                               |           | (-5,-1)       |
| sigma_obs_abs     | Observational uncertainty in absolute terms (Jy)                                        |           |               |
| log_sigma_obs_abs | Observational uncertainty in absolute terms on a log scale (Jy)                         |           |               |
| **Dust**          |                                                                                         |           |               |
| t_rim             | Temperature of the inner rim (K)                                                        | Rim       | (500,1500)    |
| tmax_rim          | Maximum temperature of the inner rim (K)                                                | Rim       | (500,1500)    |
| tmin_rim          | Minimum temperature of the inner rim (K)                                                | Rim       | (10,1000)     |
| q_rim             | exponent of the temperature power law of the inner rim                                  | Rim       | (-1,-0.1)     |
| tmax_mp           | Maximum temperature of the midplane (K)                                                 | Midplane  | (500,1500)    |
| tmin_mp           | Minimum temperature of the midpllane (K)                                                | Midplane  | (10,1000)     |
| q_mid             | exponent of the temperature power law of the midplane                                   | Midplane  | (-1,-0.1)     |
| tmax_s            | Maximum temperature of the surface layer (K)                                            | Surface   | (500,1500)    |
| tmin_s            | Minimum temperature of the surface layer(K)                                             | Surface   | (10,1000)     |
| t_change_s        | Change between two dust compositions (0=tmin_s, 1=tmax_s)                               | Surface   | (0.01,0.99)   |
| q_thin            | exponent of the temperature power law of the surface layer                              | Surface   | (-1,-0.1)     |
| q_thin_cold       | exponent of the cold half of the temperature power law of the surface layer             | Surface   | (-1,-0.1)     |
| tmax_cold_s       | Maximum temperature of the cold part of the surface layer (K)                           | Surface   | (500,1500)    |
| tmin_hot_s        | Minimum temperature of the hot part of the surface layer(K)                             | Surface   | (10,1000)     |
| tmax_abs          | Maximum temperature of the surface absorption layer (K)                                 | Surface   | (500,1500)    |
| tmin_abs          | Minimum temperature of the surface absorption layer(K)                                  | Surface   | (10,1000)     |
| t_change_abs      | Change between two dust compositions (0=tmin_abs, 1=tmax_abs)                           | Surface   | (0.01,0.99)   |
| q_abs             | exponent of the temperature power law of the surface absorption layer                   | Surface   | (-1,-0.1)     |
| q_abs_cold        | exponent of the cold half of the temperature power law of the surface absorption layer  | Surface   | (-1,-0.1)     |
| tmax_cold_abs     | Maximum temperature of the cold part of the surface absorption layer (K)                | Surface   | (500,1500)    |
| tmin_hot_abs      | Minimum temperature of the hot part of the surface absorption layer(K)                  | Surface   | (10,1000)     |
| **Gas**           |                                                                                         |           |               |
| q_emis            | exponent of the molecular temperature power law                                         | Gas       | (-1,-0.1)     |

Instead of providing the observational uncertainty as a fitting parameter, you can also provide an array called sig_obs with absolute values in Jy. Please note that a previous version (before July 2025) had a bug when running the gas_only fit with fitting the uncertainty in absolute values, that resulted in bad convergence. This is fixed now!   
If you are using two temperature power laws for the surface/absorption layer you can force them to no overlap by adding 'no_overlap_t=True' to the input file.

# Abundance dictionary

This dictionary should contain the dust file names and their scaling factors

# Molecular dictionary

The dictionary should contain the molecular names as keys, which contain a second dictionary with the settings.

| Name         | Description                                                                                               | Typical range |
| ------------ | --------------------------------------------------------------------------------------------------------- | ------------- |
| ColDens      | Constant column density (on log scale)                                                                    | (14,24)       |
| ColDens_tmin | Column density at the minimum temperature                                                                 | (14,24)       |
| ColDens_tmax | Column density at the maximum temperature                                                                 | (14,24)       |
| temis        | Constant temperature (K)                                                                                  | (25,1500)     |
| tmax         | Maximum temperature (K)                                                                                   | (25,1500)     |
| tmin         | Minimum temperature (K)                                                                                   | (25,1500)     |
| radius       | For single slab this is the radius of the emitting area for temperature power laws it is the inner radius |               |
| r_area       | For temperature power laws this corresponds to the radius of the emitting area                            |               |
