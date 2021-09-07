## MsLightweaver

Early attempt to apply the [Lightweaver framework](https://github.com/Goobley/Lightweaver) to reprocessing RADYN simulations, allowing investigation of different radiative transfer methods and a minority species approach. Advection is handled via an explicit scheme that does not play particularly well with RADYN's implicit scheme. This is implemented by MsLightweaverAdvector using the (not great) code in HydroWeno.

Note: you will need an `atmost.dat`  (`IATMT = 1` in `param.dat`) from RADYN, the CDF alone is insufficient, as every timestep is needed.