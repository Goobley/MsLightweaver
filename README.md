## MsLightweaver

Simple application of the [Lightweaver framework](https://github.com/Goobley/Lightweaver) to reprocessing RADYN simulations, allowing investigation of different radiative transfer methods and a minority species approach. Advection is directly based on RADYN's approach, but using a coloured finite difference Jacobian. PRD in the time-dependent simulations currently has a tendency of getting stuck in cycles.

Note: you will need an `atmost.dat`  (`IATMT = 1` in `param.dat`) from RADYN, the CDF alone is insufficient, as every timestep is needed.