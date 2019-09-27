## The mumag standard problem #4

This is a template solution provided to the [mumag standard problem #4](https://www.ctcms.nist.gov/~rdm/std4/spec4.html).
The solution obtained by this code is compared to D'Aquino et al's solution using the midpoint scheme, but it can also be
compared reasonably to [any of the other solutions](https://www.ctcms.nist.gov/%7Erdm/std4/results.html) available. The settings.inp files present in here can be used in any other case as they present all possible options available to the code.

## Details on solution

To obtain this particular solution, we use a 3D cell discretization scheme with **cubic** cells that are equal in size towards every
direction, and in this case, of 2.5 nm (which can be changed to demonstrate it is independent of mesh size). The magnetization
inside of each cell is assumed uniform. There is a total of Nx Ny Nz = N cubic cells with size 2.5nm each, corresponding thus to
a finite difference scheme applied to the sample.

The exchange interactions are obtained by a "6-neighbor dot product" of all the neighboring cells to any given cell. The stray
field is obtained by assuming it uniform inside of each cell and iterating over every N² pairs of cells with an equivalent dipole-dipole coupling. To establish convergence to a local minimum we make use of a data.dat file which keeps track of the system's energy, and assume equilibrium when it stops changing past the 10^(-6)fJ range.

The time integration (for the images shown) was done with a simple explicit Multistep method based on the [implicit spherical midpoint scheme](https://arxiv.org/pdf/1402.3334.pdf) by McLachlan et al, with the first step being an Euler method integration and later using the partial values from said step as the implicit side of the LLG equation for the spherical midpoint scheme. A time step of ~10fs was used, with the choice of timestep being done so as to always maintain an adimensional/computational time stepping of 0.01 or less for the Euler method.

## Results

The results are shown in figures containing both graphics of the time evolution of the spatially averaged Magnetization My/Ms, as well as images of the reversal mechanism (i.e when the value of the expected mx vector plot first crosses zero).

Looking at the time series data, we can see the method is quite accurate for the case of a 170°-to-x-axis applied field, considering how simple the implementation is. It matches precisely the switching patterns of other solutions (D'Aquino's being
showcased as example), with a difference only in frequency when reaching the equilibrium state.

The reason for this can be seen in the plot of the reversal mechanism (reversal_1). When compared to the [same vector field plot](https://www.ctcms.nist.gov/~rdm/std4/dAquino.html) by D'Aquino, note how the latter solution is fully symmetric with respect to the y-axis, i.e, any given slice of the film of the same y value points more or less towards the same directions. In our case however, there is a clear "circular" transition in the y-axis from the bottom, then to the center, and finally to the top of the plot, even though the reversal pattern appears to be the same. This skewed reversal mechanism in relation to other methods is likely the reason for the shift in frequency of reversals, and might be caused by the assumed dipole-dipole coupling of cells, which is far stronger in boundaries than deeper into the sample's bulk.

This is even more evident in reversal_2 for the case of a 190° applied field. In that case the reversal mechanism is explicitly circular around the center of the structure (even though the x-directions fully match the expected behavior by other solutions). This first skewed reversal mechanisms cause swirling patterns throughout the structure that further delay its subsequent oscillations toward equilibrium. Note however (in time_series2_2 file) that the method does converge the structure into the same state as that of others, with the distortion on reversal mechanisms not being enough to change the system's path through its energy landscape. 
