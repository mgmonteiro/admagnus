# admagnus
Micromagnetics simulation for CUDA processors

Version: Sample/Template

Author: Maxwel Gama

Contact: maxweljr@gmail.com

# What this is

This is a simple but efficient code for computing the dynamics of ferromagnets using the Landau-Lifshitz-Gilbert equation. It is meant to be helpful in introducing people to coding for CUDA parallel GPUs. In this manner, it completely does away with the usual abstractions and is quite verbose, presenting functions and routines in the intuitive way you would expect from a scientist getting into coding. There is some code which could be reused across the board for this reason, but it does no harm to performance. It is  written in straight C-CUDA with its default libraries only, rendering it suitable for use in learning with [textbooks](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf) and very easy to handle with all kinds of IDEs and even without any. It is fully used through the console terminal to be friendly to general ssh, cloud computing and cluster usage.

# What this is not

This is not a substitute for the current state of the art parallel processing software of the same nature, mumax³. It uses a first order approximation in every field which renders the corresponding Free Energy function equivalent to a Hamiltonian of similar functional form to the atomic case, given that the discretization is cubic-cell based. There are some [publications](https://www.sciencedirect.com/science/article/pii/S0304885317315032) which successfully solve problems in this formulation, which is mostly valid for a close-to-equilibrium regime. Nevertheless, this code also implements the Magnetostatic field with no approximations whatsoever other than the loss of some geometric details, which is also a feature in more advanced codes anyway, i.e it uses a N-Body parallel algorithm to iterate over the N^2 cell pairs. 

It will actually provide rather good approximations to many realistic cases and perform quite well on medium sized systems in the order of 10^2 nm, depending on the choice of material and discretization. But its region of "production level worthy" as a code is smaller than that of mumax, with the advantage that it is far easier to read and manipulate directly when studying equilibrium cases and small perturbations (e.g proposing new equivalent energy models to certain systems with magnetostatic and anisotropic interactions, etc.). This is a learning tool that can be used in real investigative cases, and one that is easier to expand into because of its first order equivalence to the discrete cases.

# How to compile

You need a CUDA GPU with compute capability higher to or equal to 2.0, As well as NVIDIA's proprietary drivers. You may contact maxweljr@gmail.com for troubleshooting but the [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) is usually enough. Be aware that your *driver* version must match your *compiler* version, as the video drivers required for running libraries are not the same as the CUDA compiler and runtime API. To be sure they are always the same version install only through the **runfile (local)** option in the [CUDA download page](https://developer.nvidia.com/cuda-downloads).

With a proper driver installation, all you need to compile is reach the folder containing the admagnus_template.cu source file and use the following command,

nvcc -o Admagnus.x admagnus_template.cu -lm -lcurand -D_FORCE_INLINES -O3

Where the -lm, -lcurand and -D_FORCE_INLINES flag are usually required by Ubuntu and other OSs in order to find the CUDA libraries (you can probably use only optimization flag -O3 in Windows if correctly installed). For more advanced options consider also compiling for the specific GPU architecture with -arch=sm_XX and also tracking register usage through the --ptxas-options=-v flag.

# How to input

The program expects two input files in any given directory where it will be run, a **coord_z.xyz** file containing the values of cell positions and cell magnetizations (in the format programs such as VMD and Jmol can promptly read), as well as a **settings.inp** file containing the simulation parameters such as cell size, time step and type of integration method. A template settings.inp is provided with all possible options covered.

To generate input coord_z.xyz files a handy C file (also verbose enough to be didactic) is provided as spin_lego, compile it simply with

g++ -o spin.x Spin_Lego_vs1.0.cpp 

and use the blueprint.inp file to create the desired cell mesh. For the sake of simplicity this program only uses cubic cell orderings, and only a few types of initial magnetization fields are available. It is trivial however to manually add a specific field directly into the C script in order to create your own initial states for simulation. The .xyz format is also versatile and easily convertible to be read by other programs, including mumax.

# How to use

With a given settings.inp and coord_z.xyz files, simply hit ./Admagnus.x on the terminal or, if it is located in a specific directory, use that address. For example, ./../dynamics/Admagnus.x if you are using the same folder structure as this git branch. Alternatively use [nohup](https://linux.101hacks.com/unix/nohup-command/) for longer simulation runs.

# How to output

After running, several output files will be created each containing different data from the simulation. In sum, the following data is extracted by default from simulations (at regular intervals given by settings.inp):

-Energy of structure (Exchange, Zeeman, Anisotropy, Magnetostatic, DMI and total)

-Reduced Magnetization components M/Ms

-Individual cell magnetization profiles in .xyz files

These can be used to plot expected values and so on.

# The model

This code models the micromagnetic Free Energy as a first order equivalent Hamiltonian much in the same way the transition from the Heisenberg model to the usual micromagnetic integrals are done. The main difference is that in this case, not only the local values such as anisotropy and exchange energy use the "6-neighbor dot product" approximation, but also the magnetostatic interaction and any other possible energy is held to first order. 

In the magnetostatic case this means we are assuming the cells interact through the regular dipole-dipole term, whereas for the DMI coupling we have the usual superexchange mechanism, and the same for other energy values. The main advantage of doing this is that, once the coupling constants have been determined, the boundary treatment is done through direct application of the fields derived from the Hamiltonian. A detailed explanation can be found in several published texts such as [this one](https://www.sciencedirect.com/science/article/abs/pii/S0304885317313148). 

Because the are no other approximations done to the magnetostatic field interactions, this code can also be used to benchmark against the use of the full N^2 interaction in comparison to other methods, as well as ascertain the influence of the magnetostatic field in certain problems with non trivial boundaries (e.g when modelling curvilinear systems with a Finite Difference Mesh).



