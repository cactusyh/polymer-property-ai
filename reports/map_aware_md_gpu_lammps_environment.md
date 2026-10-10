# Map-Aware MD GPU LAMMPS Environment

Date: 2026-09-14
Conda environment: mace-lammps
CONDA_PREFIX: /inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/mace-lammps

## Executable and version check

which lmp: /inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/lammps/lammps/build-mliap/lmp

Large-scale Atomic/Molecular Massively Parallel Simulator - 10 Sep 2025 - Development
Git info (develop / patch_10Sep2025-244-g36f7c81ced)

Usage example: lmp -var t 300 -echo screen -in in.alloy

List of command-line options supported by this LAMMPS executable:

-echo none/screen/log/both  : echoing of input script (-e)
-help                       : print this help message (-h)
-in none/filename           : read input from file or stdin (default) (-i)
-kokkos on/off ...          : turn KOKKOS mode on or off (-k)
-log none/filename          : where to send log output (-l)
-mdi '<mdi flags>'          : pass flags to the MolSSI Driver Interface
-mpicolor color             : which exe in a multi-exe mpirun cmd (-m)
-cite                       : select citation reminder style (-c)
-nocite                     : disable citation reminder (-nc)
-nonbuf                     : disable screen/logfile buffering (-nb)
-package style ...          : invoke package command (-pk)
-partition size1 size2 ...  : assign partition sizes (-p)
-plog basename              : basename for partition logs (-pl)
-pscreen basename           : basename for partition screens (-ps)
-restart2data rfile dfile ... : convert restart to data file (-r2data)
-restart2dump rfile dgroup dstyle dfile ... 
                            : convert restart to dump file (-r2dump)
-restart2info rfile         : print info about restart rfile (-r2info)
-reorder topology-specs     : processor reordering (-r)
-screen none/filename       : where to send screen output (-sc)
-skiprun                    : skip loops in run and minimize (-sr)
-suffix gpu/intel/kk/opt/omp: style suffix to apply (-sf)
-var varname value          : set index style variable (-v)

OS: Linux "Ubuntu 24.04.1 LTS" 5.15.0-41-generic x86_64

Compiler: GNU C++ 13.3.0 with OpenMP 4.5
C++ standard: C++17
Embedded fmt library version: 10.2.0
Embedded JSON class version: 3.12.0

MPI v3.1: Open MPI v4.1.7rc1, package: Open MPI root@sharp-ci-01 Distribution, ident: 4.1.7rc1, repo rev: v4.1.5-176-g6d9519e4c3, Unreleased developer copy

Accelerator configuration:

KOKKOS package API: CUDA Serial
KOKKOS package precision: double
KOKKOS package view layout: legacy
Kokkos library version: 4.7.1

FFT information:

FFT precision  = double
FFT engine  = mpiFFT
FFT library = KISS
KOKKOS FFT engine  = mpiFFT
KOKKOS FFT library = cuFFT

Active compile time flags:

-DLAMMPS_GZIP
-DLAMMPS_PNG
-DLAMMPS_JPEG
-DLAMMPS_SMALLBIG
sizeof(smallint): 32-bit
sizeof(imageint): 32-bit
sizeof(tagint):   32-bit
sizeof(bigint):   64-bit

Available compression formats:

Extension: .gz     Command: gzip
Extension: .bz2    Command: bzip2
Extension: .zst    Command: zstd
Extension: .xz     Command: xz
Extension: .lzma   Command: xz


Installed packages:

CLASS2 EXTRA-MOLECULE KOKKOS KSPACE MC ML-IAP ML-SNAP MOLECULE PYTHON RIGID 


## GPU visibility

Mon Sep 14 08:19:08 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.144.03             Driver Version: 550.144.03     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H800                    On  |   00000000:18:00.0 Off |                    0 |
| N/A   33C    P0             72W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

## LAMMPS Python binding
LAMMPS Python import OK

## Required environment variables
PYTHONPATH=/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/lammps/lammps/python:
LD_LIBRARY_PATH=/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/mace-lammps/lib:/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
PATH=/root/.codex/tmp/arg0/codex-arg0WsW2JW:/root/.nvm/versions/node/v24.14.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/path:/root/.nvm/versions/node/v24.14.0/bin:/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/mace-lammps/bin:/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/condabin:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin:/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/lammps/lammps/build-mliap
CUDA_VISIBLE_DEVICES=0

## Exact GPU command
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in INPUT_FILE

## Packages / KOKKOS evidence
-kokkos on/off ...          : turn KOKKOS mode on or off (-k)
-package style ...          : invoke package command (-pk)
-suffix gpu/intel/kk/opt/omp: style suffix to apply (-sf)
Embedded fmt library version: 10.2.0
Embedded JSON class version: 3.12.0
MPI v3.1: Open MPI v4.1.7rc1, package: Open MPI root@sharp-ci-01 Distribution, ident: 4.1.7rc1, repo rev: v4.1.5-176-g6d9519e4c3, Unreleased developer copy
KOKKOS package API: CUDA Serial
KOKKOS package precision: double
KOKKOS package view layout: legacy
Kokkos library version: 4.7.1
KOKKOS FFT engine  = mpiFFT
KOKKOS FFT library = cuFFT
Installed packages:
CLASS2 EXTRA-MOLECULE KOKKOS KSPACE MC ML-IAP ML-SNAP MOLECULE PYTHON RIGID 
angle           angle/kk        atomic          atomic/kk       body            
bond            bond/kk         charge          charge/kk       ellipsoid       
full            full/kk         hybrid          hybrid/kk       line            
molecular       molecular/kk    sphere          sphere/kk       template        
respa           verlet          verlet/kk       
cg              cg/kk           fire/old        fire            hftn            
buck/coul/cut/kk                buck/coul/long  buck/coul/long/kk               
buck/coul/msm   buck/kk         buck/long/coul/long             coul/cut        
coul/cut/kk     coul/debye      coul/debye/kk   coul/dsf        coul/dsf/kk     
coul/long       coul/long/kk    coul/msm        coul/streitz    coul/wolf       
coul/wolf/kk    meam/c          reax            reax/c          mesont/tpm      
hybrid          hybrid/omp      hybrid/kk       hybrid/molecular                
hybrid/overlay/kk               hybrid/scaled   hybrid/scaled/omp               
lj/charmm/coul/charmm/implicit/kk               lj/charmm/coul/charmm/kk        
lj/charmm/coul/long             lj/charmm/coul/long/kk          
lj/charmmfsw/coul/long          lj/charmmfsw/coul/long/kk       lj/class2       
lj/class2/coul/cut              lj/class2/coul/cut/kk           
lj/class2/coul/long             lj/class2/coul/long/kk          lj/class2/kk    
lj/cut          lj/cut/coul/cut lj/cut/coul/cut/kk              lj/cut/coul/long                
lj/cut/coul/long/kk             lj/cut/coul/msm lj/cut/kk       lj/cut/tip4p/cut                
lj/cut/tip4p/long               lj/expand       lj/expand/kk    
mliap/kk        morse           morse/kk        python          snap            
snap/kk         soft            soft/kk         table           table/kk        
tip4p/cut       tip4p/long      yukawa          yukawa/kk       zbl             
zbl/kk          zero            
class2          class2/kk       fene            fene/expand     fene/kk         
fene/nm         gaussian        gromos          harmonic        harmonic/kk     
hybrid          hybrid/kk       morse           nonlinear       quartic         
charmm          charmm/kk       class2          class2/kk       cosine          
cosine/delta    cosine/kk       cosine/periodic cosine/shift    cosine/shift/exp                
gaussian        harmonic        harmonic/kk     hybrid          hybrid/kk       
charmm          charmm/kk       charmmfsw       charmmfsw/kk    class2          
class2/kk       cosine/shift/exp                cosine/squared/restricted       
fourier         harmonic        harmonic/kk     helix           hybrid          
hybrid/kk       multi/harmonic  multi/harmonic/kk               nharmonic       
opls            opls/kk         quadratic       spherical       table           
class2          class2/kk       cossq           cvff            distance        
fourier         harmonic        harmonic/kk     hybrid          hybrid/kk       
pppm/disp       pppm/disp/tip4p pppm/kk         pppm/stagger    pppm/tip4p      
charge/regulation               cmap            cmap/kk         deform          
deform/kk       deposit         ave/spatial     ave/spatial/sphere              
dt/reset        dt/reset/kk     efield          efield/kk       ehex            
enforce2d       enforce2d/kk    evaporate       external        gcmc            
gravity         gravity/kk      halt            heat            hmc             
indent          langevin        langevin/kk     lineforce       mol/swap        
momentum        momentum/kk     move            nph             nph/kk          
