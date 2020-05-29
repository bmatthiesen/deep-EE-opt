A Globally Optimal Energy-Efficient Power Control Framework and its Efficient Implementation in Wireless Interference Networks
==================

This code package is related to the following scientific article:

Bho Matthiesen, Alessio Zappone, Karl-L. Besser, Eduard A. Jorswieck, and Merouane Debbah, "[A Globally Optimal Energy-Efficient Power Control Framework and its Efficient Implementation in Wireless Interference Networks](https://arxiv.org/abs/1812.06920)," accepted for publication in IEEE Transactions on Signal Processing.

## Contents

The branch-and-bound algorithm to generate the training, validation, and test sets. Short description of files:

* `wsee_lambert.h`: Implementation of the bounding procedure
* `BRB.h`, `util.h`, `util.cpp`: Branch-and-bound algorithm and auxiliary files
* `test_lambert.cpp`: A simple test program for wsee_lambert.h
* `test_wsee.py`: Test script for the Python module
* `wseePy.pyx.m4`: m4 File to Generate the Cython module to interface wsee_lambert.h
* `run_wsee.py`: The script that actually generates the training data
* `slurm/`: `run_wsee.py` and some other scripts are written to run on a HPC system with slurm workload manager. `sbatch.sh` starts several parallel instances of `run_wsee.py`, `sbatch_collect.sh` collects the results in a single file.
* `collect.py`: Collectes results from parallel `run_wsee.py` instances in a single file.
* `setup.py`: Build python module

## Requirements

This code was compiled and tested with Python 3.6.6, [Cython](https://cython.org/) 0.29, [Intel MKL](https://software.intel.com/mkl) 2017, [GNU Make](https://www.gnu.org/software/make/) 3.82, and [GCC](https://www.gnu.org/software/gcc/) 7.3.

In particular, the C++ code is written in C++17 and employs Intel MKL and OpenMP. Intel MKL can be replaced easily if desired. It should only be used to compute logarithms in parallel. Compiling the Cython module requires Cython. We recommend to use the [Anaconda](https://www.anaconda.com/) Python distribution.

## Building & Running

The following steps are in order to get the Python module running:

1. Adjust paths in `Makefile`
2. Run `make util.o`
3. Adjust paths in `setup.py`
4. Run `python3 setup.py build`
5. Create a link to the newly built module, e.g., `ln -s build/lib.linux-x86_64-3.6/wseePy.cpython-36m-x86_64-linux-gnu.so`
6. Test by running `test_wsee.py`
7. Adjust `run_wsee.py` and run it

For debugging purposes it might be desired to first build `test_lambert.cpp` without involving Python. Do so by running `make` without arguments. Be sure to adjust `Makefile` beforehand.

## Acknowledgements

This research was supported in part by the Deutsche Forschungsgemeinschaft (DFG) in the [Collaborative Research Center 912 "Highly Adaptive Energy-Efficient Computing."](https://tu-dresden.de/ing/forschung/sfb912) and under grant JO 801/23-1, by the European Commission through the H2020-MSCA IF-BESMART project under Grant Agreement 749336, by MIUR under the "PRIN Liquid_Edge" contract, and by the Italian Ministry of Education and Research under the program "Dipartimenti di Eccellenza 2018-2022".

We thank the Center for Information Services and High Performance Computing (ZIH) at TU Dresden for generous allocations of computer time.


## License and Referencing

This program is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

