A Globally Optimal Energy-Efficient Power Control Framework and its Efficient Implementation in Wireless Interference Networks
==================

This code package is related to the following scientific article:

Bho Matthiesen, Alessio Zappone, Karl-L. Besser, Eduard A. Jorswieck, and Merouane Debbah, "[A Globally Optimal Energy-Efficient Power Control Framework and its Efficient Implementation in Wireless Interference Networks](https://arxiv.org/abs/1812.06920)," submitted to IEEE Transactions on Signal Processing


## Abstract of Article

This work develops a novel power control framework for energy-efficient power control in wireless networks. The proposed method is a new branch-and-bound procedure based on problem-specific bounds for energy-efficiency maximization that allow for faster convergence. This enables to find the global solution for all of the most common energy-efficient power control problems with a complexity that, although still exponential in the number of variables, is much lower than other available global optimization frameworks. Moreover, the reduced complexity of the proposed framework allows its practical implementation through the use of deep neural networks. Specifically, thanks to its reduced complexity, the proposed method can be used to train an artificial neural network to predict the optimal resource allocation. This is in contrast with other power control methods based on deep learning, which train the neural network based on suboptimal power allocations due to the large complexity that generating large training sets of optimal power allocations would have with available global optimization methods. As a benchmark, we also develop a novel first-order optimal power allocation algorithm. Numerical results show that a neural network can be trained to predict the optimal power allocation policy.

## Requirements & Contents of the Code Package

This code was compiled and tested with Python 3.6.6, [Cython](https://cython.org/) 0.29, [Keras](http://keras.io) 2.2.4, [TensorFlow](https://www.tensorflow.org/) 1.12.0, [Intel MKL](https://software.intel.com/mkl) 2017, [GNU Make](https://www.gnu.org/software/make/) 3.82, and [GCC](https://www.gnu.org/software/gcc/) 7.3.

The complete source code is available in `src/`. Prior to compilation, please update the variables in the Makefiles and `setup.py` according to your needs. More detailed instructions are found in the README files in the individual directories.


## Acknowledgements

This research was supported in part by the Deutsche Forschungsgemeinschaft (DFG) in the [Collaborative Research Center 912 "Highly Adaptive Energy-Efficient Computing."](https://tu-dresden.de/ing/forschung/sfb912) and under grant JO 801/23-1, by the European Commission through the H2020-MSCA IF-BESMART project under Grant Agreement 749336, by MIUR under the "PRIN Liquid_Edge" contract, and by the Italian Ministry of Education and Research under the program "Dipartimenti di Eccellenza 2018-2022".

We thank the Center for Information Services and High Performance Computing (ZIH) at TU Dresden for generous allocations of computer time.


## License and Referencing

This program is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

