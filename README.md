Deep Learning for Optimal Energy-Efficient Power Control in Wireless Interference Networks
==================

This code package is related to the following scientific article:

Bho Matthiesen, Alessio Zappone, Eduard A. Jorswieck, and Merouane Debbah, "Deep Learning for Optimal Energy-Efficient Power Control in Wireless Interference Networks," submitted to IEEE Journal on Selected Areas in Communication.


## Abstract of Article

This work develops a deep learning power control framework for energy efficiency maximization in wireless interference networks. Rather than relying on suboptimal power allocation policies, the training of the deep neural network is based on the globally optimal power allocation rule, leveraging a newly proposed branch-and-bound procedure with a complexity affordable for the offline generation of large training sets. In addition, no initial power vector is required as input of the proposed neural network architecture, which further reduces the overall complexity. As a benchmark, we also develop a first-order optimal power allocation algorithm. Numerical results show that the neural network solution is virtually optimal, outperforming the more complex first-order optimal method, while requiring an extremely small online complexity. 


## Requirements & Contents of the Code Package

This code was compiled and tested with Python 3.6.6, [Cython](https://cython.org/) 0.29, [Keras](http://keras.io) 2.2.4, [TensorFlow](https://www.tensorflow.org/) 1.12.0, [Intel MKL](https://software.intel.com/mkl) 2017, [GNU Make](https://www.gnu.org/software/make/) 3.82, and [GCC](https://www.gnu.org/software/gcc/) 7.3.

The complete source code is available in `src/`. Prior to compilation, please update the variables in the Makefiles and `setup.py` according to your needs. More detailed instructions are found in the README files in the individual directories.


## Acknowledgements

The research of Bho Matthiesen and Eduard A. Jorswieck was supported in part by the Deutsche Forschungsgemeinschaft (DFG) in the [Collaborative Research Center 912 "Highly Adaptive Energy-Efficient Computing."](https://tu-dresden.de/ing/forschung/sfb912)

The work of A. Zappone and M. Debbah was funded by the European Commission through the H2020-MSCA IF-BESMART project under Grant Agreement 749336.

We thank the Center for Information Services and High Performance Computing (ZIH) at TU Dresden for generous allocations of computer time.


## License and Referencing

This program is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

