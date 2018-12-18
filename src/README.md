Deep Learning for Optimal Energy-Efficient Power Control in Wireless Interference Networks
==================

This code package is related to the following scientific article:

Bho Matthiesen, Alessio Zappone, Eduard A. Jorswieck, and Merouane Debbah, "[Deep Learning for Optimal Energy-Efficient Power Control in Wireless Interference Networks](https://arxiv.org/abs/1812.06920)," submitted to IEEE Journal on Selected Areas in Communication.

## Contents

This folder contains the complete source code necessary to reproduce the results in the paper. In particular, the subdirectories should be visited in the following order:

1. `channel`: Channel generation
2. `globalOpt`: Global optimization code. Generates the data for the training set.
3. `SCA`: The first-order optimal power allocation algorithm used as baseline performance measure.
4. `DL`: The deep artificial neural network, helper scripts, and evaluation code.

With the data supplied in `../data/`, it is also possible to directly skip to `DL`.

## Acknowledgements

The research of Bho Matthiesen and Eduard A. Jorswieck was supported in part by the Deutsche Forschungsgemeinschaft (DFG) in the [Collaborative Research Center 912 "Highly Adaptive Energy-Efficient Computing."](https://tu-dresden.de/ing/forschung/sfb912)

The work of A. Zappone and M. Debbah was funded by the European Commission through the H2020-MSCA IF-BESMART project under Grant Agreement 749336.

We thank the Center for Information Services and High Performance Computing (ZIH) at TU Dresden for generous allocations of computer time.


## License and Referencing

This program is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

