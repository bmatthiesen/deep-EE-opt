A Globally Optimal Energy-Efficient Power Control Framework and its Efficient Implementation in Wireless Interference Networks
==================

This code package is related to the following scientific article:

Bho Matthiesen, Alessio Zappone, Karl-L. Besser, Eduard A. Jorswieck, and Merouane Debbah, "[A Globally Optimal Energy-Efficient Power Control Framework and its Efficient Implementation in Wireless Interference Networks](https://arxiv.org/abs/1812.06920)," accepted for publication in IEEE Transactions on Signal Processing.

## Contents

This folder contains the complete source code necessary to reproduce the results in the paper. In particular, the subdirectories should be visited in the following order:

1. `channel`: Channel generation
2. `globalOpt`: Global optimization code. Generates the data for the training set.
3. `SCA`: The first-order optimal power allocation algorithm used as baseline performance measure.
4. `DL`: The deep artificial neural network, helper scripts, and evaluation code.

With the data supplied in `../data/`, it is also possible to directly skip to `DL`.

## Acknowledgements

This research was supported in part by the Deutsche Forschungsgemeinschaft (DFG) in the [Collaborative Research Center 912 "Highly Adaptive Energy-Efficient Computing."](https://tu-dresden.de/ing/forschung/sfb912) and under grant JO 801/23-1, by the European Commission through the H2020-MSCA IF-BESMART project under Grant Agreement 749336, by MIUR under the "PRIN Liquid_Edge" contract, and by the Italian Ministry of Education and Research under the program "Dipartimenti di Eccellenza 2018-2022".

We thank the Center for Information Services and High Performance Computing (ZIH) at TU Dresden for generous allocations of computer time.


## License and Referencing

This program is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

