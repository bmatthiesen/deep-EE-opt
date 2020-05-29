A Globally Optimal Energy-Efficient Power Control Framework and its Efficient Implementation in Wireless Interference Networks
==================

This code package is related to the following scientific article:

Bho Matthiesen, Alessio Zappone, Karl-L. Besser, Eduard A. Jorswieck, and Merouane Debbah, "[A Globally Optimal Energy-Efficient Power Control Framework and its Efficient Implementation in Wireless Interference Networks](https://arxiv.org/abs/1812.06920)," submitted to IEEE Transactions on Signal Processing

## Contents

Randomly generated channels and global optimal power allocation results. `wsee*-processed.h5` contains also first-order optimal power allocations for the test set. Please use `src/DL/prepareDset.py` to generate the training data from these files.

List of files:

* `channels-4.h5`, `channels-7.h5`: Channels used for training and validation for 4 and 7 users, respectively
* `wsee4-processed.h5`, `wsee7-processed.h5`: Global optimal power allocations for `channels.h5`. The training data is generated from this file
* `dset4.h5`, `dset7.h5`: Training, Validation, and Test Data
* `channels-hataUrban.h5`: Channels with Hata-COST231 Urban path loss and log-normal shadowing
* `results_hataUrban.h5`: Global optimal power allocations for `channels-hataUrban.h5`
* `channels-hataUrban-noSF.h5`: Channels with Hata-COST231 Urban path loss (no shaodwing)
* `results_hataUrban_noSF.h5`: Global optimal power allocations for `channels-hataUrban-noSF.h5`


## Acknowledgements

This research was supported in part by the Deutsche Forschungsgemeinschaft (DFG) in the [Collaborative Research Center 912 "Highly Adaptive Energy-Efficient Computing."](https://tu-dresden.de/ing/forschung/sfb912) and under grant JO 801/23-1, by the European Commission through the H2020-MSCA IF-BESMART project under Grant Agreement 749336, by MIUR under the "PRIN Liquid_Edge" contract, and by the Italian Ministry of Education and Research under the program "Dipartimenti di Eccellenza 2018-2022".

We thank the Center for Information Services and High Performance Computing (ZIH) at TU Dresden for generous allocations of computer time.


## License and Referencing

This program is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

