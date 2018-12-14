Deep Learning for Optimal Energy-Efficient Power Control in Wireless Interference Networks
==================

This code package is related to the following scientific article:

Bho Matthiesen, Alessio Zappone, Eduard A. Jorswieck, and Merouane Debbah, "Deep Learning for Optimal Energy-Efficient Power Control in Wireless Interference Networks," submitted to IEEE Journal on Selected Areas in Communication.

## Contents

Randomly generated channels and global optimal power allocation results. `results.h5` contains also first-order optimal power allocations for the test set. Please use `code/DL/prepareDset.py` to generate the training data from these files.

List of files:

* `channels.h5`: Channels used for training and validation
* `results.h5`: Global optimal power allocations for `channels.h5`. The training data is generated from this file
* `channels-hataUrban.h5`: Channels with Hata-COST231 Urban path loss and log-normal shadowing
* `results_hataUrban.h5`: Global optimal power allocations for `channels-hataUrban.h5`
* `channels-hataUrban-noSF.h5`: Channels with Hata-COST231 Urban path loss (no shaodwing)
* `results_hataUrban_noSF.h5`: Global optimal power allocations for `channels-hataUrban-noSF.h5`


## Acknowledgements

The research of Bho Matthiesen and Eduard A. Jorswieck was supported in part by the Deutsche Forschungsgemeinschaft (DFG) in the [Collaborative Research Center 912 "Highly Adaptive Energy-Efficient Computing."](https://tu-dresden.de/ing/forschung/sfb912)

The work of A. Zappone and M. Debbah was funded by the European Commission through the H2020-MSCA IF-BESMART project under Grant Agreement 749336.

We thank the Center for Information Services and High Performance Computing (ZIH) at TU Dresden for generous allocations of computer time.


## License and Referencing

This program is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

