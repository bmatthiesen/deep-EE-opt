/* Copyright (C) 2018 Bho Matthiesen
 * 
 * This program is used in the article:
 * 
 * Bho Matthiesen, Alessio Zappone, Eduard A. Jorswieck, and Merouane Debbah,
 * "Deep Learning for Optimal Energy-Efficient Power Control in Wireless
 * Interference Networks," submitted to IEEE Journal on Selected Areas in
 * Communication.
 * 
 * License:
 * This program is licensed under the GPLv2 license. If you in any way use this
 * code for research that results in publications, please cite our original
 * article listed above.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details. */

#include "wsee_lambert.h"

int
main(void)
{
	WSEE<4> tin;

	const double Pmax = 1e-3; // objval = 2.9926550989520004
	//const double Pmax = 1e2; // objval = 10.374425740825233

	// cidx = 1 from results.h5
	const double beta[][4] = {
		{8.3401758e+02, 5.9968562e+00, 9.5184622e+00, 6.0737956e-01},
		{1.3587096e+00, 3.9182301e+01, 2.0014184e-02, 1.6249435e+00},
		{3.8521406e-01, 4.6761915e-01, 8.7457578e+03, 1.8704400e+00},
		{1.2729254e-01, 2.1447293e-02, 3.1017335e-02, 1.2471862e+02}
	};
	const double mu_all = 4;
	const double psi_all = 1;
	
	
	tin.setUB(Pmax);
	tin.setLB(0);
	tin.setPrecision(1e-4);
	tin.useRelTol = true;

	for (size_t i = 0; i < tin.dim(); ++i)
	{
		tin.mu[i] = mu_all;
		tin.psi[i] = psi_all;
		tin.alpha[i] = beta[i][i];
		for (size_t j = 0; j < tin.dim(); ++j)
		{
			if (i == j)
				tin.beta[i][i] = 0.0;
			else
				tin.beta[i][j] = beta[i][j];
		}
	}

	tin.optimize();

	/* Pmax = 1e2
	 *   eps = 1e-2: 0.02 sec, 1311 (718) iter
	 *   eps = 1e-4: 61 sec, 5511124 (2987686) iter
	 */
}
