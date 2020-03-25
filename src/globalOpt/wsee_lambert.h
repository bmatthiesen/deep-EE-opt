/* Copyright (C) 2018-2019 Bho Matthiesen, Karl-Ludwig Besser
 * 
 * This program is used in the article:
 * 
 * Bho Matthiesen, Alessio Zappone, Karl-L. Besser, Eduard A. Jorswieck, and
 * Merouane Debbah, "A Globally Optimal Energy-Efficient Power Control Framework
 * and its Efficient Implementation in Wireless Interference Networks,"
 * submitted to IEEE Transactions on Signal Processing
 * 
 * License:
 * This program is licensed under the GPLv2 license. If you in any way use this
 * code for research that results in publications, please cite our original
 * article listed above.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#ifndef _WSEE_LAMBERT_H_
#define _WSEE_LAMBERT_H_

#include <stdexcept>

#include "BRB.h"

extern "C" {
	#include "mkl.h"
}

#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/constants/constants.hpp>

constexpr long double exp_minus_one = (static_cast<long double>(1) / boost::math::long_double_constants::e);

template <size_t Dim>
class WSEE : public BRB<Dim>
{
	public:
		using typename BRB<Dim>::vtype;
		using typename BRB<Dim>::RBox;
		using typename BRB<Dim>::PBox;

		double mu[Dim] __attribute__((aligned(64)));
		double psi[Dim] __attribute__((aligned(64)));
		double alpha[Dim] __attribute__((aligned(64)));
		double beta[Dim][Dim] __attribute__((aligned(64)));

		WSEE() : BRB<Dim>() { }

		double objective(const vtype& x) const
			{ return WSEEobj(x); }

	private:
		double WSEEobj(const vtype& x) const;

		void bound(RBox& r) const override final;

		const vtype& feasiblePoint(const RBox& r) const override final
			{	return r.lb(); }

		double obj(const RBox& r) const override final
			{ const vtype& p = feasiblePoint(r); return WSEEobj(p); }

		double obj(const vtype& x) const override final
			{ return WSEEobj(x); }

		bool feasible(const RBox&) const override final
			{ return true; }

		bool isEmpty(const PBox& r) const override final
			{ return false; }
};


template <size_t D>
void
WSEE<D>::bound(RBox& r) const
{
	vtype z __attribute__((aligned(64)));
	vtype tmp __attribute__((aligned(64)));

	for (size_t i = 0; i < D; ++i)
	{
		double denom = 1.0;

		for (size_t j = 0; j < D; ++j)
		{
			if (j != i)
				denom += r.lb(j) * beta[i][j];
		}

		z[i] = alpha[i] / denom; // alpha tilde
		tmp[i] = z[i] / mu[i] * psi[i] - 1.0;
	}

	for (size_t i = 0; i < D; ++i)
	{
		double lW = boost::math::lambert_w(exp_minus_one * static_cast<long double>(tmp[i]));
		r.xk(i) = (tmp[i] / lW - 1.0) / z[i];
	}

	/* clip to current box */
	for (size_t i = 0; i < D; ++i)
	{
		if (r.xk(i) > r.ub(i))
			r.xk(i) = r.ub(i);
		else if (r.xk(i) < r.lb(i))
			r.xk(i) = r.lb(i);
	}

	/* calculate bound */
	for (size_t i = 0; i < D; ++i)
		z[i] = 1 + z[i] * r.xk(i);

	vdLn(D, &z[0], &z[0]);

	r.bound = 0;
	for (size_t i = 0; i < D; ++i)
		r.bound += z[i] / (mu[i]*r.xk(i) + psi[i]);
}


template <size_t D>
double
WSEE<D>::WSEEobj(const vtype& x) const
{
	double ret = 0;
	vtype z __attribute__((aligned(64)));

	for (size_t i = 0; i < D; ++i)
	{
		double denom = 1.0;

		for (size_t j = 0; j < D; ++j)
		{
			if (j != i)
				denom += x[j] * beta[i][j];
		}

		z[i] = 1 + alpha[i] * x[i] / denom;
	}

	vdLn(D, &z[0], &z[0]);

	for (size_t i = 0; i < D; ++i)
		ret += z[i] / (mu[i]*x[i] + psi[i]);

	return ret;
}

#endif
