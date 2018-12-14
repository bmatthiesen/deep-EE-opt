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
	    using BRB<Dim>::disableReduction;

		double mu[Dim] __attribute__((aligned(64)));
		double psi[Dim] __attribute__((aligned(64)));
		double alpha[Dim] __attribute__((aligned(64)));
		double beta[Dim][Dim] __attribute__((aligned(64)));

		WSEE() : BRB<Dim>() { disableReduction = true; }

		double objective(const vtype& x) const
			{ return WSEEobj(x); }

	private:
		double WSEEobj(const vtype& x) const;

		double bound(const vtype& lb, const vtype& ub, vtype& pk) const;
		double bound(const vtype& lb, const vtype& ub) const override final
			{ vtype xk; return bound(lb, ub, xk); }
		void bound(RBox& r) const override final
			{ r.bound = bound(r.lb(), r.ub(), r.xk()); }

		double red_alpha(const size_t, const double, const vtype&, const vtype&) const override { throw std::runtime_error("reduction not implemented"); };
		double red_beta(const size_t, const double, const vtype&, const vtype&) const override { throw std::runtime_error("reduction not implemented"); };

		const vtype& feasiblePoint(const RBox& r) const override final
			{	return r.lb(); }

		double obj(const RBox& r) const override final
			{ const vtype& p = feasiblePoint(r); return WSEEobj(p); }

		bool feasible(const RBox&) const override final
			{ return true; }

		bool isEmpty(const PBox& r) const override final
			{ return false; }
};


template <size_t D>
double
WSEE<D>::bound(const vtype& lb, const vtype& ub, vtype& p) const
{
	vtype z __attribute__((aligned(64)));
	vtype tmp __attribute__((aligned(64)));

	for (size_t i = 0; i < D; ++i)
	{
		double denom = 1.0;

		for (size_t j = 0; j < D; ++j)
		{
			if (j != i)
				denom += lb[j] * beta[i][j];
		}

		z[i] = alpha[i] / denom; // alpha tilde
		tmp[i] = z[i] / mu[i] * psi[i] - 1.0;
	}

	#pragma omp parallel for
	for (size_t i = 0; i < D; ++i)
	{
		double lW = boost::math::lambert_w(exp_minus_one * static_cast<long double>(tmp[i]));
		p[i] = (tmp[i] / lW - 1.0) / z[i];
	}

	/* clip to current box */
	for (size_t i = 0; i < D; ++i)
	{
		if (p[i] > ub[i])
			p[i] = ub[i];
		else if (p[i] < lb[i])
			p[i] = lb[i];
	}

	/* calculate bound */
	for (size_t i = 0; i < D; ++i)
		z[i] = 1 + z[i] * p[i];

	vdLn(D, &z[0], &z[0]);

	double ret = 0;
	for (size_t i = 0; i < D; ++i)
		ret += z[i] / (mu[i]*p[i] + psi[i]);

	return ret;
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
