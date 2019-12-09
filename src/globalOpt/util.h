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

#ifndef _DEBUG_H
#define _DEBUG_H

#include <cmath>
#include <vector>
#include <memory>

size_t getPeakRSS();

template <typename T>
bool
close(const T a, const T b, const double rtol = 1e-4, const double atol = 1e-7)
{
	return std::fabs(a - b) > atol + rtol * std::fabs(b);
}

template <class InputIt1, class InputIt2>
bool
allclose(InputIt1 first1, InputIt1 last1, InputIt2 first2, const double rtol = 1e-4, const double atol = 1e-7)
{
	while (first1 != last1)
	{
		if (close(*first1, *first2, rtol, atol))
			return false;

		++first1;
		++first2;
	}
	return true;
}

template <class T>
class MiniPool
{
public:
	std::unique_ptr<T> get()
	{
		if (cache_.size() == 0)
		{
			return std::make_unique<T>();
		}
		auto r = std::move(cache_.back());
		cache_.pop_back();
		return std::move(r);
	}

	void put(std::unique_ptr<T>&& ptr)
	{
		cache_.emplace_back(std::move(ptr));
	}
private:
	std::vector<std::unique_ptr<T>> cache_;
};
#endif
