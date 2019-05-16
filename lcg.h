#ifndef LCG_H
#define LCG_H

#include <cstddef>
#include <cmath>

namespace lcg
{
	struct state
	{
		std::size_t last;
		std::size_t a;
		std::size_t c;
		std::size_t mod;
	};

	std::size_t gen_n(state& s)
	{
		return s.last = (s.a * s.last + s.c) % s.mod;
	}

	double norm_gen(state& s)
	{
		return gen_n(s) / (double)s.mod;
	}

	double exp_transform(double u, double mean)
	{
		return -(mean) * std::log(1 - u);
	}
}

#endif
