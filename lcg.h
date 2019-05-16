#ifndef LCH_H
#define LCH_H
#include <cstddef>
#include <cmath>
namespace lcg
{
    struct state{
        std::size_t last;
        std::size_t a;
        std::size_t c;
        std::size_t mod;
    };

    std::size_t gen_n(state & s)
    {
        return s.last = (s.a * s.last + s.c) % s.mod;
    }
    double normalized_gen(state &s)
    {
        return gen_n(s)/((double)s.mod);
    }
    template <typename T>
    T exp_transform(double u, double beta)
    {
        double lambda = 1/beta;
        return ((-lambda) * log(1-u));
    }

}
#endif
