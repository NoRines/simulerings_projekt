#ifndef PRNG_H
#define PRNG_H

void prng(unsigned int* values, int multipl, int c, int mod_base, int seed, int amount)
{
	values[0] = (seed * multipl + c) % mod_base;
	for (int i = 1; i < amount; i++)
	{
		values[i] = (values[i - 1] * multipl + c) % mod_base;
	}
}
#endif