#include <iostream>
#include "lcg.h"

int main()
{
	using namespace lcg;

	state s = {1, 22695477, 1, (unsigned int)(1 << 31)};

	for(int i = 0; i < 10000; i++)
		std::cout << exp_transform(norm_gen(s), 2.0) << ",";

	return 0;
}
