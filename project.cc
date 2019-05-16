

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"

#include "lcg.h"

using namespace ns3;

int main(int argc, char** argv)
{
	//lcg::state s = {1, 13, 1, 100};

	Ptr<ExponentialRandomVariable> x = CreateObject<ExponentialRandomVariable> ();
	x->SetAttribute ("Mean", DoubleValue (0.5));


	for(int i = 0; i < 10000; i++)
	{
		//std::cout << lcg::norm_gen(s) << std::endl;
		std::cout << x->GetValue() << ",";
		//val = x->GetValue();
		//val = lcg::norm_gen(s);
	}


	return 0;
}
