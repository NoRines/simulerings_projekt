#include <iostream>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"

#include "lcg.h"

using namespace ns3;

int main(int argc, char** argv)
{
	// A B C D E F G Server Router
	// 0 1 2 3 4 5 6 7      8
	
	LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
	LogComponentEnable("MyUdpEchoServerApplication", LOG_LEVEL_INFO);

	NodeContainer container;
	container.Create(9);

	NodeContainer ncAtoE = NodeContainer(container.Get(0), container.Get(4));
	NodeContainer ncEtoG = NodeContainer(container.Get(4), container.Get(6));
	NodeContainer ncBtoF = NodeContainer(container.Get(1), container.Get(5));
	NodeContainer ncFtoG = NodeContainer(container.Get(5), container.Get(6));
	NodeContainer ncCtoF = NodeContainer(container.Get(2), container.Get(5));
	NodeContainer ncDtoG = NodeContainer(container.Get(3), container.Get(6));
	NodeContainer ncGtoServer = NodeContainer(container.Get(6), container.Get(7));
	NodeContainer ncGtoRouter = NodeContainer(container.Get(6), container.Get(8));

	PointToPointHelper p2p;

	p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
	NetDeviceContainer dcAtoE = p2p.Install(ncAtoE);
	NetDeviceContainer dcEtoG = p2p.Install(ncEtoG);
	NetDeviceContainer dcBtoF = p2p.Install(ncBtoF);
	NetDeviceContainer dcCtoF = p2p.Install(ncCtoF);
	NetDeviceContainer dcDtoG = p2p.Install(ncDtoG);

	p2p.SetDeviceAttribute("DataRate", StringValue("8Mbps"));
	NetDeviceContainer dcFtoG = p2p.Install(ncFtoG);
	NetDeviceContainer dcGtoRouter = p2p.Install(ncGtoRouter);

	p2p.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
	NetDeviceContainer dcGtoServer = p2p.Install(ncGtoServer);


	InternetStackHelper internet;
	internet.Install(container);


	Ipv4AddressHelper ipv4;

	ipv4.SetBase("10.10.0.0", "255.255.255.0");
	Ipv4InterfaceContainer iAtoE = ipv4.Assign(dcAtoE);

	ipv4.SetBase("10.10.1.0", "255.255.255.0");
	Ipv4InterfaceContainer iEtoG = ipv4.Assign(dcEtoG);

	ipv4.SetBase("10.10.2.0", "255.255.255.0");
	Ipv4InterfaceContainer iBtoF = ipv4.Assign(dcBtoF);

	ipv4.SetBase("10.10.3.0", "255.255.255.0");
	Ipv4InterfaceContainer iCtoF = ipv4.Assign(dcCtoF);

	ipv4.SetBase("10.10.4.0", "255.255.255.0");
	Ipv4InterfaceContainer iDtoG = ipv4.Assign(dcDtoG);

	ipv4.SetBase("10.10.5.0", "255.255.255.0");
	Ipv4InterfaceContainer iFtoG = ipv4.Assign(dcFtoG);

	ipv4.SetBase("10.10.6.0", "255.255.255.0");
	Ipv4InterfaceContainer iGtoServer = ipv4.Assign(dcGtoServer);

	ipv4.SetBase("10.10.7.0", "255.255.255.0");
	Ipv4InterfaceContainer iGtoRouter = ipv4.Assign(dcGtoRouter);


	MyUdpEchoServerHelper serverHelper(9, InetSocketAddress(iGtoRouter.GetAddress(1), 300));
	double starttime = 1.0;
	double stoptime = 100.0;
	ApplicationContainer serverApp = serverHelper.Install(ncGtoServer.Get(1));
	serverApp.Start(Seconds(starttime));
	serverApp.Stop(Seconds(stoptime));

	UdpEchoClientHelper clientHelper(iGtoServer.GetAddress(1), 9);
	clientHelper.SetAttribute ("MaxPackets", UintegerValue (100000000));
	clientHelper.SetAttribute ("Interval", TimeValue (Seconds (0.1)));
	clientHelper.SetAttribute ("PacketSize", UintegerValue (1024));

	PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), 300));
	ApplicationContainer packetSinkApp = packetSinkHelper.Install(ncGtoRouter.Get(1));
	packetSinkApp.Start(Seconds(1.0));
	packetSinkApp.Stop(Seconds(100.0));


	double appstarttime = 2.0;
	ApplicationContainer clientAppAE = clientHelper.Install(ncAtoE.Get(0));
	clientAppAE.Start(Seconds(appstarttime));
	clientAppAE.Stop(Seconds(stoptime));
	ApplicationContainer clientAppBF = clientHelper.Install(ncBtoF.Get(0));
	clientAppBF.Start(Seconds(appstarttime));
	clientAppBF.Stop(Seconds(stoptime));
	ApplicationContainer clientAppCF = clientHelper.Install(ncCtoF.Get(0));
	clientAppCF.Start(Seconds(appstarttime));
	clientAppCF.Stop(Seconds(stoptime));
	ApplicationContainer clientAppDG = clientHelper.Install(ncDtoG.Get(0));
	clientAppDG.Start(Seconds(appstarttime));
	clientAppDG.Stop(Seconds(stoptime));
	Ipv4GlobalRoutingHelper::PopulateRoutingTables();

	p2p.EnablePcap("proj_router", dcGtoRouter.Get(1), true);
	p2p.EnablePcap("proj_server", dcGtoServer.Get(1), true);
	Simulator::Run();
	Simulator::Destroy();

	return 0;
}
