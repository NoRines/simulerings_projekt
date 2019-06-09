#include <iostream>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"


#include "lcg.h"

using namespace ns3;


void TcPacketsInQueue (QueueDiscContainer qdiscs, Ptr<OutputStreamWrapper> stream)
{
	//Get current queue size value and save to file.
	*stream->GetStream () << Simulator::Now().GetSeconds();
	for(std::size_t i = 0; i < qdiscs.GetN(); i++)
	{
		Ptr<QueueDisc> p = qdiscs.Get (i);
		uint32_t size = p->GetNPackets(); 
		*stream->GetStream() << "," << size;
	}
	*stream->GetStream() << std::endl;
}

static void GenerateTraffic (Ptr<Socket> socket, Ptr<ExponentialRandomVariable> randomSize,	Ptr<ExponentialRandomVariable> randomTime)
{
	uint32_t pktSize = randomSize->GetInteger (); //Get random value for packet size
	socket->Send (Create<Packet> (pktSize));

	Time pktInterval = Seconds(randomTime->GetValue ()); //Get random value for next packet generation time 
	Simulator::Schedule (pktInterval, &GenerateTraffic, socket, randomSize, randomTime); //Schedule next packet generation
}



int main(int argc, char** argv)
{
	// A B C D E F G Server Router
	// 0 1 2 3 4 5 6 7      8
	
	//LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
	//LogComponentEnable("MyUdpEchoServerApplication", LOG_LEVEL_INFO);
	
	double simTime = 6.0;

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
	p2p.SetQueue ("ns3::DropTailQueue", "MaxSize", StringValue ("1p"));

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

	TrafficControlHelper tch;
	tch.SetRootQueueDisc("ns3::FifoQueueDisc", "MaxSize", StringValue("1000p"));
	
	QueueDiscContainer qdiscs = tch.Install(dcAtoE);
	qdiscs.Add(tch.Install(dcEtoG));
	qdiscs.Add(tch.Install(dcBtoF));
	qdiscs.Add(tch.Install(dcCtoF));
	qdiscs.Add(tch.Install(dcDtoG));
	qdiscs.Add(tch.Install(dcFtoG));
	qdiscs.Add(tch.Install(dcGtoRouter));
	qdiscs.Add(tch.Install(dcGtoServer));

	AsciiTraceHelper asciiTraceHelper;
	Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream("all_queues.tr");

	*stream->GetStream() << "time," << "AfromE," << "EfromA," << "EfromG," << "GfromE,"
		<< "BfromF," << "FfromB," << "CfromF," << "FfromC," << "DfromG," << "GfromD,"
		<< "FfromG," << "GfromF," << "GfromRouter," << "RouterFromG," << "GfromServer," << "ServerFromG" << std::endl;

	for(float t = 1.0f; t < simTime; t+=0.00001f)
	{
		Simulator::Schedule(Seconds(t), &TcPacketsInQueue, qdiscs, stream);
	}



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
	double stoptime = simTime;
	ApplicationContainer serverApp = serverHelper.Install(ncGtoServer.Get(1));
	serverApp.Start(Seconds(starttime));
	serverApp.Stop(Seconds(stoptime));


	PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), 300));
	ApplicationContainer packetSinkApp = packetSinkHelper.Install(ncGtoRouter.Get(1));
	packetSinkApp.Start(Seconds(starttime));
	packetSinkApp.Stop(Seconds(stoptime));


	TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");


	auto DefThing = [&tid, starttime](Ptr<Node> node, Ipv4Address serverAddress, double meanInterTime, double meanSize)
	{
		Ptr<Socket> sourceA = Socket::CreateSocket(node, tid);
		sourceA->Connect (InetSocketAddress (serverAddress, 9));
	
		double mean = meanInterTime;
		Ptr<ExponentialRandomVariable> randomTime = CreateObject<ExponentialRandomVariable> ();
		randomTime->SetAttribute ("Mean", DoubleValue (mean));
	
		//mean = (bitRate/(8*4800)-30); // (1 000 000 [b/s])/(8 [b/B] * packet service rate [1/s]) - 30 [B (header bytes)]
		//std::cout << mean << std::endl;
		mean = meanSize;
		Ptr<ExponentialRandomVariable> randomSize = CreateObject<ExponentialRandomVariable> ();
		randomSize->SetAttribute ("Mean", DoubleValue (mean));

		Simulator::ScheduleWithContext (sourceA->GetNode()->GetId(), Seconds (starttime), &GenerateTraffic, sourceA, randomSize, randomTime);
	};
	
	DefThing(ncAtoE.Get(0), iGtoServer.GetAddress(1), 0.002, 70);
	DefThing(ncBtoF.Get(0), iGtoServer.GetAddress(1), 0.002, 70);
	DefThing(ncCtoF.Get(0), iGtoServer.GetAddress(1), 0.0005, 70);
	DefThing(ncDtoG.Get(0), iGtoServer.GetAddress(1), 0.001, 70);

	Ipv4GlobalRoutingHelper::PopulateRoutingTables();

	//FlowMonitorHelper flowmon;
	//Ptr<FlowMonitor> monitor = flowmon.InstallAll();

	p2p.EnablePcap("proj_router", dcGtoRouter.Get(1), true);
	p2p.EnablePcap("proj_server", dcGtoServer.Get(1), true);
	p2p.EnablePcap("proj_g", dcGtoServer.Get(0), true);
	Simulator::Stop(Seconds(simTime));
	Simulator::Run();

	//Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
	//std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();


	Simulator::Destroy();

	return 0;
}
