/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright 2007 University of Washington
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "ns3/log.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/address-utils.h"
#include "ns3/nstime.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/socket.h"
#include "ns3/udp-socket.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"

#include "my-udp-echo-server.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MyUdpEchoServerApplication");

NS_OBJECT_ENSURE_REGISTERED (MyUdpEchoServer);

TypeId
MyUdpEchoServer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MyUdpEchoServer")
    .SetParent<Application> ()
    .SetGroupName("Applications")
    .AddConstructor<MyUdpEchoServer> ()
    .AddAttribute ("Port", "Port on which we listen for incoming packets.",
                   UintegerValue (9),
                   MakeUintegerAccessor (&MyUdpEchoServer::m_port),
                   MakeUintegerChecker<uint16_t> ())
	.AddAttribute("RouterAddress", "Address to router to route packets to.",
					AddressValue(),
					MakeAddressAccessor(&MyUdpEchoServer::m_routerAddress),
					MakeAddressChecker())
    .AddTraceSource ("Rx", "A packet has been received",
                     MakeTraceSourceAccessor (&MyUdpEchoServer::m_rxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("RxWithAddresses", "A packet has been received",
                     MakeTraceSourceAccessor (&MyUdpEchoServer::m_rxTraceWithAddresses),
                     "ns3::Packet::TwoAddressTracedCallback")
  ;
  return tid;
}

MyUdpEchoServer::MyUdpEchoServer ()
{
  NS_LOG_FUNCTION (this);
  randVar = CreateObject<UniformRandomVariable>();
}

MyUdpEchoServer::~MyUdpEchoServer()
{
  NS_LOG_FUNCTION (this);
  m_socket = 0;
  m_socket6 = 0;
}

void
MyUdpEchoServer::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  Application::DoDispose ();
}

void MyUdpEchoServer::SetRouterAddress(Address routerAddress)
{
	m_routerAddress = routerAddress;
}

void 
MyUdpEchoServer::StartApplication (void)
{
  NS_LOG_FUNCTION (this);

  if (m_socket == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      m_socket = Socket::CreateSocket (GetNode (), tid);
      InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), m_port);
      if (m_socket->Bind (local) == -1)
        {
          NS_FATAL_ERROR ("Failed to bind socket");
        }
      if (addressUtils::IsMulticast (m_local))
        {
          Ptr<UdpSocket> udpSocket = DynamicCast<UdpSocket> (m_socket);
          if (udpSocket)
            {
              // equivalent to setsockopt (MCAST_JOIN_GROUP)
              udpSocket->MulticastJoinGroup (0, m_local);
            }
          else
            {
              NS_FATAL_ERROR ("Error: Failed to join multicast group");
            }
        }
    }

  if (m_socket6 == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      m_socket6 = Socket::CreateSocket (GetNode (), tid);
      Inet6SocketAddress local6 = Inet6SocketAddress (Ipv6Address::GetAny (), m_port);
      if (m_socket6->Bind (local6) == -1)
        {
          NS_FATAL_ERROR ("Failed to bind socket");
        }
      if (addressUtils::IsMulticast (local6))
        {
          Ptr<UdpSocket> udpSocket = DynamicCast<UdpSocket> (m_socket6);
          if (udpSocket)
            {
              // equivalent to setsockopt (MCAST_JOIN_GROUP)
              udpSocket->MulticastJoinGroup (0, local6);
            }
          else
            {
              NS_FATAL_ERROR ("Error: Failed to join multicast group");
            }
        }
    }

  m_socket->SetRecvCallback (MakeCallback (&MyUdpEchoServer::HandleRead, this));
  m_socket6->SetRecvCallback (MakeCallback (&MyUdpEchoServer::HandleRead, this));
}

void 
MyUdpEchoServer::StopApplication ()
{
  NS_LOG_FUNCTION (this);

  if (m_socket != 0) 
    {
      m_socket->Close ();
      m_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    }
  if (m_socket6 != 0) 
    {
      m_socket6->Close ();
      m_socket6->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    }
}

void 
MyUdpEchoServer::HandleRead(Ptr<Socket> socket)
{
	NS_LOG_FUNCTION (this << socket);

	bool toRouter = false;
	double r = randVar->GetValue(0.0, 1.0);
	if(r > 0.7)
		toRouter = true;

	Ptr<Packet> packet;
	Address from;
	Address localAddress;
	while ((packet = socket->RecvFrom (from)))
	{
		HandleReadInternal(socket, packet, localAddress, from, toRouter);
	}
}

void 
MyUdpEchoServer::HandleReadInternal (Ptr<Socket> socket, Ptr<Packet> packet, Address& localAddress, Address& from, bool toRouter)
{
	socket->GetSockName (localAddress);
	m_rxTrace (packet);
	m_rxTraceWithAddresses (packet, from, localAddress);

	if (InetSocketAddress::IsMatchingType (from))
	{
		NS_LOG_INFO ("At time " << Simulator::Now ().GetSeconds () << "s server received " << packet->GetSize () << " bytes from " <<
	             InetSocketAddress::ConvertFrom (from).GetIpv4 () << " port " <<
	             InetSocketAddress::ConvertFrom (from).GetPort ());
	}
	else if (Inet6SocketAddress::IsMatchingType (from))
	{
		NS_LOG_INFO ("At time " << Simulator::Now ().GetSeconds () << "s server received " << packet->GetSize () << " bytes from " <<
	             Inet6SocketAddress::ConvertFrom (from).GetIpv6 () << " port " <<
	             Inet6SocketAddress::ConvertFrom (from).GetPort ());
	}
	
	packet->RemoveAllPacketTags ();
	packet->RemoveAllByteTags ();
	
	NS_LOG_LOGIC ("Echoing packet");

	// If the message should go to the router use the router address
	if(toRouter == false)
		socket->SendTo (packet, 0, from);
	else
		socket->SendTo (packet, 0, m_routerAddress);
	
	if (InetSocketAddress::IsMatchingType (from))
	{
		NS_LOG_INFO ("At time " << Simulator::Now ().GetSeconds () << "s server sent " << packet->GetSize () << " bytes to " <<
	             InetSocketAddress::ConvertFrom (from).GetIpv4 () << " port " <<
	             InetSocketAddress::ConvertFrom (from).GetPort ());
	}
	else if (Inet6SocketAddress::IsMatchingType (from))
	{
		NS_LOG_INFO ("At time " << Simulator::Now ().GetSeconds () << "s server sent " << packet->GetSize () << " bytes to " <<
	             Inet6SocketAddress::ConvertFrom (from).GetIpv6 () << " port " <<
	             Inet6SocketAddress::ConvertFrom (from).GetPort ());
	}
}

}
