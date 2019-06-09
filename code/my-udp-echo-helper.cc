#include "my-udp-echo-helper.h"
#include "ns3/my-udp-echo-server.h"
#include "ns3/udp-echo-server.h"
#include "ns3/udp-echo-client.h"
#include "ns3/uinteger.h"
#include "ns3/names.h"

namespace ns3 {

MyUdpEchoServerHelper::MyUdpEchoServerHelper (uint16_t port, Address routerAddress)
{
  m_factory.SetTypeId (MyUdpEchoServer::GetTypeId ());
  SetAttribute ("Port", UintegerValue (port));
  m_routerAddress = routerAddress;
}

void 
MyUdpEchoServerHelper::SetAttribute (
  std::string name, 
  const AttributeValue &value)
{
  m_factory.Set (name, value);
}

ApplicationContainer
MyUdpEchoServerHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
MyUdpEchoServerHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
MyUdpEchoServerHelper::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}

Ptr<Application>
MyUdpEchoServerHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<MyUdpEchoServer> app = m_factory.Create<MyUdpEchoServer> ();
  app->SetRouterAddress(m_routerAddress);
  node->AddApplication (app);

  return app;
}


// CLIENT CODE STARTS HERE::::::::::::::::::::::::::::::::::::::::::::::::;SD:;SDGÖJSLDLGKJSDLKGJSLKDJGLK



MyUdpEchoClientHelper::MyUdpEchoClientHelper (Address address, uint16_t port)
{
  m_factory.SetTypeId (UdpEchoClient::GetTypeId ());
  SetAttribute ("RemoteAddress", AddressValue (address));
  SetAttribute ("RemotePort", UintegerValue (port));
}

MyUdpEchoClientHelper::MyUdpEchoClientHelper (Address address)
{
  m_factory.SetTypeId (UdpEchoClient::GetTypeId ());
  SetAttribute ("RemoteAddress", AddressValue (address));
}

void 
MyUdpEchoClientHelper::SetAttribute (
  std::string name, 
  const AttributeValue &value)
{
  m_factory.Set (name, value);
}

void
MyUdpEchoClientHelper::SetFill (Ptr<Application> app, std::string fill)
{
  app->GetObject<UdpEchoClient>()->SetFill (fill);
}

void
MyUdpEchoClientHelper::SetFill (Ptr<Application> app, uint8_t fill, uint32_t dataLength)
{
  app->GetObject<UdpEchoClient>()->SetFill (fill, dataLength);
}

void
MyUdpEchoClientHelper::SetFill (Ptr<Application> app, uint8_t *fill, uint32_t fillLength, uint32_t dataLength)
{
  app->GetObject<UdpEchoClient>()->SetFill (fill, fillLength, dataLength);
}

ApplicationContainer
MyUdpEchoClientHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
MyUdpEchoClientHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
MyUdpEchoClientHelper::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}

Ptr<Application>
MyUdpEchoClientHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<Application> app = m_factory.Create<UdpEchoClient> ();
  node->AddApplication (app);

  return app;
}

} // namespace ns3
