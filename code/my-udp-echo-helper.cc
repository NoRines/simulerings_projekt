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

}
