---
title: "Why is an aca-py agent not connected with the von-network, and why is the swagger ui not working?"
date: "2024-12-14"
id: "why-is-an-aca-py-agent-not-connected-with-the-von-network-and-why-is-the-swagger-ui-not-working"
---

alright, let's break this down. i've definitely seen this kind of situation pop up more than once, and it's usually a mix of configuration gremlins and network hiccups.  we're talking about a few moving parts here – aca-py agents, the von-network, and swagger ui, so we need to check each one carefully.

first, the aca-py agent not connecting to the von network. this is usually a matter of how the agent is set up to find and connect to the network. think of it like trying to find a specific house in a city, if the address is wrong or the network connection is down, you are not going to get there. i spent a solid week once trying to get a dozen agents to talk, and it turned out to be a single misplaced character in the genesis file path. believe me that is not the way i like to spend my weekends.

let's start with the basics. the aca-py agent needs to know about the von network, and it does this through what’s usually referred to as a genesis transaction file. this file tells the agent where to find the network's nodes and how to interact with them.

here’s what i typically check when i have connection issues:

* **genesis file validity:** is the genesis file actually pointing to the correct von network, you'd be surprised how many times a test network file ends up used for production? check its contents to make sure that the node ids and ip addresses are correct and match the von-network you intend to use, for example if you want to use a test network you need the test node ips not the production ones. also that the encoding is correct, i've had encoding issues before where the file got saved in the wrong format.
*   **agent configuration:** the agent's configuration file needs to reference the correct path to the genesis file. this might be in the configuration file of the agent itself or as a command line argument when you start the agent. you need to double check that the path matches with where you actually have the file in your system. a typo in that is enough to throw the whole thing off.
*   **network connectivity:** can the machine the agent is running on actually reach the von-network nodes? this is a simple but crucial test. a good old ping is a good first step or using telnet to test if there is a connection. if the machine is running in a container this needs to be double checked because those containers might have a virtual network that requires port forwarding to be able to connect with the nodes of the network, and not just the machine host ip.
*   **firewalls:** check that the firewall on the machine where the agent is running is not blocking traffic to/from the von network nodes. make sure that the firewall rules are configured properly so the ports that the von-network nodes are exposing are allowed to be connected.

i usually start with something like this in my python code, i'm using here the `aries_cloudagent` library:

```python
import asyncio
from aries_cloudagent.core.agent import Agent
from aries_cloudagent.config.injection_context import InjectionContext
from aries_cloudagent.config.provider import ClassProvider
from aries_cloudagent.transport.inbound.http import HttpInboundTransport
from aries_cloudagent.transport.outbound.http import HttpOutboundTransport
from aries_cloudagent.ledger.indy import IndyLedger
import os

async def start_agent():
  context = InjectionContext()

  # configuration settings for the agent
  context.settings["default_label"] = "my-agent" # label for the agent
  context.settings["inbound_transport_config"] = [
            {"host": "0.0.0.0", "port": 8000, "transport_class": ClassProvider(HttpInboundTransport)}
        ]
  context.settings["outbound_transport_config"] = [{"transport_class": ClassProvider(HttpOutboundTransport)}]
  context.settings["ledger.genesis_file"] = "/path/to/your/genesis.txn" # <--- this must be correct
  context.settings["ledger.ledger_pool_name"] = "von_network" # name of your von network
  context.settings["ledger.auto_register"] = False # disable auto registration
  context.settings["plugin_paths"] = ["aries_cloudagent.protocols.basicmessage.v1_0"] # we need the plugins

  agent = Agent(context=context)
  await agent.start()

  # basic message send test
  ledger = context.inject(IndyLedger)
  ledger_ready = await ledger.is_ready()

  if ledger_ready:
    print("agent connected to the network!")
  else:
    print("agent couldn't connect to the network :(")

  await asyncio.sleep(3600) # keep agent alive
  await agent.terminate()


if __name__ == "__main__":
  asyncio.run(start_agent())
```

this basic code shows you how an agent is setup, particularly you have to point the `ledger.genesis_file` to the correct path. the `ledger.ledger_pool_name` needs to match your network name (von in this example) and disabling the auto registration, this might be important for specific network setups. also make sure you have the basic message plugin in case you want to send a message to another agent in the network to test communications.

another thing i would check is the agent logs, typically aca-py outputs quite verbose logs that can tell you exactly what's happening and where it might have failed to connect, like a missing file, failed connection to the node, wrong ip address etc. usually this is where the real fun begins, and where most errors can be found.

now, about the swagger ui not working. this generally means that either the swagger configuration on your agent isn't set up correctly or there is a problem connecting with the swagger ui server.

here's what i go through:

*   **swagger enabled:** check if swagger is actually enabled in the aca-py agent config. there’s often a setting to specifically turn it on or off. i can’t tell you how many times i was scratching my head until i realized i had left the ui disabled in the agent setup.
*   **port configuration:** swagger ui usually runs on a specific port that the agent serves. make sure the port is correctly set and that the swagger ui is pointed to the correct port where your agent is running.
*   **correct url:** are you accessing the ui with the correct url? it often is something like `http://localhost:port/api/docs`. you can usually find the correct url in your agent's documentation or logs, make sure the scheme (`http` or `https`) is also correct.
*   **dependency issues:** sometimes, if there are some missing dependencies for the swagger module, the ui won't work. make sure that the agent has all the correct dependencies installed for swagger to work, this is usually in the agent documentation as required dependencies.
*   **firewalls:** again, firewalls can be your nemesis here. confirm that the port that the swagger ui is using is not blocked by a firewall. you might need to configure your firewall to allow traffic on the specific port.

here is a small example of how to activate swagger ui in the python code:

```python
import asyncio
from aries_cloudagent.core.agent import Agent
from aries_cloudagent.config.injection_context import InjectionContext
from aries_cloudagent.config.provider import ClassProvider
from aries_cloudagent.transport.inbound.http import HttpInboundTransport
from aries_cloudagent.transport.outbound.http import HttpOutboundTransport
from aries_cloudagent.ledger.indy import IndyLedger
import os

async def start_agent():
  context = InjectionContext()

  # configuration settings for the agent
  context.settings["default_label"] = "my-agent" # label for the agent
  context.settings["inbound_transport_config"] = [
            {"host": "0.0.0.0", "port": 8000, "transport_class": ClassProvider(HttpInboundTransport)}
        ]
  context.settings["outbound_transport_config"] = [{"transport_class": ClassProvider(HttpOutboundTransport)}]
  context.settings["ledger.genesis_file"] = "/path/to/your/genesis.txn"
  context.settings["ledger.ledger_pool_name"] = "von_network"
  context.settings["ledger.auto_register"] = False
  context.settings["plugin_paths"] = ["aries_cloudagent.protocols.basicmessage.v1_0"]
  # enable swagger ui
  context.settings["admin.enabled"] = True # enable admin
  context.settings["admin.admin_inbound_transports"] = [
            {"host": "0.0.0.0", "port": 8081, "transport_class": ClassProvider(HttpInboundTransport)}
        ] # swagger ui port
  context.settings["admin.host"] = "0.0.0.0" # swagger host
  context.settings["admin.port"] = 8081 # swagger ui port


  agent = Agent(context=context)
  await agent.start()


  # basic message send test
  ledger = context.inject(IndyLedger)
  ledger_ready = await ledger.is_ready()

  if ledger_ready:
    print("agent connected to the network!")
  else:
    print("agent couldn't connect to the network :(")

  await asyncio.sleep(3600) # keep agent alive
  await agent.terminate()


if __name__ == "__main__":
  asyncio.run(start_agent())
```

in this code we can see the `admin.enabled` setting, this turns on the swagger ui module. then we have `admin.admin_inbound_transports` which tells the agent to listen in a specific port for the swagger ui and then finally `admin.host` and `admin.port` configures the host and port for the ui. after this we would expect the swagger ui to work correctly by navigating to the configured address. this configuration is usually a simple one, but if the ui still doesn’t work check the logs of the agent for any error messages that might help.

here's a simple cli command i use to check for running process:

```bash
ps aux | grep "python"
```

this will list any python processes running, and if the agent is there you can then check the logs for any errors. i also like to check with telnet or `curl` if the ports configured are actually open and listening, like so:

```bash
curl http://localhost:8081/api/docs # try the swagger port
curl http://localhost:8000 # try the agent port
```

this might give you some clues if you have connection problems from the outside, always try first locally before moving to more complex tests.

for resources, i would recommend the *hyperledger aries documentation* as a first step. there are quite a few good pages there and examples. also, the *indy ledger documentation* will help you with some details on how the network works, especially the genesis files.  i also found useful a paper named *a comparative analysis of distributed ledger technologies for identity management* by p. perez et al (2018), it can give you a good background on the architecture, and some theoretical base of how these systems work. sometimes going back to basics is the way to go. this are good resources to start with, but do a search for new papers in the subject because things are constantly evolving in this area. and remember if all fails you can always restart everything, that sometimes does the trick (i wish i were joking).
