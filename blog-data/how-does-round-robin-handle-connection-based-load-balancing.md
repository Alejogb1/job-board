---
title: "How does round-robin handle connection-based load balancing?"
date: "2024-12-16"
id: "how-does-round-robin-handle-connection-based-load-balancing"
---

Alright, let’s talk about round-robin load balancing in the context of connection-based traffic. I’ve seen this implemented—and debugged it—more times than I care to remember. It’s one of those foundational concepts that everyone touches, but the nuances can trip up even seasoned developers. My experience, especially during my time at a company scaling out their microservices architecture, has given me a practical, rather than purely theoretical, understanding of how this works.

Round-robin, at its core, is a deterministic algorithm. It doesn’t try to be particularly smart. It’s about simplicity and predictability. In essence, it distributes incoming connection requests sequentially across a list of available servers. Think of it like a rotating queue; each server gets its turn before the cycle repeats. This is in contrast to other load balancing methods, such as least connection or weighted round-robin, which consider server load or capacity.

When applied to connection-based protocols (like tcp), round-robin usually deals with the initiation of new connections. The load balancer maintains a list of backend servers. Each time a new client request to establish a connection is received, the load balancer selects the next server in that list, according to the defined sequence. Once a connection is established, the subsequent communications for that session will continue to use the same backend server. This is crucial for maintaining session state.

Let’s say we have three servers, `server_a`, `server_b`, and `server_c`, our typical configuration for an application where each server is running an instance of the service. The first incoming connection would be directed to `server_a`. The second to `server_b`, and the third to `server_c`. The fourth connection will then cycle back to `server_a` and so on. This simple, cyclic distribution method is the essence of round-robin in the context of persistent connections.

One advantage is that it requires minimal overhead on the load balancer. There’s no need to constantly monitor server loads or perform complex calculations. This makes it very performant and easy to implement. However, it also has drawbacks. A primary one is that, without any regard to server performance or current load, servers can become overloaded if they're not all equally capable, or if some are handling more persistent, intensive client connections. This can result in uneven load distribution even with round-robin initially spreading connections evenly.

Now, let’s dive into some code to solidify this understanding. Keep in mind these examples are simplified for clarity and not necessarily production-ready.

**Example 1: Basic In-Memory Round-Robin Implementation (Python)**

```python
class RoundRobinBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.index = 0

    def get_next_server(self):
        server = self.servers[self.index]
        self.index = (self.index + 1) % len(self.servers)
        return server


# Example Usage
servers = ["server_a", "server_b", "server_c"]
balancer = RoundRobinBalancer(servers)

for _ in range(6):
    print(f"Connection assigned to: {balancer.get_next_server()}")
```

This Python example demonstrates a basic in-memory round-robin approach. It initializes a list of servers and keeps an internal index to track the next server to be selected. The `%` operator here ensures the index wraps around to the beginning after reaching the end of the server list, creating the circular logic we expect from round robin.

**Example 2: Handling Server Removal (Simplified Java)**

```java
import java.util.ArrayList;
import java.util.List;

public class RoundRobinBalancer {

    private List<String> servers;
    private int index = 0;

    public RoundRobinBalancer(List<String> servers) {
        this.servers = new ArrayList<>(servers);
    }

    public synchronized String getNextServer() {
        if(servers.isEmpty()){
           return null; // Handle edge case.
        }
        String server = servers.get(index);
        index = (index + 1) % servers.size();
        return server;
    }


    public synchronized void removeServer(String serverToRemove) {
          servers.remove(serverToRemove);

         if (index >= servers.size()){
           index = 0;
        }
    }


    public static void main(String[] args) {
        List<String> initialServers = new ArrayList<>(List.of("server_a", "server_b", "server_c"));
        RoundRobinBalancer balancer = new RoundRobinBalancer(initialServers);

        for(int i = 0; i < 7; i++) {
            System.out.println("Connection Assigned to: " + balancer.getNextServer());
        }

        balancer.removeServer("server_b");

        System.out.println("Server b removed.");

        for(int i=0; i<5; i++){
            System.out.println("Connection Assigned to: " + balancer.getNextServer());
        }
    }

}
```
This Java example introduces a crucial aspect: dynamically handling server removal. In real-world scenarios, servers can fail or be taken offline for maintenance. This example adds a `removeServer` method to remove a server from the list and adjust the index to ensure subsequent requests are directed to the remaining servers. The synchronized keyword prevents threading issues when removing or accessing the shared state.

**Example 3: A Simple Illustration using Go Routines**
```go
package main

import (
	"fmt"
	"sync"
	"time"
)

type RoundRobinBalancer struct {
	servers []string
	index   int
	mu      sync.Mutex
}

func NewRoundRobinBalancer(servers []string) *RoundRobinBalancer {
	return &RoundRobinBalancer{servers: servers, index: 0}
}


func (r *RoundRobinBalancer) getNextServer() string {
   r.mu.Lock()
    defer r.mu.Unlock()
    server := r.servers[r.index]
    r.index = (r.index + 1) % len(r.servers)
    return server
}

func main() {
	servers := []string{"server_a", "server_b", "server_c"}
	balancer := NewRoundRobinBalancer(servers)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(connectionNum int) {
			defer wg.Done()
			server := balancer.getNextServer()
			fmt.Printf("Connection %d assigned to: %s\n", connectionNum, server)
			time.Sleep(100 * time.Millisecond) // Simulate connection processing.
		}(i)
	}
	wg.Wait()
}
```

This Go example utilizes goroutines to simulate concurrent requests. This demonstrates how the round-robin balancer behaves when handling multiple incoming connection requests at once. The `sync.Mutex` ensures that the index and server list access is synchronized when called concurrently from multiple goroutines.

Implementing these algorithms in real-world scenarios involves dealing with various network constraints, timeout policies, health checks, and more. The three examples here only focus on illustrating the core principle.

For a deeper understanding of load balancing beyond round-robin, I recommend looking into *“High Performance Web Sites: Essential Knowledge for Frontend Engineers”* by Steve Souders for frontend performance strategies and the *“Site Reliability Engineering: How Google Runs Production Systems”* book, which provides broader context to the overall system architecture of which load balancing is part. *“TCP/IP Illustrated, Volume 1: The Protocols”* by W. Richard Stevens provides an excellent in-depth explanation of the network protocols and concepts that underly these systems, as well as for understanding the mechanics of connection-based communications. Understanding these fundamentals is crucial before delving into implementation details.

In summary, round-robin load balancing, while seemingly straightforward, presents its own set of trade-offs. Its simplicity and low overhead make it an attractive option, particularly when the backend servers have similar performance capabilities and consistent loads. However, it’s essential to be aware of its limitations and consider alternative techniques if you need more granular control or dynamic adjustments based on real-time load. It's never a “one-size-fits-all,” but one that can be a strong foundation for many applications.
