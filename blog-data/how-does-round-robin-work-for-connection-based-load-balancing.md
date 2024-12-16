---
title: "How does round-robin work for connection based load-balancing?"
date: "2024-12-16"
id: "how-does-round-robin-work-for-connection-based-load-balancing"
---

Okay, let’s delve into the mechanics of round-robin load balancing, particularly within the context of connection-based scenarios. I've spent considerable time optimizing systems using this approach, and it’s a workhorse for a reason, although it has its limitations as well.

The fundamental idea behind round-robin load balancing is straightforward: distribute incoming client connections sequentially across a pool of available servers. Imagine a carousel; each connection takes a 'seat' on the next available server in the rotation. This distribution happens on a connection-by-connection basis, not request by request if we're discussing connection-based protocols like tcp or websockets. This aspect is quite crucial, and it distinguishes it from load balancing strategies better suited for stateless protocols such as http requests.

In the practical sense, a load balancer typically maintains a simple counter or index. When a new connection arrives, it uses this counter to determine which server should handle it. After assignment, the counter increments (or wraps around if it hits the end of the server list), preparing it for the subsequent connection. This process doesn't take into account server load, health, or capacity; it's a purely mechanical distribution mechanism.

Let’s illustrate this with a few code examples, starting with a Python-based pseudo-code:

```python
class RoundRobinLoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current_server_index = 0

    def get_next_server(self):
        server = self.servers[self.current_server_index]
        self.current_server_index = (self.current_server_index + 1) % len(self.servers)
        return server

#example usage
servers = ["server1.example.com:8080", "server2.example.com:8080", "server3.example.com:8080"]
lb = RoundRobinLoadBalancer(servers)

for i in range(6):
  print(f"Connection {i+1} assigned to: {lb.get_next_server()}")
```

This is a skeletal implementation, focusing on the logical flow of server selection. We have a list of server addresses, `servers`, and the `current_server_index` variable tracks which server should receive the next connection.  The modulo operator (`%`) is essential here, ensuring we wrap back to the beginning of the list once we reach the end. This simple mechanism is at the heart of the round-robin approach. This simplistic approach provides fair distribution if all the underlying servers have similar capabilities and health.

Now, let's consider a slightly more involved example, demonstrating how this could manifest using a simple java server-side implementation, where a load-balancer acts as a proxy:

```java
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;
import java.util.ArrayList;

public class RoundRobinProxy {

    private List<String> servers;
    private int currentServerIndex = 0;

    public RoundRobinProxy(List<String> servers) {
        this.servers = servers;
    }

    public void start(int port) throws IOException {
        ServerSocket serverSocket = new ServerSocket(port);
        System.out.println("Load Balancer listening on port: " + port);

        while (true) {
            Socket clientSocket = serverSocket.accept();
            System.out.println("New connection from: " + clientSocket.getInetAddress());

            Thread proxyThread = new Thread(() -> {
              try {
                handleConnection(clientSocket);
              } catch (IOException e) {
                  e.printStackTrace();
              }
            });
            proxyThread.start();
        }
    }

    private void handleConnection(Socket clientSocket) throws IOException {
        String serverAddress = getNextServer();
        String[] parts = serverAddress.split(":");
        String host = parts[0];
        int serverPort = Integer.parseInt(parts[1]);


        try (Socket serverSocket = new Socket(host, serverPort);
                InputStream clientInput = clientSocket.getInputStream();
                OutputStream clientOutput = clientSocket.getOutputStream();
                InputStream serverInput = serverSocket.getInputStream();
                OutputStream serverOutput = serverSocket.getOutputStream()) {

           // Transfer data in two threads so the proxy works bidirectionally.
           Thread clientToServerThread = new Thread(() -> { transfer(clientInput,serverOutput);});
           Thread serverToClientThread = new Thread(() -> { transfer(serverInput,clientOutput);});
           clientToServerThread.start();
           serverToClientThread.start();
          clientToServerThread.join();
          serverToClientThread.join();
         } catch (Exception e) {
           System.err.println("Error while handling socket " + e.getMessage());
          } finally {
            clientSocket.close();
        }

    }
    private String getNextServer() {
      String server = servers.get(currentServerIndex);
      currentServerIndex = (currentServerIndex + 1) % servers.size();
      return server;
    }
   private void transfer(InputStream input, OutputStream output){
        try{
            byte[] buffer = new byte[1024];
            int bytesRead;
            while((bytesRead = input.read(buffer))!= -1){
                output.write(buffer,0,bytesRead);
            }
        }catch(IOException e){
            // Handle any error here... or just drop connection
        }
    }


    public static void main(String[] args) throws IOException {
        List<String> servers = new ArrayList<>();
        servers.add("localhost:8081");
        servers.add("localhost:8082");
        servers.add("localhost:8083");
        RoundRobinProxy proxy = new RoundRobinProxy(servers);
        proxy.start(8080);
    }
}
```

In this java snippet, we've added the complexities of network interaction. The `RoundRobinProxy` establishes a `ServerSocket`, and upon accepting a client connection, it selects the next target server based on the round-robin strategy. This example also showcases threading for bi-directional data flow between the client and server. It's a highly simplified example for demonstration purposes, and production systems would need far more robustness and error handling. This demonstrates the principle, however: the load balancer acts as an intermediary proxy, directing client connections according to the defined algorithm.

Finally, let's touch on potential performance considerations. A naive implementation might introduce performance bottlenecks if not designed efficiently. For example, locks around the `current_server_index` in a multithreaded environment can cause contention. Instead, a better approach may involve using atomic variables to modify the counter, which removes a lot of that overhead. In a high throughput scenario, even these atomic operations might become a bottleneck.  In that case, a sharded load balancing approach (where each load balancer handles a subset of the total connections, thus distributing the load of managing the index across multiple load balancers) would be appropriate.  Here is an extremely simplistic example using atomic variables:

```java
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class RoundRobinLoadBalancerAtomic {
    private List<String> servers;
    private AtomicInteger currentServerIndex = new AtomicInteger(0);

    public RoundRobinLoadBalancerAtomic(List<String> servers){
        this.servers= servers;
    }
    public String getNextServer(){
        int index = currentServerIndex.getAndIncrement();
        return servers.get(index % servers.size());
    }

    public static void main(String[] args){
        List<String> servers = List.of("server1", "server2", "server3");
        RoundRobinLoadBalancerAtomic balancer = new RoundRobinLoadBalancerAtomic(servers);
        for (int i = 0; i < 7; i++){
            System.out.println("Connection "+ i+1 + " : " + balancer.getNextServer());
        }
    }

}
```

This version of our load balancer utilizes `AtomicInteger` to increment the `currentServerIndex` variable in a thread-safe way, preventing the need for explicit locks. This atomic operation is generally more efficient than lock-based synchronization, reducing potential bottlenecks in a multi-threaded environment.

In summary, round-robin load balancing is a straightforward method of distributing connections. While easy to implement, it assumes equivalent server capacity. Real-world deployments often require more sophisticated strategies that factor in server health, load, and network latency.  For a deeper dive into load balancing theory and techniques, I would recommend exploring "High Performance Browser Networking" by Ilya Grigorik, which goes into the details of how connection management is used in the real world and various load balancing techniques, or "Site Reliability Engineering" by Betsy Beyer, Chris Jones, Jennifer Petoff, Niall Richard Murphy, a broader, but extremely valuable reference, into the considerations of running large scale systems. These resources provide a solid foundation for further exploration of more complex load balancing approaches and their trade-offs.
