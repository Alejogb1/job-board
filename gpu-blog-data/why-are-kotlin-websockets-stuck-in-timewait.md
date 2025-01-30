---
title: "Why are Kotlin websockets stuck in TIME_WAIT?"
date: "2025-01-30"
id: "why-are-kotlin-websockets-stuck-in-timewait"
---
Kotlin websockets exhibiting a persistent `TIME_WAIT` state, even after what appears to be a clean closure, often points to a fundamental misunderstanding of the TCP protocol's connection lifecycle, specifically when handled within the context of asynchronous network operations common in Kotlin coroutines and multiplatform networking libraries. This isn't strictly a Kotlin problem, but the way libraries like Ktor or specific native socket implementations interact with the underlying OS networking stack can exacerbate these issues if not managed carefully.

A core fact is that the `TIME_WAIT` state is not a problem per se; it's a necessary part of TCP's reliability mechanisms. When a TCP connection is closed, the endpoint that performs the active close (sends the `FIN` packet first) enters `TIME_WAIT`. This ensures that any delayed packets from the previous connection reach their destination and are properly processed, preventing data corruption or interpretation errors on subsequent connections using the same socket tuple (local address, local port, remote address, remote port). The duration of this `TIME_WAIT` is typically twice the Maximum Segment Lifetime (2MSL), a configurable operating system setting. In practice, this often translates to about 1 to 4 minutes, varying by OS.

The problem arises when the server, particularly one that rapidly establishes and tears down websocket connections, such as a chat server or real-time data feed, repeatedly transitions to `TIME_WAIT` faster than the OS allows these connections to clear. If this server also initiates the close (which is often the case in server-driven shutdowns), the same port on the server side can accumulate many sockets in `TIME_WAIT`. When the number of sockets in `TIME_WAIT` reaches the operating system’s ephemeral port limit, new connection attempts will fail until enough time passes for sockets to age out of this state. While the application-level code might believe it closed the socket, from the network stack’s perspective, the resources are still held until the OS releases them.

I've encountered this myself while building a real-time analytics dashboard using Ktor. During a heavy load test, the server began exhibiting connection failures. After some investigation with `netstat`, the system revealed a large number of server sockets in `TIME_WAIT`. The application's websocket handling logic was sound, but it did not anticipate the rapid cycling of connections. The default behavior for Ktor (and many frameworks) is to let the operating system handle socket closing without explicit intervention, relying on graceful cleanup based on input stream closure and connection timeouts. This approach is usually sufficient, but it fails under load when rapid creation and destruction of many websocket connections become the norm.

Let's illustrate this with a simplified scenario using Ktor on the server side.

**Code Example 1: Basic Ktor Websocket Setup (Vulnerable to TIME_WAIT Accumulation)**

```kotlin
import io.ktor.server.application.*
import io.ktor.server.routing.*
import io.ktor.server.websocket.*
import io.ktor.websocket.*
import kotlinx.coroutines.delay
import java.time.Duration

fun Application.module() {
    install(WebSockets) {
        pingPeriod = Duration.ofSeconds(15) // Keep-alive
        timeout = Duration.ofSeconds(15) // Timeout
        maxFrameSize = Long.MAX_VALUE // Allow large frames
    }

    routing {
        webSocket("/echo") {
            try {
                for (frame in incoming) {
                   //Process frame and potentially close here.
                   if(frame is Frame.Text) {
                       val textFrame = frame as Frame.Text
                       send(Frame.Text("Received: ${textFrame.readText()}"))
                      // Simplified Example: Closing connection immediately after response for illustrative purposes
                       close(CloseReason(CloseReason.Codes.NORMAL, "Closing"))
                    }
                }
            }
            catch(e: Exception) {
                  // Handle exceptions gracefully.
            }
        }
    }
}

fun main(args: Array<String>) {
  io.ktor.server.netty.EngineMain.main(args)
}

```

In this basic example, each websocket connection sends back a message received from the client. Once this send-response cycle is complete, I call `close()` to signal the connection's end. Although this looks fine, the server actively closes the connection, making it responsible for entering `TIME_WAIT`. Repeating this rapid connect-send-close pattern will quickly accumulate sockets in `TIME_WAIT` and exhaust server ports.

A crucial detail here is that `close()` doesn’t immediately relinquish resources at the OS level. It signals the termination sequence but doesn’t guarantee immediate resource release, especially when the server is the initiator.

**Code Example 2: Attempting Socket Reuse (Potentially Problematic)**

```kotlin
import io.ktor.server.application.*
import io.ktor.server.routing.*
import io.ktor.server.websocket.*
import io.ktor.websocket.*
import kotlinx.coroutines.delay
import java.net.StandardSocketOptions
import java.time.Duration

fun Application.module() {

  install(WebSockets) {
      pingPeriod = Duration.ofSeconds(15) // Keep-alive
      timeout = Duration.ofSeconds(15) // Timeout
      maxFrameSize = Long.MAX_VALUE // Allow large frames
      configure {
          socketOptions {
             //This would be more effective if server was accepting multiple connection on same socket.
            setOption(StandardSocketOptions.SO_REUSEADDR, true)
            setOption(StandardSocketOptions.SO_REUSEPORT, true)
          }
      }
  }

    routing {
        webSocket("/echo") {
            try {
                for (frame in incoming) {
                     //Process frame and potentially close here.
                   if(frame is Frame.Text) {
                       val textFrame = frame as Frame.Text
                       send(Frame.Text("Received: ${textFrame.readText()}"))
                      // Simplified Example: Closing connection immediately after response for illustrative purposes
                       close(CloseReason(CloseReason.Codes.NORMAL, "Closing"))
                    }
                }
            }
            catch(e: Exception) {
                  // Handle exceptions gracefully.
            }
        }
    }
}

fun main(args: Array<String>) {
  io.ktor.server.netty.EngineMain.main(args)
}
```

Here, I tried to use the socket options `SO_REUSEADDR` and `SO_REUSEPORT` when configuring the websockets on the server. These options, while helpful in some situations like faster server restarts, do *not* directly resolve issues with `TIME_WAIT`. They primarily allow binding to ports already held by sockets in `TIME_WAIT` or in the `CLOSE_WAIT` state, but they don't prevent the state itself. In a websocket scenario where each client uses a separate ephemeral port, these options are not relevant for fixing the TIME_WAIT problem, since these sockets, though they are on the same server IP, are not on the same port as previous connections. They are being used, so you might get another port on the next connection to the same host.

**Code Example 3: Adjusting Closure and Connection Handling (More Robust)**

```kotlin
import io.ktor.server.application.*
import io.ktor.server.routing.*
import io.ktor.server.websocket.*
import io.ktor.websocket.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.channels.ClosedReceiveChannelException
import java.time.Duration

fun Application.module() {
    install(WebSockets) {
        pingPeriod = Duration.ofSeconds(15) // Keep-alive
        timeout = Duration.ofSeconds(15) // Timeout
        maxFrameSize = Long.MAX_VALUE // Allow large frames
    }

    routing {
        webSocket("/echo") {
            try {
                for (frame in incoming) {
                   if(frame is Frame.Text) {
                       val textFrame = frame as Frame.Text
                       send(Frame.Text("Received: ${textFrame.readText()}"))

                       //Instead of closing ourselves, let client close.
                        //This is more scalable and will reduce TIME_WAIT on the server-side
                   }

                }
            }
            catch(e: ClosedReceiveChannelException){
                //This handles a client side disconnect more gracefully
              println("Client closed connection")
            }
            catch(e: Exception) {
                  println("Exception while handling socket: $e")
            }
        }
    }
}

fun main(args: Array<String>) {
  io.ktor.server.netty.EngineMain.main(args)
}
```

In this modified approach, the server listens for messages. Instead of proactively closing the connection, I now handle exceptions related to channel closure. Now the server *waits* for a client disconnect (`ClosedReceiveChannelException` means that the client initiated the connection close.) rather than doing it itself. When the client closes the websocket, the *client* socket enters `TIME_WAIT` instead. This avoids the server accumulating sockets in the `TIME_WAIT` state, allowing it to more effectively handle rapid reconnections from clients. This approach relies on clients being responsible for closing the connection, which can be implemented in most websocket clients. This shifts the `TIME_WAIT` burden to clients, where it is less likely to be a problem since clients do not typically have the same volume of connections.

Several strategies can mitigate these issues. One crucial adjustment is to change the application's logic so that the client actively closes the websocket connection. This shifts the burden of entering `TIME_WAIT` to the client side where the accumulation is less problematic. In scenarios where a server-side close is unavoidable, one must carefully examine any keep-alive mechanisms or timeout configurations to ensure a balance between responsiveness and socket exhaustion. Also consider scaling strategies such as using a load balancer that is more efficient at handling many connections or using a different architecture that uses UDP instead.

For further learning, I recommend focusing on resources describing TCP connection states (particularly `TIME_WAIT`), asynchronous networking patterns, operating system tuning for socket limits, and specific guides for the networking libraries used within Kotlin (such as Ktor documentation). Understanding the underlying network behavior will lead to more robust and scalable websocket applications. Consulting resources covering high-performance networking and systems programming will also prove invaluable.
