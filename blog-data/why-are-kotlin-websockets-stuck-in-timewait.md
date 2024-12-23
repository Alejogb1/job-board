---
title: "Why are Kotlin websockets stuck in TIME_WAIT?"
date: "2024-12-23"
id: "why-are-kotlin-websockets-stuck-in-timewait"
---

Alright, let's tackle this. I've definitely seen my share of websocket issues over the years, and the dreaded TIME_WAIT state is a classic head-scratcher. It's a common situation, especially when dealing with abrupt disconnections or poorly managed connection lifecycles. Seeing your Kotlin websockets stuck there is often a symptom of the TCP protocol doing its job – albeit sometimes inconveniently from our perspective – and there are a few common culprits in the Kotlin ecosystem. Let's break it down.

The TIME_WAIT state, at its core, isn’t really about Kotlin specifically; it's a standard part of TCP's connection closing handshake. When a TCP connection is closed, the side that initiated the close (the one that sent the *FIN* packet first) usually enters TIME_WAIT. This waiting period exists for a couple of crucial reasons, chiefly, to ensure that delayed or re-ordered packets from the connection have time to be fully delivered and processed. Without it, we risk data loss or corruption when a new connection is established on the same port, and we risk receiving fragments from the previous connection.

Now, why this becomes prominent with Kotlin websockets has to do with how we manage these connections, especially when combined with a library. Typically, we would be using a library like Ktor or similar, and it is likely their implementation may have a specific behavior around connection closing.

One common scenario is the "client-side initiated close". Imagine your Kotlin websocket client, maybe an Android application or some server-side process, loses its network connection abruptly. The server, on its end, will eventually detect the disconnection, and initiate a clean close, likely sending a *FIN* packet. However, your client, which initiated the close attempt, and therefore the subsequent close on the server side, now enters TIME_WAIT. Because the OS has allocated a port for that connection, it needs to be held for a while to prevent collision with a new connection on the same port. If the client initiates a new websocket connection *very quickly*, it will use a new local port, often preventing a direct collision. However, the port on the server will remain in TIME_WAIT for the predefined timeout.

Another less frequent case is when your server-side Kotlin websocket code isn’t properly closing the connection. Perhaps an exception during the close procedure is silently swallowed, or resources are not de-allocated correctly. The server may never fully send the *FIN*, which causes the client to wait indefinitely. Less frequently, but possible, is that the client may not be responding correctly to the *FIN* packet, hence the socket stays in TIME_WAIT. It's important that the closing of the connection happens in both sides, and each side needs to correctly receive the acknowledge package from the other side.

Finally, network congestion can amplify the problem, since the delayed packets take longer to arrive, so the TIME_WAIT period will be extended. Network configuration on your firewall or load balancer might also play a role in exacerbating these issues.

So, what can you do about it? You can’t eliminate TIME_WAIT, since it is essential for TCP. However, you can mitigate the number of sockets stuck in TIME_WAIT, or avoid causing them. Here are a few concrete solutions based on my experience:

**1. Ensure Clean Disconnect Handling:** This is critical. Make sure your websocket library is properly handling connection closes and that both the client and server are participating in the close process correctly. The server should close the socket only after the client initiates the close, or after a timeout if the client is unresponsive.

   Here's an example snippet of a client-side code that is trying to handle the closure more robustly, which could be part of your kotlin project using a hypothetical `WebsocketClient` class that implements `Closeable`:

   ```kotlin
    import kotlinx.coroutines.*
    import java.io.Closeable
    import java.util.concurrent.atomic.AtomicBoolean


    class WebsocketClient(private val url: String) : Closeable {

        private var session : WebsocketSession? = null
        private var isOpen = AtomicBoolean(false)

        suspend fun connect() {
            if (isOpen.get()) return
            session = connectToWebsocket(url)
            isOpen.set(true)
            // ... continue processing incoming messages...
        }

        override fun close() {
            if (!isOpen.getAndSet(false)) return
            runBlocking {
                session?.close()
            }
            session = null
        }

        suspend fun sendMessage(message: String) {
            session?.send(message)
        }

        private suspend fun connectToWebsocket(url: String) : WebsocketSession {
            // Implementation of actual websocket connection, using a library such as ktor-client
            // For illustrative purposes:  assume it is a suspending function that returns a WebsocketSession
            return websocketSessionDummy()
        }

        private fun websocketSessionDummy() : WebsocketSession {
          return object: WebsocketSession {
            override suspend fun send(message: String) = println("Sending: $message")
            override suspend fun close() = println("Closing connection")
            override suspend fun receive() : String = ""
          }
        }
        interface WebsocketSession {
          suspend fun send(message: String)
          suspend fun close()
          suspend fun receive() : String
        }
    }

    fun main() = runBlocking {

        val client = WebsocketClient("ws://test-url/socket")
        client.connect()
        client.sendMessage("hello")
        delay(1000)
        client.close()
    }
    ```

   This example provides a basic structure for handling the lifecycle of a websocket connection within a `WebsocketClient` class, ensuring that closing procedures are explicitly managed. Note how the `close()` method makes sure that the socket is only closed if it is open, and it also uses a flag to prevent multiple closes at the same time.

**2. Graceful Shutdown Procedures:** Instead of simply killing the process or closing the connection abruptly, introduce logic to gracefully close the websocket. This often involves sending a "goodbye" message to the server, or using a predefined close code with the `close()` function in your websocket library.

    ```kotlin
    import io.ktor.client.*
    import io.ktor.client.plugins.websocket.*
    import io.ktor.websocket.*
    import kotlinx.coroutines.*
    import java.util.concurrent.atomic.AtomicBoolean

    suspend fun gracefulClose() {
       val client = HttpClient {
            install(WebSockets)
        }

        val url = "ws://echo.websocket.events"
        var session: WebSocketSession? = null
        var isOpen = AtomicBoolean(false)

        try {
            client.webSocket(url = url) {
              session = this
              isOpen.set(true)
              println("Connected to $url")
              // Some kind of receive loop here (not implemented)
             }
        } catch(e : Exception) {
          println("error $e")
        } finally {
             println("Closing gracefully")
            session?.close(CloseReason(CloseReason.Codes.NORMAL, "Client closing"))
            isOpen.set(false)
            client.close()
        }

        delay(1000) //keep the process alive to observe the console output

    }


    fun main() = runBlocking {
        gracefulClose()
    }
    ```

    This code shows how we gracefully close the websocket connection using ktor's `close` method, with a custom message that will be sent to the server. This might help in scenarios where you need to manage different disconnection scenarios.

**3. TCP Keep-Alive Settings:** Consider adjusting the TCP keep-alive settings on your server or client. While this won’t directly prevent TIME_WAIT, setting a smaller timeout might help with faster detection of broken connections, which will ultimately lead to a smaller number of TIME_WAIT sockets. However, proceed with caution as it can introduce other unwanted side effects if not tuned properly for your use case.

   This snippet below is a configuration of ktor, adding the TCP keep alive on the client configuration.

   ```kotlin
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.websocket.*
import io.ktor.network.sockets.*
import kotlinx.coroutines.*

    suspend fun configureTcpKeepAlive() {
        val client = HttpClient(CIO) {
            install(WebSockets)
            engine {
              socketTimeout = 10000 // milliseconds
              configure {
                tcp {
                    keepAlive = true
                }
              }
            }
        }
        val url = "ws://echo.websocket.events"
        try {
            client.webSocket(url = url) {
                println("Connected to $url")
                // Some kind of receive loop here (not implemented)
                delay(1000)

             }
        } catch (e : Exception) {
          println("error $e")
        } finally {
            client.close()
        }
    }

    fun main() = runBlocking {
        configureTcpKeepAlive()
    }

   ```

   This snippet demonstrates the configuration of a Ktor HTTP client with a WebSocket plugin, where we set the TCP `keepAlive` property in the underlying CIO engine. Adjusting values like the `socketTimeout` and other networking-related options can affect how sockets are handled, and therefore the number of TIME_WAIT sockets seen.

It's worth noting that diagnosing these kinds of issues requires close observation of your server logs and network traffic with tools like `tcpdump` or Wireshark. Also, if you are working with a cloud provider or managed server, there might be platform-specific considerations when dealing with timeouts and network configurations, especially on load balancers or API gateways.

For further reading, I recommend looking into *TCP/IP Illustrated, Volume 1* by W. Richard Stevens and *Unix Network Programming, Volume 1* by the same author. These books provide an in-depth understanding of the TCP protocol and its intricacies. Additionally, the official documentation for your specific websocket library (e.g., Ktor) is always a valuable resource.

In conclusion, while TIME_WAIT is a standard TCP mechanism, it can become a problem with websockets if not handled properly. Careful management of connection lifecycles and a robust approach to closing connections in both server-side and client-side code are key to minimizing issues. It's often not just about the code, but also the underlying network and OS configurations.
