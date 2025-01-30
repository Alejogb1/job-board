---
title: "How can I display remote server logs in real-time within a Jenkins job?"
date: "2025-01-30"
id: "how-can-i-display-remote-server-logs-in"
---
Real-time display of remote server logs within a Jenkins job necessitates a solution that avoids polling, which is inefficient and introduces latency.  My experience troubleshooting similar integration issues across diverse platforms—from embedded systems to large-scale cloud deployments—points to a robust, push-based approach leveraging technologies like websockets or tailing the log file with a suitable tool, feeding that output to Jenkins.  This approach drastically reduces the overhead compared to regularly querying the server for log updates.

**1.  Clear Explanation:**

The optimal strategy involves establishing a persistent connection between the remote server and the Jenkins build environment.  Polling the server for log updates incurs unnecessary network traffic and delays, significantly impacting build times, especially when dealing with verbose logging.  Instead, we utilize a mechanism where the server actively pushes log entries to Jenkins as they are written.  This can be achieved using various technologies; however, I'll primarily focus on two:

* **Websockets:** This provides a full-duplex communication channel, enabling bidirectional data flow.  The remote server maintains a persistent websocket connection with a dedicated Jenkins plugin or a custom script running within the Jenkins job.  Log entries are sent as messages over this connection, facilitating real-time display.  This approach is particularly suitable for environments with high log volume and a need for immediate updates.

* **Log Tailing and Output Redirection:** This method leverages a command-line utility (like `tail -f`) on the remote server to monitor the log file for changes. The output of `tail -f` is then piped to a program that streams the data to Jenkins. This can be achieved through `ssh` with output redirection and piped to a Jenkins plugin designed for handling standard output streams in real-time or via a dedicated service like syslog, if the remote server supports it. This approach is simpler to implement than websockets for less demanding scenarios.


**2. Code Examples with Commentary:**

**Example 1: Using `tail -f` and `ssh` (simpler approach):**

```bash
#!/bin/bash
ssh user@remote_server "tail -f /path/to/log/file.log" | while read line; do
  echo "$line" >> jenkins_log.txt
  #Optional: Add code here to trigger Jenkins build steps based on log content, like alerting
done
```

This script uses `ssh` to connect to the remote server and executes `tail -f` to monitor the log file. The output is then piped to a `while` loop, which writes each received line to a local file (`jenkins_log.txt`) within the Jenkins workspace.  This file can then be displayed in Jenkins using a dedicated plugin designed for displaying text files. This method is straightforward, but the connection can be interrupted by network issues and requires frequent checks for the file changes within the Jenkins job.

**Example 2:  Implementing a custom WebSocket server (more complex, scalable approach):**

```java
//Illustrative Java snippet for a basic WebSocket server (using a suitable library like Spring Websocket)
//This requires a Jenkins plugin or custom script capable of connecting to the WebSocket endpoint.

import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LogWebSocketHandler extends TextWebSocketHandler {
    private List<WebSocketSession> sessions = new ArrayList<>();

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
      //Handle incoming messages (if needed)
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        sessions.add(session);
        // Start listening for log events on the server
        new Thread(() -> {
            try {
                // Add code to tail your log file and send it over the WebSocket.
                // Example:
                ProcessBuilder pb = new ProcessBuilder("tail", "-f", "/path/to/log/file.log");
                Process process = pb.start();
                java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()));
                String line;
                while ((line = reader.readLine()) != null){
                    for (WebSocketSession s : sessions) {
                        s.sendMessage(new TextMessage(line));
                    }
                }
            } catch (IOException e){
                // handle exception
            }
        }).start();
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        sessions.remove(session);
    }
}
```

This Java snippet demonstrates the core logic of a WebSocket server.  A suitable websocket library (such as Spring Websocket) would be required for a production-ready implementation.  This server listens for new connections and then, upon connection establishment, starts a background thread which continuously tails the log file. Each new log line is then broadcast to all connected clients (Jenkins in this case). This provides a scalable and robust solution compared to the simpler approach using `tail -f` and `ssh`.

**Example 3: Utilizing a Syslog Server (efficient for large-scale deployments):**

This approach involves configuring the remote server to send log entries to a centralized syslog server. A dedicated Jenkins plugin (or a custom script) can then subscribe to the syslog server to receive log updates.  The code example for this depends heavily on the specific syslog server and plugin used.  The core concept is that the remote server pushes logs to the syslog server, eliminating the need for the Jenkins job to actively pull log data.

```bash
# Example remote server configuration (rsyslog)
# Add this line to the rsyslog configuration file:
# *.* @@syslog-server-ip:514

# Jenkins plugin would need to receive and display logs from syslog-server-ip:514
```

This snippet shows a basic rsyslog configuration.  You'd need a corresponding Jenkins plugin (or a script) to read the logs from the syslog server.  This is efficient because the syslog server acts as a central point for log aggregation and distribution.

**3. Resource Recommendations:**

* Consult the documentation for your specific Jenkins version.
* Investigate available Jenkins plugins specializing in log aggregation and display.  Thoroughly examine their features, compatibility, and security implications.
* Familiarize yourself with various command-line tools relevant to log file manipulation, such as `tail`, `grep`, `awk`, and `sed`.
* Explore the documentation for chosen websocket library (if applicable).  Understand the nuances of establishing and maintaining WebSocket connections.
* Understand the syslog protocol and configuration options if choosing the syslog approach.  Consider different syslog implementations and their respective capabilities.  Security best practices should be followed for all networking aspects.


By implementing a push-based solution leveraging websockets, `tail -f` with `ssh`, or a syslog server, you can efficiently display remote server logs in real-time within your Jenkins jobs, providing crucial feedback and streamlining the continuous integration/continuous delivery (CI/CD) process.  Remember to prioritize security best practices throughout the implementation and appropriately handle potential errors and exceptions.  Choosing the most appropriate method depends heavily on your specific infrastructure, logging volumes and the required level of real-time interaction.
