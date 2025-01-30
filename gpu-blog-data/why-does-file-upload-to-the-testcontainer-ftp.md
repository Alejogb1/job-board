---
title: "Why does file upload to the Testcontainer FTP server fail with 'Connection refused' after a connection is established?"
date: "2025-01-30"
id: "why-does-file-upload-to-the-testcontainer-ftp"
---
The "Connection refused" error during an FTP upload to a Testcontainers-managed FTP server, specifically after a successful initial connection, often points to a passive mode configuration issue within the FTP protocol itself, particularly concerning how the server reports its data port range and how the client subsequently interprets it. I've frequently encountered this during integration tests involving file uploads where a standard FTP client library interacts with an embedded FTP server started with Testcontainers.

The core of the problem lies in the way FTP handles data transfers. FTP operates using two separate connections: a control connection (typically on port 21), used for commands and responses, and a data connection, used for the actual file transfer. While active mode dictates the client initiates the data connection to the server, the more common passive mode (PASV command) is used when a client is behind a firewall or NAT. The FTP server opens a random port for the data connection, communicates this port to the client through the control connection, and the client then attempts to establish the data connection on this specified port.

In the Testcontainers context, the embedded FTP server runs inside a Docker container, which is essentially a virtualized network environment. The server, upon receiving the PASV command, returns a port number that is valid *within* its own container’s network namespace. The client, however, running on the host machine, cannot directly connect to this port; it's effectively an internal IP address and port within the container that isn't routed to the host. This mismatch leads the client to attempt a connection to a non-existent address, resulting in the "Connection refused" error, despite the initial successful login. Testcontainers does not automatically forward these random data ports, nor is it feasible to predict and expose every conceivable data port.

Therefore, the critical step is to configure either the FTP server, the client, or both, to accommodate this difference. The most robust and common approach, in my experience, is to explicitly control the range of passive ports used by the FTP server and configure Testcontainers to map these host ports to the container ports. This allows the client to connect to the host-exposed ports and the Testcontainers infrastructure to correctly route that to the internal port inside the container. Another option, less flexible but sometimes necessary, is to force active mode on the client if direct host connection to the container network is viable and the server is configured to accept active mode connections, a less common setup.

Let's examine three code examples illustrating these points, focusing on passive mode configuration. The first demonstrates a basic setup without proper port mapping, leading to failure. The second configures the FTP server to use a specific passive port range, and Testcontainers to map those ports. The third showcases how to handle passive mode within a standard FTP client in Java.

**Example 1: Failing Test Setup - Inadequate Port Mapping**

```java
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;
import java.io.IOException;
import java.net.SocketException;
import org.apache.commons.net.ftp.FTPClient;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class FtpTestFailing {

    @Test
    public void testFtpUploadFails() throws IOException {

        GenericContainer<?> ftpContainer = new GenericContainer<>(DockerImageName.parse("delftware/alpine-ftp:latest"))
                                        .withExposedPorts(21); // Only exposing control port

        ftpContainer.start();
        String ftpAddress = ftpContainer.getHost();
        int ftpPort = ftpContainer.getMappedPort(21);
        
        FTPClient ftpClient = new FTPClient();
        
        try {
             ftpClient.connect(ftpAddress, ftpPort);
             ftpClient.login("test", "test");
             ftpClient.enterLocalPassiveMode(); // Client requests Passive Mode

             assertThrows(SocketException.class, () -> ftpClient.storeFile("test.txt", getClass().getResourceAsStream("/test.txt")));

         }
         finally{
             if (ftpClient.isConnected()) {
                 ftpClient.disconnect();
             }
             ftpContainer.stop();
         }
    }
}
```

This example demonstrates a common mistake: only exposing the control port. The container is started and the FTP client attempts a file upload. However, when the `storeFile()` method is called, a `SocketException` due to "Connection refused" arises, indicating that the data connection in passive mode failed because no port was exposed from the dynamic range assigned within the container by the server. The critical flaw is the absence of port mapping of any passive data ports.

**Example 2: Successful Test Setup – Specific Port Range Mapping**

```java
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;
import java.io.IOException;
import java.io.InputStream;
import java.net.SocketException;
import org.apache.commons.net.ftp.FTPClient;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;


public class FtpTestSuccess {
    @Test
    public void testFtpUploadSuccess() throws IOException {
        int passivePortMin = 40000;
        int passivePortMax = 40010;

        GenericContainer<?> ftpContainer = new GenericContainer<>(DockerImageName.parse("delftware/alpine-ftp:latest"))
                 .withExposedPorts(21, passivePortMin, passivePortMax)
                 .withEnv("FTP_PASSIVE_PORTS", passivePortMin + "-" + passivePortMax);


        ftpContainer.start();

        String ftpAddress = ftpContainer.getHost();
        int ftpPort = ftpContainer.getMappedPort(21);

        FTPClient ftpClient = new FTPClient();

        try {
            ftpClient.connect(ftpAddress, ftpPort);
            ftpClient.login("test", "test");
            ftpClient.enterLocalPassiveMode();
            InputStream inputStream = getClass().getResourceAsStream("/test.txt");
           Assertions.assertTrue(ftpClient.storeFile("test.txt", inputStream));
         }
        finally{
            if (ftpClient.isConnected()) {
                ftpClient.disconnect();
            }
            ftpContainer.stop();
        }
    }
}

```
In this second example, the FTP server within the container is explicitly configured to use a specific passive port range by setting the `FTP_PASSIVE_PORTS` environment variable. Additionally, these ports (40000 to 40010, inclusive) are explicitly exposed using `withExposedPorts` within the Testcontainers setup. This maps the host ports to the container’s internal ports, allowing the client to correctly initiate the data connection. The file upload proceeds successfully as the client’s request via the PASV command is now directed to an exposed port and routable to the server’s specified internal port. The test assertion verifies the successful file store operation and highlights the difference between the failing example.

**Example 3: FTP Client Passive Mode Configuration**

```java
import org.apache.commons.net.ftp.FTPClient;
import org.apache.commons.net.ftp.FTPReply;
import java.io.IOException;
import java.io.InputStream;
import java.net.SocketException;
import java.net.SocketTimeoutException;

public class FtpClientUtil {

    public static boolean uploadFile(String server, int port, String user, String password, String fileName, InputStream fileStream) throws IOException {
        FTPClient ftpClient = new FTPClient();
        boolean success = false;

            try {
             ftpClient.connect(server, port);
             if(!FTPReply.isPositiveCompletion(ftpClient.getReplyCode())){
                 ftpClient.disconnect();
                 throw new IOException("Error connecting to ftp server");
             }
              
            boolean loggedIn=ftpClient.login(user, password);
            if(!loggedIn){
                  ftpClient.disconnect();
                  throw new IOException("Login failed");
            }


            ftpClient.enterLocalPassiveMode();
              success = ftpClient.storeFile(fileName, fileStream);


             if(!success){
                  throw new IOException("File transfer failed");

              }

        } finally {
                if (ftpClient.isConnected()) {
                    try{
                    ftpClient.logout();
                    }catch(SocketTimeoutException ignored){
                        // ignored
                    }
                    ftpClient.disconnect();
                }
            }
    return success;

    }
}

```

This third example provides a reusable method to upload a file to an FTP server. It demonstrates correct usage of the `enterLocalPassiveMode` method on the FTPClient. The method encapsulates the connection, login and upload operations and disconnection in a try-finally block, ensuring proper resource management and handling potential `IOException`. The explicit error handling and checks demonstrate a more robust client approach than might be implemented within simple unit test scenarios.

In summary, the core issue causing "Connection refused" after an initial connection with an FTP server running in Testcontainers stems from the network isolation of Docker containers and the nature of FTP’s passive mode. By properly configuring the FTP server to use a specific range of passive ports and subsequently exposing these ports in Testcontainers, you enable a client running on the host to correctly access the internal ports within the container. The client side must also use passive mode to facilitate a data transfer. Employing methods such as the one shown above facilitates robust FTP interactions across varied hosting environments.

For further study, I recommend consulting the official documentation for Testcontainers, which details various options for networking and port mapping. Additionally, review the Apache Commons Net library documentation for a thorough understanding of the FTP client implementation and the nuances of passive and active mode operation. Understanding the interplay between container networking and the FTP protocol is essential for robust automated testing and real world deployments with this technology.
