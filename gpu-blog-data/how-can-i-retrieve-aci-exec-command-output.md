---
title: "How can I retrieve ACI exec command output in .NET?"
date: "2025-01-30"
id: "how-can-i-retrieve-aci-exec-command-output"
---
Retrieving the output of `exec` commands from Azure Container Instances (ACI) programmatically requires a multi-faceted approach, primarily leveraging the Azure SDK for .NET and asynchronous operations. I've frequently encountered this need when automating container health checks and data extraction processes in my own microservices architecture. The absence of a direct synchronous "get output" method necessitates a workflow involving command submission, status polling, and stream retrieval, demanding careful resource management and error handling.

Fundamentally, we don't directly execute commands and retrieve results like invoking a local process. Instead, we initiate a command within the ACI container, then monitor its execution state. Once the command completes, the standard output (stdout) and standard error (stderr) streams are made accessible for retrieval. The process revolves around the `Container` resource, specifically its `ExecuteCommandAsync` method, followed by examination of its associated `ContainerExecResponse` and subsequently, stream handling.

The initial step involves creating an instance of `Azure.ResourceManager.ContainerInstance.ContainerResource` using an existing container group resource ID and container name. This object provides the entry point for executing commands. The `ExecuteCommandAsync` method accepts a `ContainerExecRequest` object. This object defines the command to run within the container and an optional `TerminalSize` structure, although the latter is not strictly necessary for command output retrieval. The method returns a `Response<ContainerExecResponse>` encapsulating information about the command execution.

This response is critical for accessing the command's streams. The `ContainerExecResponse` contains properties such as `WebSocketUri`, `Password`, and `Error`, with `WebSocketUri` being the crucial piece for obtaining stdout and stderr output. This URI represents a websocket connection to the command's output stream.

To establish the websocket connection, the .NET framework's `System.Net.WebSockets` namespace is utilized. A `ClientWebSocket` is instantiated, and using the `WebSocketUri` extracted from the `ContainerExecResponse`, a connection is established asynchronously. The response `Password` is used as a subprotocol for authenticating the websocket connection. Once connected, the `ReceiveAsync` method of the `ClientWebSocket` allows the retrieval of data from the streams.

Data received through the websocket represents chunks of the command's output, which often require processing as byte arrays or strings. Since the stream can be fragmented, continuous reading from the socket is required until the command finishes and the socket is closed. The process must also handle potential websocket errors or socket disconnects during stream retrieval.

Below are three code examples demonstrating this process, each illustrating slightly different aspects of the workflow:

**Example 1: Basic command execution and stream retrieval**

```csharp
using Azure.Core;
using Azure.Identity;
using Azure.ResourceManager;
using Azure.ResourceManager.ContainerInstance;
using Azure.ResourceManager.ContainerInstance.Models;
using System.Net.WebSockets;
using System.Text;

public async Task<string> ExecuteAciCommand(string containerGroupName, string containerName, string command)
{
    var credential = new DefaultAzureCredential();
    var armClient = new ArmClient(credential);

    var containerGroupResourceId = new ResourceIdentifier($"/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ContainerInstance/containerGroups/{containerGroupName}");
    var containerGroup = armClient.GetContainerGroupResource(containerGroupResourceId);
    var container = containerGroup.GetContainerResource(containerName);

    var execRequest = new ContainerExecRequest(command);
    var execResponse = await container.ExecuteCommandAsync(execRequest);

    if (execResponse?.Value == null || string.IsNullOrEmpty(execResponse.Value.WebSocketUri))
    {
        Console.WriteLine($"Failed to execute command or invalid websocket URI: {execResponse?.Value?.Error?.Message}");
        return string.Empty;
    }

    using (var client = new ClientWebSocket())
    {
        client.Options.AddSubProtocol(execResponse.Value.Password);
        await client.ConnectAsync(new Uri(execResponse.Value.WebSocketUri), CancellationToken.None);

        var buffer = new byte[1024];
        var output = new StringBuilder();

        while (client.State == WebSocketState.Open)
        {
            var result = await client.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
            if(result.MessageType == WebSocketMessageType.Close){
                break;
            }

            if (result.Count > 0)
            {
              output.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
            }
        }
        return output.ToString();
    }
}
```

This example provides a foundational implementation for command execution and stream retrieval. It demonstrates essential steps, including establishing a websocket connection, receiving output chunks and concatenating them into a string and includes a basic check to handle null or empty `WebSocketUri` which may indicate command failures.  Error handling here is minimal, mainly logging a message if the initial command execution fails.

**Example 2: Enhanced error handling and stream differentiation**

```csharp
using Azure.Core;
using Azure.Identity;
using Azure.ResourceManager;
using Azure.ResourceManager.ContainerInstance;
using Azure.ResourceManager.ContainerInstance.Models;
using System.Net.WebSockets;
using System.Text;
using System.IO;
using System.Collections.Generic;

public async Task<(string StdOut, string StdErr)> ExecuteAciCommandWithStreams(string containerGroupName, string containerName, string command)
{
    var credential = new DefaultAzureCredential();
    var armClient = new ArmClient(credential);

    var containerGroupResourceId = new ResourceIdentifier($"/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ContainerInstance/containerGroups/{containerGroupName}");
    var containerGroup = armClient.GetContainerGroupResource(containerGroupResourceId);
    var container = containerGroup.GetContainerResource(containerName);

    var execRequest = new ContainerExecRequest(command);
    var execResponse = await container.ExecuteCommandAsync(execRequest);

    if (execResponse?.Value == null || string.IsNullOrEmpty(execResponse.Value.WebSocketUri))
    {
        Console.WriteLine($"Failed to execute command or invalid websocket URI: {execResponse?.Value?.Error?.Message}");
        return (string.Empty, string.Empty);
    }

    using (var client = new ClientWebSocket())
    {
        client.Options.AddSubProtocol(execResponse.Value.Password);
        try
        {
           await client.ConnectAsync(new Uri(execResponse.Value.WebSocketUri), CancellationToken.None);
        }
        catch (WebSocketException ex)
        {
             Console.WriteLine($"Failed to connect to websocket: {ex.Message}");
            return (string.Empty,string.Empty);
        }

        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        var buffer = new byte[1024];

         while (client.State == WebSocketState.Open)
        {
            WebSocketReceiveResult result;
            try{
                result = await client.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
            }
            catch(WebSocketException ex)
            {
                Console.WriteLine($"Error during websocket receive: {ex.Message}");
                break;
            }

            if(result.MessageType == WebSocketMessageType.Close){
                break;
            }

            if (result.Count > 0)
            {
               var message = Encoding.UTF8.GetString(buffer, 0, result.Count);

                if(message.StartsWith("1")){
                    stdout.Append(message.Substring(1)); // 1 marks stdout
                }
                else if(message.StartsWith("2")){
                    stderr.Append(message.Substring(1)); // 2 marks stderr
                }
                 else{
                       stdout.Append(message); //Default treat it as stdout if no prefix
                }
            }
        }
        return (stdout.ToString(), stderr.ToString());

    }
}
```

This example enhances the previous one by differentiating between standard output and standard error streams. Data streamed back from the websocket are prefixed with either "1" for stdout or "2" for stderr. The code now processes this prefix to route each message fragment accordingly, before stripping the prefix. Additionally, it includes a `try-catch` around the websocket connection attempt and the receive operation, offering better robustness against communication failures. This method returns a tuple, containing both stdout and stderr, making the analysis of command outcomes much more informative.

**Example 3: Using streams for large outputs and cancellation**

```csharp
using Azure.Core;
using Azure.Identity;
using Azure.ResourceManager;
using Azure.ResourceManager.ContainerInstance;
using Azure.ResourceManager.ContainerInstance.Models;
using System.Net.WebSockets;
using System.Threading;
using System.IO;

public async Task<Stream> ExecuteAciCommandAsStream(string containerGroupName, string containerName, string command, CancellationToken cancellationToken)
{
  var credential = new DefaultAzureCredential();
    var armClient = new ArmClient(credential);

    var containerGroupResourceId = new ResourceIdentifier($"/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ContainerInstance/containerGroups/{containerGroupName}");
    var containerGroup = armClient.GetContainerGroupResource(containerGroupResourceId);
    var container = containerGroup.GetContainerResource(containerName);

    var execRequest = new ContainerExecRequest(command);
    var execResponse = await container.ExecuteCommandAsync(execRequest);

     if (execResponse?.Value == null || string.IsNullOrEmpty(execResponse.Value.WebSocketUri))
    {
        Console.WriteLine($"Failed to execute command or invalid websocket URI: {execResponse?.Value?.Error?.Message}");
        return Stream.Null;
    }

    var client = new ClientWebSocket();
    client.Options.AddSubProtocol(execResponse.Value.Password);
    try
    {
        await client.ConnectAsync(new Uri(execResponse.Value.WebSocketUri), cancellationToken);
    }
     catch (WebSocketException ex)
        {
             Console.WriteLine($"Failed to connect to websocket: {ex.Message}");
            client.Dispose();
            return Stream.Null;
        }

    return new WebSocketStream(client, cancellationToken);

}

public class WebSocketStream : Stream{
    private ClientWebSocket _client;
    private CancellationToken _cancellationToken;

    public WebSocketStream(ClientWebSocket client, CancellationToken cancellationToken)
    {
        _client = client;
        _cancellationToken = cancellationToken;
    }
    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => throw new NotSupportedException();
    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
    public override void Flush() => throw new NotSupportedException();
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

    public override async Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
    {
         if (_client.State != WebSocketState.Open)
            {
                return 0;
            }

           try
            {
              var result = await _client.ReceiveAsync(new ArraySegment<byte>(buffer, offset, count), _cancellationToken);
              if(result.MessageType == WebSocketMessageType.Close){
                  _client.Abort();
                  return 0;
              }
                return result.Count;
            }
              catch (WebSocketException ex)
            {
               Console.WriteLine($"Error during websocket receive: {ex.Message}");
               _client.Abort();
               return 0;
            }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
              if(_client !=null){
                 _client.Dispose();
              }
        }
        base.Dispose(disposing);
    }
}
```

This final example focuses on handling large command outputs more efficiently by returning a `Stream`. Instead of accumulating the entire output in memory, it provides a custom stream wrapper around the websocket, allowing the data to be consumed incrementally. This allows processing of large result sets without memory exhaustion. Furthermore, the code takes a `CancellationToken`, enhancing control over the operation and allowing cancellation if required. This approach is most suitable for scenarios where only parts of the stream are needed or large output requires more sophisticated parsing.

For further exploration, I recommend consulting the official Azure SDK documentation for Container Instance, specifically the `Azure.ResourceManager.ContainerInstance` namespace.  Additionally, delving into the `System.Net.WebSockets` namespace documentation on Microsoft Learn is beneficial for deeper understanding of websocket programming in .NET.  Finally, studying best practices for asynchronous programming in .NET is highly advised, especially concerning resource management with the `using` statement and efficient stream handling.
