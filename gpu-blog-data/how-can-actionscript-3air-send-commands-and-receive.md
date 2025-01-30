---
title: "How can ActionScript 3/AIR send commands and receive replies?"
date: "2025-01-30"
id: "how-can-actionscript-3air-send-commands-and-receive"
---
Within the Adobe AIR and Flash Player runtime, establishing bidirectional communication – the ability to send commands and receive replies – between different parts of an application or even between an application and an external service requires careful consideration of several mechanisms. ActionScript 3 (AS3), the language powering this environment, provides options primarily revolving around network communication or inter-application communication via AIR's NativeProcess API. My experience working on a large-scale digital signage platform using AIR heavily relied on a robust command and reply architecture, highlighting the importance of selecting the appropriate approach for a given scenario. I've found that `XMLSocket`, `URLLoader` (in conjunction with server-side processing), and `NativeProcess` offer distinct advantages and disadvantages, impacting the overall architecture of the system. I will detail these, focusing on implementation and application.

The core challenge lies in implementing a system that not only dispatches a command with associated data, but also reliably receives a response, processes it, and potentially updates the application state. Each method has limitations and best uses. For instance, `XMLSocket` is particularly suitable for real-time bidirectional communication. It allows a persistent, socket-based connection to be established with a server, enabling instantaneous transmission and reception of text-based data. This is advantageous in applications requiring immediate updates or server-pushed events, like a chat application or a shared collaborative tool. The limitation is that it’s specifically a socket-based approach, and often requires the server to be designed around socket communication as well.

`URLLoader`, on the other hand, provides a more traditional request-response pattern. A request is sent to a specific URL, usually to a server-side script, and the server responds with data. This works well in situations where you're making API calls to external services, or handling data persistence to a backend. Unlike XMLSocket, `URLLoader` is inherently stateless; each request is treated independently. You’d have to implement a system for correlating request-response pairs in more complex scenarios. This method is generally more versatile, due to its broad compatibility with standard web protocols.

`NativeProcess` provides a fundamentally different capability. It allows an AIR application to spawn a separate, native process. This means you can communicate with completely different applications or scripts, using pipes for input/output. This approach shines when the core application needs to interact with external command-line tools or needs to delegate CPU-intensive tasks to another process. However, its setup is significantly more complex than other methods, and introduces the challenges of managing separate processes, ensuring proper security context, and handling inter-process communication.

Let's explore how these methods can be implemented. Starting with `XMLSocket`, a basic example of sending a command and receiving a response can be implemented as follows:

```actionscript
import flash.net.XMLSocket;
import flash.events.Event;
import flash.events.ProgressEvent;
import flash.utils.ByteArray;

public class XMLSocketClient
{
    private var socket:XMLSocket;
    private var commandQueue:Array = [];

    public function XMLSocketClient(host:String, port:int)
    {
        socket = new XMLSocket();
        socket.addEventListener(Event.CONNECT, onConnect);
        socket.addEventListener(Event.CLOSE, onClose);
        socket.addEventListener(ProgressEvent.SOCKET_DATA, onSocketData);
        socket.connect(host, port);
    }

     public function sendCommand(command:String, data:Object):void {
        var encodedData:String = JSON.stringify({command:command, data:data});
        commandQueue.push(encodedData); // Queue for potential reconnection
        if(socket.connected) {
            socket.send(encodedData + "\0"); // Use a null terminator to mark message end
        }
     }

    private function onConnect(event:Event):void {
        trace("Connected to server");
        // Send queued commands if any
        while(commandQueue.length > 0) {
            socket.send(commandQueue.shift() + "\0");
        }
    }

     private function onClose(event:Event):void
    {
        trace("Socket closed");
        // Handle socket closure logic (e.g., try reconnecting)
    }

    private function onSocketData(event:ProgressEvent):void {
      var receivedData:String = socket.readUTFBytes(socket.bytesAvailable);
      var messages:Array = receivedData.split("\0"); // Split messages delimited by null terminator
      for(var i:int = 0; i < messages.length - 1; i++) { // Skip empty string at the end
         if(messages[i].length > 0){
           handleResponse(messages[i]);
         }
      }
    }

     private function handleResponse(data:String):void
    {
        var parsedData:Object = JSON.parse(data);
        trace("Response received: " + parsedData.command + " , " + parsedData.data);

        //  Application specific logic ( update ui based on received command )
    }

    public function close():void {
         if (socket.connected) {
            socket.close();
        }
    }
}
```

In this example, the `XMLSocketClient` establishes a connection and sends JSON-encoded commands. The null byte serves as a delimiter when there may be multiple messages being transmitted in a short time frame. This ensures that complete messages are processed. I am using JSON for serializing complex data and `readUTFBytes` with a `\0` delimiter because XMLSocket does not provide native support for JSON or multi-message processing. The `handleResponse` function is where you'd implement the application-specific logic of how to react to different commands or data.

Moving on to `URLLoader`, consider this implementation:

```actionscript
import flash.net.URLLoader;
import flash.net.URLRequest;
import flash.net.URLRequestMethod;
import flash.net.URLVariables;
import flash.events.Event;
import flash.events.IOErrorEvent;
import flash.events.SecurityErrorEvent;

public class APIClient
{
     private var loader:URLLoader;

    public function APIClient()
    {
    }

    public function sendCommand(command:String, data:Object, endpoint:String):void {
        loader = new URLLoader();
        var request:URLRequest = new URLRequest(endpoint);
        request.method = URLRequestMethod.POST;
        var variables:URLVariables = new URLVariables();
        variables.command = command;
        variables.data = JSON.stringify(data); // Serialize for url encoding.
        request.data = variables;
         loader.addEventListener(Event.COMPLETE, onComplete);
        loader.addEventListener(IOErrorEvent.IO_ERROR, onIOError);
        loader.addEventListener(SecurityErrorEvent.SECURITY_ERROR, onSecurityError);

        try
        {
            loader.load(request);
        }
        catch (error:Error)
        {
            trace("Error loading url : " + error.message);
        }
    }

    private function onComplete(event:Event):void
    {
        try
        {
            var response:Object = JSON.parse(loader.data);
            trace("Response received: " + response.command + " , " + response.data);

            // Handle response as needed.
        }
        catch (parseError:Error) {
             trace("Error parsing response: " + parseError.message);
        }

        loader.removeEventListener(Event.COMPLETE, onComplete);
        loader.removeEventListener(IOErrorEvent.IO_ERROR, onIOError);
        loader.removeEventListener(SecurityErrorEvent.SECURITY_ERROR, onSecurityError);
    }

     private function onIOError(event:IOErrorEvent):void {
        trace("IO Error : " + event.text);
         loader.removeEventListener(Event.COMPLETE, onComplete);
         loader.removeEventListener(IOErrorEvent.IO_ERROR, onIOError);
         loader.removeEventListener(SecurityErrorEvent.SECURITY_ERROR, onSecurityError);
    }

     private function onSecurityError(event:SecurityErrorEvent):void {
         trace("Security Error : " + event.text);
         loader.removeEventListener(Event.COMPLETE, onComplete);
         loader.removeEventListener(IOErrorEvent.IO_ERROR, onIOError);
         loader.removeEventListener(SecurityErrorEvent.SECURITY_ERROR, onSecurityError);
    }
}
```
This `APIClient` uses `URLLoader` to make a POST request, sending a command and data as URL-encoded variables. The server-side script at the specified `endpoint` should interpret this data and return a JSON response, which is then parsed in `onComplete`. Error handling is crucial here, as network requests can fail for many reasons, which are addressed by the event listeners.

Finally, consider the `NativeProcess` approach. This is far more involved, but can be crucial in specific use cases. For a demonstration, the example below will send a command to a process, which will then send it back:

```actionscript
import flash.desktop.NativeProcess;
import flash.desktop.NativeProcessStartupInfo;
import flash.events.NativeProcessExitEvent;
import flash.events.ProgressEvent;
import flash.filesystem.File;
import flash.utils.ByteArray;

public class ProcessCommunicator
{
    private var process:NativeProcess;
    private var stdOutBuffer:ByteArray;
    private var stdErrBuffer:ByteArray;

    public function ProcessCommunicator(processPath:String)
    {
        var processFile:File = new File(processPath);
        var startupInfo:NativeProcessStartupInfo = new NativeProcessStartupInfo();
        startupInfo.executable = processFile;

        process = new NativeProcess();
        process.addEventListener(NativeProcessExitEvent.EXIT, onProcessExit);
        process.addEventListener(ProgressEvent.STANDARD_OUTPUT_DATA, onStdOutData);
        process.addEventListener(ProgressEvent.STANDARD_ERROR_DATA, onStdErrData);
        process.start(startupInfo);
        stdOutBuffer = new ByteArray();
        stdErrBuffer = new ByteArray();

    }

    public function sendCommand(command:String, data:Object):void {
         var encodedData:String = JSON.stringify({command:command, data:data});
        var dataToSend:ByteArray = new ByteArray();
        dataToSend.writeUTFBytes(encodedData);
        process.standardInput.writeBytes(dataToSend);
        process.standardInput.close();
    }

    private function onProcessExit(event:NativeProcessExitEvent):void
    {
        trace("Process exited with code: " + event.exitCode);
        process.removeEventListener(NativeProcessExitEvent.EXIT, onProcessExit);
        process.removeEventListener(ProgressEvent.STANDARD_OUTPUT_DATA, onStdOutData);
        process.removeEventListener(ProgressEvent.STANDARD_ERROR_DATA, onStdErrData);

    }

    private function onStdOutData(event:ProgressEvent):void {
        stdOutBuffer.writeBytes(process.standardOutput,0, process.standardOutput.bytesAvailable);
        var data:String = stdOutBuffer.readUTFBytes(stdOutBuffer.length);
         try{
             var parsedData:Object = JSON.parse(data);
             trace("Native process Output: " + parsedData.command + " , " + parsedData.data);
         } catch (error:Error) {
           trace("Process Output :" + data);
         }
           stdOutBuffer.clear();
    }

    private function onStdErrData(event:ProgressEvent):void
    {
          stdErrBuffer.writeBytes(process.standardError,0,process.standardError.bytesAvailable);
           trace("Process Error : " + stdErrBuffer.readUTFBytes(stdErrBuffer.length));
          stdErrBuffer.clear();
    }


    public function close():void {
       if(process != null){
          process.exit();
       }
    }

}
```

This code instantiates a `NativeProcess`, establishes event listeners for standard output/error, and then sends data via standard input. The separate native process needs to listen to standard input, read UTF-8 encoded JSON data, and then write it back to its standard output. In practice, a server application or script that does something based on the received command would be utilized in the process. The important aspect here is data must be properly serialized. The example assumes the process echoes back the same data. Error handling is critical as launching external processes can present many challenges.

For additional learning, resources from Adobe's developer documentation on ActionScript 3, covering `XMLSocket`, `URLLoader`, and `NativeProcess`, are extremely beneficial. The official language reference is indispensable for understanding the intricacies of these classes and their event models. Additionally, articles and blog posts detailing best practices for communication patterns in AS3/AIR applications offer a wealth of practical information. Specifically, resources focusing on asynchronous programming techniques would provide further benefit. Utilizing a test application that can listen on the chosen communication channel while developing your application is also an effective strategy to ensure proper command/response logic.
