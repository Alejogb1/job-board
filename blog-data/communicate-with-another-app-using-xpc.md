---
title: "communicate with another app using xpc?"
date: "2024-12-13"
id: "communicate-with-another-app-using-xpc"
---

so you want to get two apps chatting using XPC right I get it Been there done that probably even bought the t-shirt

Let me tell you XPC can be a bit of a trip at first its this inter process communication thing Apple throws at you It's not exactly a walk in the park but once you wrap your head around it it's pretty powerful stuff I remember my first dance with XPC it was a total headache I was trying to get a helper app to do some heavy lifting like image processing so my main app wouldn't get bogged down and become unresponsive I was so new to it I was trying to pass complex objects directly through the connection like some kind of naive newbie it failed spectacularly crashes left and right memory leaks galore good times

Anyway enough reminiscing lets break this down because I know that feel of staring at the screen wondering what on earth you are doing wrong

So XPC is all about creating services these services are basically like separate programs that run in their own sandbox and they can be called from other applications to do things You're not directly passing pointers or objects between apps you're sending messages that get serialized and deserialized that's the important bit

There are a few key players in this drama you've got your main application you know the app that everyone sees and then there's the XPC service itself a background process doing all the work the service provides an interface that your main app can call and they communicate using mach ports behind the scenes you don't need to care much about those unless you really want to dig into the deep end of IPC

Here's the basic gist of how you set it up I'll give you some Swift-ish examples that should get you going I'm assuming you are doing it for macOS or iOS because XPC on other systems is a different beast

First you need to create your service target In Xcode you add a new target select "macOS" or "iOS" and then choose XPC Service I normally call it something like YourAppNameHelper it gives you some starter code

Inside that XPC service target you will have a `YourServiceName.swift` or whatever you chose for it where you'll define your interface using a protocol conforming to `NSXPCListenerDelegate` inside you will define functions the main app can call it will look something like this

```swift
import Foundation

@objc protocol MyServiceProtocol {
  func doSomethingWith(string: String, reply: @escaping (String) -> Void)
  func processData(data: Data, completion: @escaping (Data?, Error?) -> Void)
}

class MyService: NSObject, MyServiceProtocol, NSXPCListenerDelegate{

    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection newConnection: NSXPCConnection) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: MyServiceProtocol.self)
        newConnection.exportedObject = self
        newConnection.resume()
        return true
    }
    func doSomethingWith(string: String, reply: @escaping (String) -> Void) {
    let result = string + " processed!"
    reply(result)
    }
    func processData(data: Data, completion: @escaping (Data?, Error?) -> Void) {
        // Simulate some data processing
      DispatchQueue.global(qos: .background).async {
        Thread.sleep(forTimeInterval: 0.5) // Simulate some work
          let processedData = Data(data.reversed())
            completion(processedData,nil)
      }
    }
}

let serviceDelegate = MyService()
let listener = NSXPCListener.service()
listener.delegate = serviceDelegate
listener.resume()

```

 that's our service you've got the protocol you've got the implementation which I have been very brief for example purposes don't do that in real code make it more robust you also have the listener that's waiting for connection requests it's pretty straightforward now over on your main app where you will actually call that service first thing you do you have to create an NSXPCConnection to the service

Here is how your main app can connect and call to the service remember to replace `com.yourcompany.YourAppName.YourAppNameHelper` with the bundle ID of your XPC service that's important

```swift
import Foundation
import AppKit //or UIKit depending on the platform

class MyClient {
  private var connection: NSXPCConnection?
  private var myService: MyServiceProtocol?

  func connectToService() {
    connection = NSXPCConnection(serviceName: "com.yourcompany.YourAppName.YourAppNameHelper") //replace this
    connection?.remoteObjectInterface = NSXPCInterface(with: MyServiceProtocol.self)
    connection?.invalidationHandler = { [weak self] in
        print("Connection invalidated")
        self?.connection = nil
        self?.myService = nil
    }
    connection?.resume()
      myService = connection?.remoteObjectProxy as? MyServiceProtocol
    print("connection established and ready to send messages")

  }
  
    func sendMessage() {
      guard let myService = myService else {
      print("no connection to service yet, call connectToService")
          return
      }
        myService.doSomethingWith(string: "Hello from main app") { result in
        print("Received result from service: \(result)")
      }
   }
    func sendData() {
         guard let myService = myService else {
      print("no connection to service yet, call connectToService")
          return
      }
        let myData = Data(bytes: [0x01, 0x02, 0x03, 0x04])
        myService.processData(data: myData) { processedData, error in
              if let error = error {
                print("Error processing data \(error)")
                return
              }
              if let processedData = processedData {
                 print("processed data is \(processedData)")
              }
         }
    }
}

 let client = MyClient()
client.connectToService()
client.sendMessage()
client.sendData()

```

In this main app example I first create a connection to the service then I'm using the `remoteObjectProxy` it's almost like calling a function on a remote computer magic right? and yes it does have a very long name I know I've heard that a million times

You'll notice we're using blocks or closures they're the only way to get data back from the service XPC is asynchronous so you can't just call a function and expect an immediate return that's how it avoids blocking the main thread remember my previous problem of blocking it was a bad bad experience

Now the tricky part is passing data around you can pass simple types like `String` `Int` `Data` and things that conform to the `NSSecureCoding` protocol if you're passing more complex objects you'll need to either make them conform to `NSSecureCoding` or serialize them to `Data` and back You can use something like `JSONEncoder` and `JSONDecoder` to do that or you could use something like protocol buffers if you're fancy and want a lot of performance I've been there before using JSON for large image files it was not my best coding moment for sure it became slow as snails

Let me give you an example where we are passing and returning a custom class it's a bit more involved we need to make the object conform to NSSecureCoding

```swift
// Custom class needs to conform to NSSecureCoding
class MyCustomObject: NSObject, NSSecureCoding {
    static var supportsSecureCoding: Bool = true
    let name: String
    let value: Int

    init(name: String, value: Int) {
        self.name = name
        self.value = value
    }

    func encode(with coder: NSCoder) {
        coder.encode(name, forKey: "name")
        coder.encode(value, forKey: "value")
    }

    required init?(coder: NSCoder) {
        guard let name = coder.decodeObject(forKey: "name") as? String else { return nil }
        self.name = name
        self.value = coder.decodeInteger(forKey: "value")
    }
}
//in the service class
 func doSomethingWith(object: MyCustomObject, reply: @escaping (MyCustomObject) -> Void) {
    let result = MyCustomObject(name: object.name + " processed" , value: object.value * 2)
    reply(result)
    }
//in the protocol
 func doSomethingWith(object: MyCustomObject, reply: @escaping (MyCustomObject) -> Void)
//in the app
     func sendObject() {
     guard let myService = myService else {
      print("no connection to service yet, call connectToService")
          return
      }
        let myObject = MyCustomObject(name: "Initial" , value: 10)
        myService.doSomethingWith(object: myObject) { result in
        print("received back the object \(result.name) \(result.value)")
      }
   }
```

Remember to register these classes with the `NSXPCInterface` so the framework can marshal them to and from the XPC process there is an advanced concept that you can read about if you need it it will help a lot when dealing with custom types

A few things to watch out for if you get stuck and have connection problems remember to check your entitlements and make sure the service and the main app can see each other entitlements are not always clear they can make you lose a lot of hair believe me I've almost lost all mine because of entitlements

Also if the XPC service crashes it's not going to help much you'll get errors or the connection will get invalidated which we handled there by setting the connection to nil

For resources that you could consult I'd highly recommend the official Apple documentation of course it's a bit dense but it's the source of truth there are also a bunch of WWDC videos on XPC I'd look at the ones that deal with inter process communication I wish I've had those videos when I first dealt with that there was no stackoverflow back in my day

I also found some excellent resources on the web about debugging XPC specifically a couple of blog posts from some random guy in the internet I don't know his name but they were pretty good but there were no open source books that I can recommend about XPC most books are very general about macOS and do not get into the specifics of that subject you might be able to check out some "advanced macOS programming books" maybe but not very probable

And one last joke for you to end this long winded explanation why don’t scientists trust atoms? Because they make up everything its a classic I know

Anyway I hope this is enough to get you started with XPC just remember to take things one step at a time and don't try to pass the whole world through your connection all at once that's just a recipe for disaster trust me I know
