---
title: "python canopen library example?"
date: "2024-12-13"
id: "python-canopen-library-example"
---

Alright so you’re asking about `python-canopen` right Been there done that probably more times than I care to admit Look I’ve wrestled with CANopen a fair bit and `python-canopen` is well its a tool lets just say it’s a tool that requires understanding the ins and outs to avoid a world of pain

See I started my CANopen journey back in like 2015 we were building this automated test rig for industrial motor controllers think huge noisy things whirring and clanking not exactly your friendly desktop setup Anyway we needed a way to command these controllers and get feedback in real time CANopen seemed like the best option at the time and python well python was my language of choice So `python-canopen` was the answer or so I thought

First off setting up the basic communication thats always a hurdle You need to make sure you have your CAN interface set up right and the baud rate needs to match the network You wouldn't believe how many hours I’ve burned because of a silly baud rate mismatch I swear it’s like the classic “did you plug it in” of the embedded world

Here’s a basic snippet just to get you started assuming you have the `can` and `canopen` libraries installed of course:

```python
import can
import canopen

# Configure the CAN interface
bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=250000)

# Create the network object
network = canopen.Network()
network.connect(bus)

# you can use your specific node id here its the "address"
node_id = 1 
node = canopen.BaseNode402(node_id, network)

# we should start the node state machine
node.nmt.state = 'PRE-OPERATIONAL'

print("CANopen network setup complete")
```

Okay so here the first part is CAN bus setup It can be a `virtualcan` or whatever you need to make the connection with the physical CAN network Next we create a network object which is our primary handler for everything related to the CANopen network We connect this network object to the previously created CAN bus object And last we create our node with a certain node ID and after that we set the node to a `PRE-OPERATIONAL` state This is our first minimal CANopen connection

Now you'll notice I'm using `BaseNode402` This is because most of the time you’re dealing with servo drives or frequency converters following the CiA402 profile if you are not I suggest you read the `CiA301` standard documentation which you will need to understand more than the code itself This `BaseNode402` simplifies a lot of the common operations like setting the operation mode moving the motor and reading feedback but if your device is different you should inherit this class and overwrite the behaviour of it You could even implement your own `Node` class for special devices You'll want to check your device's EDS file (electronic data sheet) this is absolutely crucial This file details all the communication parameters object dictionary entries and supported services for that specific device Its like the instruction manual but for your CANopen communication and its a must have if you want to communicate effectively

Speaking of object dictionaries accessing them is fundamental to working with CANopen and `python-canopen` makes it relatively easy The object dictionary is a big table of variables each identified by an index and sub-index To read a value you just specify the address and the type and the library will handle the rest The tricky part is knowing where to find the address of what you want For example to access the current position of the motor controller using a node this could be something like this:

```python
#Lets assume the node has been created before in our code

position_index = 0x6064 #this is the index for the position
position_subindex = 0x00 #and the subindex for the position

try:
    position = node.sdo[position_index][position_subindex].raw
    print(f"Current motor position: {position}")
except canopen.SdoAbortedError as e:
        print(f"Error reading position: {e}")

```
Here we are using the node object to access the SDO (Service Data Object) this method allow us to read and write to the object dictionary we try to read the position value from the object dictionary of the CANopen node If it works we print it If not we print the error Now for the SDO the index and subindex must be valid according to the EDS file

Writing to the object dictionary is equally as important You use this to configure the device enable it and make it do what you want For instance to set the target position lets assume our motor is configured to work in the position control mode this example is something like:

```python
#Lets assume the node has been created before in our code

target_position_index = 0x607A # this is the target position index
target_position_subindex = 0x00 #and subindex

try:
        node.sdo[target_position_index][target_position_subindex].raw = 1000 #Set target to 1000
        print("Target position set to 1000")
except canopen.SdoAbortedError as e:
        print(f"Error writing target position: {e}")

```

See we use the same structure as before to read but now we are writing the value `1000` to the target position register Again you can find the address of each register in your devices EDS file If your motor does not work in position control mode this will fail or it could be written to the register but nothing will happen The reason is not the program its the CANopen device so its very important to know what exactly is happening in the object dictionary of your device that is why you need to check the EDS file so you know what is the object that you are changing and what its behaviour

One more thing to keep in mind is that CANopen is asynchronous You don’t get immediate responses always You usually have to poll for data or rely on PDO (Process Data Object) mapping This whole PDO thing can be a bit much at first but its essential for real time data streaming and control In most of the time if you are working with servos or motor drivers you will only use the SDO to configure the system initially and then you will start moving your motors using PDOs in a real time control loop

Another gotcha is error handling CANopen errors can be cryptic and sometimes the errors will not be so obvious for example a timeout when you try to read a register from the device it might not be the device problem It could be a bus problem a configuration problem or even a bad EDS file It can also be a problem with your software implementation You will need to make sure that your exception handling is robust and gives you enough feedback to diagnose the problem and in most cases it helps to print the messages using the debug mode available for the library in order to gain better insight in the CANopen network

And one more thing if you’re thinking about integrating `python-canopen` into a large scale system think very carefully about how you’re going to structure your code The object-oriented approach that python provides is a great way to keep your code organized and modular Remember that a simple `while True` loop reading and writing in a continuous manner is not a real solution for a more advanced application For example a simple controller could become a state machine and you need to make the interface between the CANopen network and your control algorithm very simple and fast

My first attempts looked like spaghetti code with global variables all over the place trust me when I say its a nightmare to debug My advice is to modularize everything use classes to represent your CANopen nodes abstract away the low level details and remember to think of the problem as a state machine So its state is always well defined and you can handle each state appropriately

Oh and about resources don’t just rely on tutorials online They are great to start but you need to dive deeper into the documentation If you really want to master CANopen you should definitely read “Embedded Networking with CAN and CANopen” by Olaf Pfeiffer and Chris P Keydel This book gives you a detailed insight of the standard and the protocol and it’s a great start. Also the official documentation of the CiA301 and CiA402 standards are indispensable this documentation contains every single detail of the protocol

By the way I once spent a whole week debugging a motor that would randomly stop only to realize that a different node on the network had the same node ID classic CANopen headache you see its like that joke a programmer gets two problems but when he tries to solve them he ends up with 1000 problems

So yeah that’s my experience with `python-canopen` It’s a powerful library but it requires a solid understanding of CANopen and a careful coding approach Be prepared for some headaches but also for the satisfaction of seeing your devices communicate in real time If you need more details about a specific topic please ask I have wrestled with CANopen for long enough and I might have a solution or at least I can point you to a better direction for your specific application
