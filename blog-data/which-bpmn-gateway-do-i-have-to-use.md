---
title: "which bpmn gateway do i have to use?"
date: "2024-12-13"
id: "which-bpmn-gateway-do-i-have-to-use"
---

 so you're asking about BPMN gateways a classic headache if you haven't spent enough time wrestling with process flows Trust me I've been there done that got the t-shirt and probably a few scars too Let's break this down without the fancy business jargon just straight up technical talk

First off it's not about picking *a* gateway it's about picking *the right* gateway for the specific branching logic you need BPMN doesn't just throw a bunch of gates at you and say "go nuts" each one has a distinct purpose and misunderstanding that purpose is like trying to debug code without knowing the language You'll just be flailing around

Let's start with the basics the exclusive gateway This guy is your workhorse the "if else" statement of BPMN Only one path out of this gateway can be active at any given time It's controlled by a condition expression Think about it like this you've got a process where if a customer is from California then you send them one email if they aren't you send them a different one That's classic exclusive gateway territory

Here's how you might see it in a BPMN XML or diagram kinda rough example I'm not gonna bore you with the entire XML structure

```xml
<bpmn:exclusiveGateway id="ExclusiveGateway_1">
    <bpmn:incoming>Flow_1</bpmn:incoming>
    <bpmn:outgoing>Flow_2</bpmn:outgoing>
    <bpmn:outgoing>Flow_3</bpmn:outgoing>
</bpmn:exclusiveGateway>
<bpmn:sequenceFlow id="Flow_2" sourceRef="ExclusiveGateway_1" targetRef="Activity_California">
    <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">${customer.state == 'CA'}</bpmn:conditionExpression>
</bpmn:sequenceFlow>
<bpmn:sequenceFlow id="Flow_3" sourceRef="ExclusiveGateway_1" targetRef="Activity_OtherState" />
```

See the condition expression there `customer.state == 'CA'` That's what controls the flow Think of it as a switch only one path is ever flipped at any one time I've personally screwed this up so many times forgetting a default flow and leaving processes hanging in limbo so trust me double check your defaults if you're not sure where to go

Then you've got the parallel gateway This is your "and" you need things to happen simultaneously Multiple outgoing paths are all activated when the incoming flow reaches this gateway You might use it when you need to do multiple things in parallel like send an email to the customer and update their profile at the same time

Here's a quick snippet showing that:

```xml
<bpmn:parallelGateway id="ParallelGateway_1">
    <bpmn:incoming>Flow_4</bpmn:incoming>
    <bpmn:outgoing>Flow_5</bpmn:outgoing>
    <bpmn:outgoing>Flow_6</bpmn:outgoing>
</bpmn:parallelGateway>
<bpmn:sequenceFlow id="Flow_5" sourceRef="ParallelGateway_1" targetRef="SendEmailActivity" />
<bpmn:sequenceFlow id="Flow_6" sourceRef="ParallelGateway_1" targetRef="UpdateProfileActivity" />
```

Notice there are no conditions The process just fires off both paths together You gotta make sure if your using a parallel gateway that you have a join gateway later on if necessary to bring everything back together else you'll end up with orphaned processes and your system will be in an unholy mess

And a parallel gateway is probably the gateway that has caused me the most headaches since you forget to close the thread with a joining parallel gateway and the process hangs forever never finishes This also leads to all sorts of memory problems and resource wastage and sometimes it is really hard to debug

Then there's the inclusive gateway This guy's like a more flexible version of the exclusive gateway it's an "or" not an "exclusive or" Multiple paths *can* be activated depending on the conditions This is where things get a bit more complex It's not always immediately obvious what the behavior is You should try to avoid using these as much as possible they create a lot of confusion and hard to follow diagrams

Here's a quick example to make it easier:

```xml
<bpmn:inclusiveGateway id="InclusiveGateway_1">
    <bpmn:incoming>Flow_7</bpmn:incoming>
    <bpmn:outgoing>Flow_8</bpmn:outgoing>
    <bpmn:outgoing>Flow_9</bpmn:outgoing>
</bpmn:inclusiveGateway>
<bpmn:sequenceFlow id="Flow_8" sourceRef="InclusiveGateway_1" targetRef="Activity_SendSms">
    <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">${customer.allowSms == true}</bpmn:conditionExpression>
</bpmn:sequenceFlow>
<bpmn:sequenceFlow id="Flow_9" sourceRef="InclusiveGateway_1" targetRef="Activity_SendPush">
     <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">${customer.allowPush == true}</bpmn:conditionExpression>
</bpmn:sequenceFlow>
```

Now in this scenario if `customer.allowSms` and `customer.allowPush` are both true then *both* `Activity_SendSms` and `Activity_SendPush` will be activated If only one of them is true then only that path will be activated If none is true then neither will be activated

These are the big three but you also have the event based gateway which is often used for handling timeouts and external triggers which is more complex

But when to use which that's the real question right Its not just about blindly following a diagram its about understanding the flow of data in your process

If you have a condition that needs to be exclusive ie only one path is available use an exclusive gateway If you need things to be done in parallel use a parallel gateway And if you need to do more than one thing depending on different conditions use an inclusive gateway

And here is the funny part my favorite BPMN gateway is the one that is actually used correctly I have seen so many messed up process diagrams that it actually makes me wonder what everyone was doing with their lives The point is make sure that you know what you are doing or you'll be in trouble

I know this seems like just a basic explanation but these things can get hairy really quickly

For resources I wouldn't go for a single website or a blog post they rarely give the full picture Instead I'd suggest "BPMN Method and Style" by Bruce Silver it's a solid practical guide for writing clear diagrams that people can actually understand This guy has seen it all and you can really learn from his experiences Then you have "Workflow Management Models Method and Systems" by Wil van der Aalst it dives deep into the theoretical side of workflow and process modelling You won't get instant answers from it but you get a very deep understanding on how things are working and why they work in the first place

And if you really want to go deep into the technical stuff I'd recommend looking at the OMG specification for BPMN that's where you find all the official definitions and details

Finally always test and iterate on your BPMN diagrams and if you are using a process engine make sure that you simulate your process to make sure it will behave the way you expect it too Never assume anything and test everything you can think of Your future self will thank you for it and if you don't your future self will hunt you down

Anyway that's my take on gateways you should have a good handle on them now If not feel free to ask more questions but try to be as specific as you can It will make it easier for me and for you to resolve the issue
