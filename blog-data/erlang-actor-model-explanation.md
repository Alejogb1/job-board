---
title: "erlang actor model explanation?"
date: "2024-12-13"
id: "erlang-actor-model-explanation"
---

Alright so you wanna dive into Erlang's actor model eh I get it it's a beast but a beautiful one once you wrap your head around it I've wrestled with this thing for years literally back in the early 2000s when I was building a distributed chat server in a custom Lisp dialect yeah I know Lisp a weird choice but hey it worked and it kinda hinted at the actor model before I actually knew about Erlang and its beauty

So essentially the Erlang actor model isn't about class inheritance or shared memory it's about little independent processes aka actors that communicate via messages think of it like a bunch of tiny email servers each handling its own thing and only talking to other servers through sending well emails yeah simple enough right No shared mutable state no deadlocks that are inherent in the typical shared-memory concurrent systems thats a big relief I'm telling you dealing with those locking issues and race conditions used to give me nightmares now i sleep like a baby even when debugging a thousand processes at once

Each actor has its own mailbox it receives messages processes them and maybe sends messages to other actors or creates new actors It’s like a tiny machine with its own brain its own small workspace and a well organized in-tray

Here’s the basic idea think of an actor as a loop:

```erlang
loop() ->
  receive
    { From, Message} ->
       io:format("Actor ~p received message: ~p from: ~p~n", [self(), Message, From]),
      % do something with Message
      loop();
  end.
```

This is a really simple example it just receives a message prints it to the console then calls itself again thats the whole loop thing It's endlessly listening waiting for new work to arrive via messages

To spawn this actor you do:

```erlang
start_actor() ->
   spawn(fun() -> loop() end).
```
`spawn/1` creates a new process that executes the function that is passed to it `fun() -> loop() end` this returns the pid process id of the newly created actor and this pid is how you address the actor when sending it messages you basically use it as the recipients email address if you will

Now to actually send a message to that actor we use `!`

```erlang
send_message(Pid, Message) ->
    Pid ! { self(), Message }.
```

We use the `!` operator which is the send operator we send the message `Message` along with our process id `self()` to the destination actor pid `Pid` remember the mailbox thing

And this isn't it no this is the bare bones foundation The real power of the Erlang actor model stems from a few key concepts

First **Asynchronous Communication** messages aren’t blocking you send a message and move on you're not sitting there waiting for a response unless you specifically design it that way That's what I love about it you can start a bunch of calculations send them all off then come back for the results later and not have your main thread getting blocked that blocking crap used to be my worst nightmare honestly and I'm not saying it lightly that was truly an issue

Second **Isolation** each actor has its own memory space they don't have shared access to each other's memory its like each actor lives in its own bubble No data race issues no accidentally clobbering each others variables no shared state problems no headaches

Third **Supervision** this is a big one Erlang has this notion of supervisors These are special actors that are responsible for watching over other actors if an actor crashes the supervisor can restart it and bring it back to life This whole "let it crash" philosophy is game changing I can tell you that I used to have a debug hell in my other life until I finally understood this and my life got way better I'm not kidding

Fourth **Distribution** Erlang's actors can run on different machines as if they were all on the same system that distributed chat server I mentioned before thats the place where I first got to apply and understand this thing I mean you can send a message to an actor on another machine as easily as you send it to one on the same machine transparently I have actually used it to control a network of remote robots doing real stuff on the field its insane how simple things become with this model

Okay I know that was a bit of a info dump but it’s important stuff When I was first getting into this actor model stuff I spent hours reading through Joe Armstrong's “Programming Erlang” book man that's the place where it all started for me Its a great resource even today I actually recommend checking it out If you want something a little more academic check out the "A Theory of Actors" paper by Carl Hewitt or Gul Agha its a bit dry but its like the ur-text for all this actor model goodness its worth its while I can tell you that

You might ask yourself well that looks easy why all the fuss It is easy in its design but you have to really get the mindset shift from classical concurrent and parallel programming the key is to think of everything as independent interacting entities that communicate only via messages you have to think of your application as a network of little interacting state machines I don't know if that is a proper way to think of it I'm just giving it my personal experience over the years because you will eventually get the hang of it eventually trust me I have had to rewire my brain completely when I first used it and it did take some time

And here’s a simple example of how it actually works together a ping pong example:

```erlang
ping(PongPid) ->
    receive
        pong ->
            io:format("Ping received pong~n"),
            timer:sleep(1000),
            PongPid ! { self(), ping},
            ping(PongPid);
        Other ->
          io:format("Ping received something weird: ~p~n", [Other]),
          ping(PongPid);
      end.


pong() ->
    receive
        { PingPid, ping } ->
            io:format("Pong received ping from: ~p~n", [PingPid]),
            timer:sleep(1000),
            PingPid ! pong,
            pong();
        Other ->
            io:format("Pong received something weird: ~p~n", [Other]),
            pong();
    end.

start_ping_pong() ->
  PongPid = spawn(fun() -> pong() end),
  PingPid = spawn(fun() -> ping(PongPid) end),
  PingPid ! {self(), ping}.
```

Run `start_ping_pong()` and watch the ping and pong messages flow That is the simplest working example I could create just for you. And there you have it.

Some key takeaways think message passing over shared memory think no locking no weird state think let it crash think supervision think distributed systems think actors

One last thing before you go when I was learning this stuff I actually sent an infinite number of messages to myself just to see what would happen my system crashed but I laughed about it because it was so easy to restart it and just fix that code it was actually pretty funny at least to me back then That is why I love this thing It gives you an incredibly powerful and robust way to build concurrent and distributed systems without the typical headaches That’s really what made me stick with this approach you know

And I think that's about it for the Erlang actor model explanation in a nutshell You might need to wrestle with it yourself a little bit get your hands dirty so to speak it's not something you just understand from reading about it you got to code with it You have to feel it that is how I learned it

Good luck you got this just remember to "let it crash" and to keep your messages flowing asynchronously.
