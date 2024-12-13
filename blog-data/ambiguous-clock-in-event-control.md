---
title: "ambiguous clock in event control?"
date: "2024-12-13"
id: "ambiguous-clock-in-event-control"
---

Alright so this "ambiguous clock in event control" thing yeah I've been there done that bought the t-shirt and probably wrote a poorly documented library about it back in the day Let's break this down in my own way I'm assuming we're talking about scenarios where you have multiple things happening in your system that are time based and it's not entirely clear when those things *actually* happened or how they relate to each other timing wise.

This happens all the time when you're dealing with anything remotely real-time or distributed I remember once working on a distributed logging system it was a mess to debug the system would often say an event happened at x time on one server but at x + 2 seconds on a different one It was like time decided to go on a holiday without telling us. The data was basically useless for understanding the timeline and order of events. You can spend hours staring at logs and still not figure out what happened first it is a common issue.

It's usually not because the hardware is failing its more likely a problem with how you are handling the timestamps or the lack of proper clock synchronization. If you rely on local system clocks without any form of synchronization you are gonna get into a world of pain. This issue comes up not only on distributed systems but even on single machines doing a bunch of async stuff It is the same principle.

So what do you do well firstly forget about relying on naive system clocks for anything critical. They drift they wander they get interrupted they are not your friends. What you need is a reliable clock source and a robust method to use it. There are a couple of ways to approach this depending on what you are doing.

One common thing you see a lot is using a network time protocol NTP This is pretty common you have all heard of it but for the newer folks basically every computer can talk to external time servers that give you a fairly accurate view of the current time NTP isn't perfect by any means but its good enough for most situations. I know someone who told me that one time NTP made his computer 5 minutes late so he always arrives late to work. I think he is just making excuses.

Here’s an example in Python using the `ntplib` package:

```python
import ntplib
import time

def get_ntp_time():
    try:
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org', version=3)
        return response.tx_time
    except ntplib.NTPException as e:
        print(f"Error getting time from NTP: {e}")
        return None

if __name__ == "__main__":
    ntp_time = get_ntp_time()
    if ntp_time:
        print(f"Current NTP time: {ntp_time}")
        print(f"Current time (local): {time.time()}")
    else:
        print("Failed to retrieve NTP time.")

```
This example is straightforward it queries `pool.ntp.org` and prints both NTP time and your local system time. You’ll see the difference. This is a pretty basic but useful first step.
This is useful for most of the problems but sometimes NTP isn't sufficient you need something more accurate. If you need higher precision we start looking into things like GPS clocks or more specialized hardware timing solutions These are less common in the general software development space but you need them for high precision control systems.

But regardless of the time source you use you need to ensure your events have proper timestamps associated with them The best practice is to timestamp as close to the event source as possible. In the case of a distributed system this is on the machine that generates the event before it propagates it further. Here is an example of a basic event structure:

```python
import time
import uuid
from dataclasses import dataclass

@dataclass
class Event:
    event_id: str
    timestamp: float
    event_type: str
    payload: dict

def create_event(event_type: str, payload: dict) -> Event:
    """Creates an event with a timestamp and a unique ID"""
    return Event(
        event_id = str(uuid.uuid4()),
        timestamp = time.time(),
        event_type = event_type,
        payload = payload
    )

if __name__ == "__main__":
    event = create_event("user_login", {"username": "john_doe"})
    print(f"Event ID: {event.event_id}")
    print(f"Timestamp: {event.timestamp}")
    print(f"Event Type: {event.event_type}")
    print(f"Payload: {event.payload}")
```
This code generates a unique id for the event records the timestamp using `time.time()` and includes a type and a data payload. You should be more specific with timestamps by using NTP or better. I’m showing you a simplified example for clarity.

Now once you have your timestamped events in your distributed system you are going to need to combine all this data. Here's the issue though it's not just a matter of sorting by timestamps because with clock drift the sort order might be misleading It would be cool to say that the event with the earliest timestamp happened first but the problem is that an event with an earlier timestamp may actually have happened later. This is called causal order and you can use different techniques to maintain causal order.
A common solution is to use something like logical clocks such as vector clocks or Lamport clocks These clocks are not actual time rather they are counters that increase whenever an event occurs. The ordering is not based on wall time it is rather based on the order the system saw the events. So if two events were seen sequentially by the same node one after another the second event's logical clock will be greater than the first one.

Here is a very simplified example of how Lamport clocks could be applied in Python:

```python
class LamportClock:
    def __init__(self):
        self.time = 0

    def tick(self):
        """Increment the clock."""
        self.time += 1
        return self.time
    
    def merge(self, received_time):
         """Merges the local clock with a received clock value"""
         self.time = max(self.time, received_time) + 1
         return self.time

def create_event_with_lamport(clock: LamportClock, event_type: str, payload: dict) -> dict:
        return {
          "event_id": str(uuid.uuid4()),
          "timestamp": clock.tick(),
          "event_type": event_type,
          "payload": payload
        }
if __name__ == '__main__':
   clock_a = LamportClock()
   clock_b = LamportClock()

   event_1 = create_event_with_lamport(clock_a,"event_a",{"data":"hello"})
   print(f"event 1: {event_1}")
   
   event_2 = create_event_with_lamport(clock_a,"event_b",{"data":"world"})
   print(f"event 2: {event_2}")

   event_3 = create_event_with_lamport(clock_b,"event_c",{"data":"from b"})
   print(f"event 3: {event_3}")

   merged_time = clock_a.merge(event_3["timestamp"])
   event_4 = create_event_with_lamport(clock_a,"event_d",{"data":"merged"})

   print(f"event 4: {event_4}")
```

With this example you can see that although event 3 occurs independently its logical timestamp is incorporated in node A and the new timestamp takes into account event 3. Lamport clocks are good but they don't define an ordering for concurrent events. Vector clocks are better in this regard but also more complex.

If you want to learn more about clocks and time synchronization I recommend looking into the Leslie Lamport's papers on logical time or the book "Distributed Algorithms" by Nancy Lynch. These are classics and are a must-read for anyone working with distributed systems. For NTP you can look into RFC 5905 and a lot of articles that explain it.

So to summarize "ambiguous clock in event control" is usually due to not taking into account clock drift and ordering of events You need accurate time sources or other ways to order events properly if you want to make sense of your distributed system or async stuff you are doing. Good luck and remember time is not always what it seems.
