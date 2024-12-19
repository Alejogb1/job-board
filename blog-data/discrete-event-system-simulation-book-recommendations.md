---
title: "discrete event system simulation book recommendations?"
date: "2024-12-13"
id: "discrete-event-system-simulation-book-recommendations"
---

Okay so you're looking for resources on discrete event system simulation right been there done that a few times myself Lets talk books since you asked

First thing that pops to mind when thinking about this kinda thing is the classic stuff you know the foundations My undergrad days were fueled by these types of books I remember struggling so hard with some of the more theoretical concepts but trust me its worth it in the long run Its like learning the rules of a game before you try to win every single round You really do need to understand all the bits and pieces before you start creating elaborate systems

So for a real strong base I'd suggest looking at "Discrete-Event System Simulation" by Jerry Banks It's like the bible for this stuff Honestly I probably slept with this book under my pillow for a good semester it was that intense Okay not literally but I did spend hours pouring over it Its heavy on the theory the math and a lot of queuing models Its not always a joyride to read but trust me again when you hit a roadblock later on this book is the first place you'll go back to and find some answers its got everything from random number generation to model validation Its a little bit older so it wont have all the latest hottest things in the simulation world but the core principles are timeless This is ground zero

Then once you've got a good grip on the underlying concepts you gotta think about implementation I had my first big simulation project back in my university days I was trying to simulate a super simple factory conveyor belt system with like three different stations It was supposed to be easy but its always the simplest seeming things that get you right The whole timing mechanism was a nightmare I mean really I spent an entire week debugging a stupid clock cycle error

For actual implementation I really liked "Simulation Modeling and Analysis" by Averill M Law. This one goes a lot deeper into the practical applications It has lots of examples using different programming languages usually focusing on general-purpose languages but its the logic that matters not the syntax The examples aren't copy-paste ready in most cases but its so well explained that you can port the logic over to whatever setup you are working with at the moment I remember adapting some of their queuing system examples to my old factory simulation and it helped me out a lot in terms of structuring the code in a clean manner. I would say if Banks is the bible Law is a really good guidebook or the instruction manual.

Let me show you some simple python implementations its not complicated i'll keep it simple and basic for demonstration purposes

Here's a basic event loop implementation

```python
import heapq
import time

class Event:
    def __init__(self, time, action, *args):
        self.time = time
        self.action = action
        self.args = args

    def __lt__(self, other):
         return self.time < other.time

class EventQueue:
    def __init__(self):
        self.events = []

    def add(self, event):
        heapq.heappush(self.events, event)

    def next(self):
        if self.events:
            return heapq.heappop(self.events)
        return None

def simulate(event_queue, duration):
    current_time = 0
    while current_time < duration:
        event = event_queue.next()
        if event:
            current_time = event.time
            event.action(*event.args)
        else:
            break # No more events

def arrival_event(event_queue, current_time):
   print(f"Arrival at time {current_time}")
   # this will schedule next arrival after 1 time unit for demonstration
   next_arrival_time = current_time + 1
   event_queue.add(Event(next_arrival_time, arrival_event, event_queue, next_arrival_time))

# example
if __name__ == "__main__":
    event_queue = EventQueue()
    start_time = 0
    event_queue.add(Event(start_time, arrival_event, event_queue,start_time))
    simulate(event_queue, 5)
```

This is basically the core of any discrete event system you got an event queue where you add events ordered by time and then the simulation loop just executes the events in order If you think about it a whole sophisticated discrete system is just this concept repeated and added to by adding more sophisticated handling of event actions or scheduling. This is about as basic as it can get.

Now a little more practical you gotta think about generating some random numbers You gotta be very mindful when using pseudorandom number generators If you dont seed properly you'll be surprised how not random it actually is For that check this code:

```python
import random

def generate_interarrival_times(num_events, mean_interarrival_time):
    interarrival_times = []
    for _ in range(num_events):
      interarrival_time = random.expovariate(1/mean_interarrival_time)
      interarrival_times.append(interarrival_time)
    return interarrival_times

def process_times(num_events, mean_process_time):
    process_time_list = []
    for _ in range(num_events):
        process_time = random.expovariate(1/mean_process_time)
        process_time_list.append(process_time)
    return process_time_list

if __name__ == "__main__":
  num_events = 10
  mean_interarrival_time = 2 # time units
  mean_process_time = 3 # time units
  interarrival_times_generated = generate_interarrival_times(num_events,mean_interarrival_time)
  process_times_generated = process_times(num_events, mean_process_time)
  print("interarrival times:", interarrival_times_generated)
  print("process times:", process_times_generated)
```

This little snippet is about generating interarrival times and processing times using an exponential distribution A lot of real world events can be modeled using an exponential distribution and this little example is very useful as a base tool. We all love the exponential distribution dont we?

Oh I remember when I first used that for a traffic simulation it felt like I was playing god controlling the traffic flow. I mean for like five seconds before it was all chaos I tried modeling a highway with really poor traffic management using different distributions and well I learned really quick how things can easily go south without proper planning. (That was really bad planning on my end).

And of course you cant forget the most important part validating your model. Its all fun and games till the simulation says that a single server can process 100000 requests per second when you have no server capable of that level of processing. So if the model does not make sense even on the surface you probably did something wrong.

This is the most important thing to consider so many simulation projects go completely sideways here so I would recommend "Verification and Validation of Simulation Models" by Robert G. Sargent. This book was my savior in my master's thesis I spent countless hours poring over different techniques to validate the simulation and this book explained it so well. Its important to remember you are creating a representation of reality not reality itself.

Here's an example of how you can structure some basic statistics collection in simulation that can help in your validation process:

```python
class StatsCollector:
    def __init__(self):
      self.queue_times = []
      self.waiting_times = []
      self.server_utilization = 0
      self.num_processed = 0
    def record_queue_time(self,time):
      self.queue_times.append(time)
    def record_waiting_time(self,time):
      self.waiting_times.append(time)
    def update_server_utilization(self, time, total_time):
      self.server_utilization = time / total_time
    def increment_processed_count(self):
      self.num_processed += 1
    def get_stats(self):
      if self.queue_times:
          avg_queue_time = sum(self.queue_times)/len(self.queue_times)
      else:
          avg_queue_time = 0

      if self.waiting_times:
          avg_waiting_time = sum(self.waiting_times) / len(self.waiting_times)
      else:
          avg_waiting_time = 0
      return { "average queue time" : avg_queue_time,
              "average waiting time" : avg_waiting_time,
              "server utilization" : self.server_utilization,
              "number of processes" : self.num_processed
              }

if __name__ == "__main__":
    stats = StatsCollector()
    stats.record_queue_time(2)
    stats.record_queue_time(3)
    stats.record_waiting_time(5)
    stats.record_waiting_time(7)
    stats.increment_processed_count()
    stats.increment_processed_count()
    stats.update_server_utilization(20,50) # 20 time units of server utilization out of 50
    my_stats = stats.get_stats()
    print(my_stats)
```

You can always go wild and add more data points but this little class encapsulates the very basics of collection metrics when doing a simulation run This is an extremely simplified version obviously.

So there you have it a pretty solid foundation to get you started These books I mentioned are not the easiest read they can be quite dense and theoretical but all these concepts are worth the effort. I spent way too many hours banging my head against a wall in my early days and after reading through all these its amazing how much time I saved in debugging all my system later on. And as always start with simple models first before attempting the really big ones
