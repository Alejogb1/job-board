---
title: "embedded firmware architecture?"
date: "2024-12-13"
id: "embedded-firmware-architecture"
---

Alright so embedded firmware architecture huh Been there done that got the t-shirt several t-shirts actually Lets dive in

First off its not exactly a monolithic thing there is no single best architecture it really depends on the project the hardware the team size all that jazz I've seen some absolute monstrosities and some elegant designs Its not a one size fits all situation and anyone telling you that probably hasnt dealt with a real embedded system.

Now I will tell you about the stuff that has worked well for me and the pitfalls to avoid.

Okay so lets start with the basics you are probably thinking about the overall structure right like how do we organize the code Its not just about writing the code its about how the components of the firmware communicate each other.

Back in the day like my first real gig I was working on a sensor network thing think lots of little nodes transmitting data back to a central hub. We started with a very monolithic approach you know all the code living in one massive loop it was terrible. It was so bad debugging was a nightmare modifications were risky and adding features was just like throwing darts in the dark. It was this massive ball of spaghetti and then someone told me hey this is a real world problem you cant just rewrite everything in 3 days you need something that works better so we actually had to learn.

We used like state machines. Each node was modelled as a state machine and then we made them communicate with a simple message-passing system. It was not perfect but it was a huge improvement because each node could be developed more independently and we could debug specific states and transitions a lot easier plus now we could add new sensors without having to touch the core of the system and it was a huge win for the team.

So here's a snippet of how a state machine for a simple sensor reading node could look like this is just a basic version and its in plain C not fancy but you get the idea

```c
typedef enum {
    STATE_IDLE,
    STATE_READING_SENSOR,
    STATE_TRANSMITTING_DATA
} NodeState;

NodeState currentNodeState = STATE_IDLE;

void processNode(){
    switch (currentNodeState) {
    case STATE_IDLE:
        //Do some idling stuff
        if (timeToReadSensor()){
            currentNodeState = STATE_READING_SENSOR;
        }
        break;
    case STATE_READING_SENSOR:
         readSensorData();
         currentNodeState = STATE_TRANSMITTING_DATA;
         break;
    case STATE_TRANSMITTING_DATA:
         transmitData();
         currentNodeState = STATE_IDLE;
         break;
    }
}

int main(){
  while (1) {
    processNode();
  }
  return 0;
}
```

That is a simplified example but the state machines makes it easier to follow the flow of the program and the modularity helps with code maintainability. This is why I started to really get into architecture because you know the code can be great but if you cannot maintain it or understand how it works it is not going to help you at all.

Now another thing that i've seen a lot in different projects is using a layered architecture. Think of it like an onion with each layer depending on the layers below it. You have like hardware abstraction at the bottom then your driver layer and then your business logic in the top layer. This allows you to replace drivers or even hardware without changing a single line of the business logic layer. We once had to swap sensors on an industrial control thing and we did not have to change any business logic code. It saved a lot of time and headaches believe me. I will give you the details of that specific case.

So it was the same company but a different team for a different industrial control system. We had to change a temperature sensor due to supply chain issues. We had implemented a decent layered architecture so all we had to do was implement a new sensor driver with the same interface as the previous one and nothing else had to be changed it was like plug-and-play.

Here's a basic C header file example for a sensor abstraction layer

```c
// sensor_interface.h
#ifndef SENSOR_INTERFACE_H
#define SENSOR_INTERFACE_H
typedef struct {
    float (*read_temperature)();
    float (*read_humidity)();
} SensorInterface;
extern SensorInterface currentSensor;
#endif
```

This header defines an interface for accessing sensor readings, the actual implementation would be defined in a sensor driver file and then we select the implementation in the main program.

So this was a great win and its something i've used across different projects ever since.

One other trick I picked up over the years is to never underestimate the power of task scheduling if you have a microcontroller and your doing more than a single thing. I am not just talking about using a full blown OS here. I am saying that you may not want the overhead of a full RTOS for everything. For example, lets say you have a communications task a sensor reading task and a display update task. These can be implemented using cooperative or preemptive scheduling. Cooperative is okay for basic stuff but preemptive is the way to go if you have more complex time sensitive operations.

I use a basic round-robin scheduler all the time when I do not need anything fancy that i build myself based on a timer interrupt.

Here's an example of a simple cooperative scheduler you may implement

```c
#include <stdint.h>
#include <stdbool.h>

typedef void (*task_func_t)(void);

typedef struct {
    task_func_t task;
    bool enabled;
} task_t;

#define MAX_TASKS 3

task_t tasks[MAX_TASKS];
uint8_t current_task_index = 0;

void add_task(task_func_t task) {
    if (current_task_index < MAX_TASKS) {
        tasks[current_task_index].task = task;
        tasks[current_task_index].enabled = true;
        current_task_index++;
    }
}

void run_scheduler(){
    for (int i=0; i<current_task_index; i++){
        if (tasks[i].enabled) {
            tasks[i].task();
        }
    }
}

void task1(){
    // do task 1 stuff
}

void task2(){
    // do task 2 stuff
}

int main(){
    add_task(task1);
    add_task(task2);
    while (1){
       run_scheduler();
    }
    return 0;
}
```

This scheduler is very basic but you get the gist You create a bunch of functions that do your tasks and then the scheduler simply runs them one after the other. Of course a proper preemptive scheduler would be more complex and use interrupt to achieve that.

Alright lets get a little deeper shall we? Memory management in embedded systems it is a topic on its own and very important. You are dealing with limited resources and you have to know how to use them well. Dynamic memory allocation is often something to be avoided since it may lead to memory leaks or fragmentation. There is a well know expression that when you free your memory too late its called a memory leak when you free it too early it is called a double free and we definitely don't want to do either.

Static allocation and memory pools are your friends here. You define all of your memory needs upfront and then allocate it statically in the system. You may need to be mindful of the stack size too because you will not be happy if you get stack overflows and those can be very tricky to debug. The worst stack overflow I had was when i had to use a logic analyzer to find a loop where the compiler was not optimizing the local variables and it was generating new memory for them for each iteration inside a loop. It was not fun let me tell you. I swear the hardware vendor team was secretly laughing at us. It is something that we always check now.

Also you might want to take a look at flash memory management because often we are not using a proper filesystem and our data can end up fragmented or you might want to implement a journaling system to prevent corruption when the power gets interrupted during writes. This is definitely something that will make your system more robust but requires a good design.

I remember one project where we had an external flash for logging sensor data. It was not very well implemented initially so if power was lost during a write the data would be inconsistent or even corrupted. We ended up implementing a log structured file system to fix the issue it is a bit more overhead but its much more robust.

Oh and lets talk about code quality you need to have a good team culture because even if the architecture is beautiful the code can be full of bugs and impossible to maintain if the coding style is inconsistent and there is no code review process. I highly recommend static code analysis tools for catching common mistakes early in the cycle. They are your friends trust me. Now this is important because you may not even have a chance to have a debugger connected to your board.

So that is a good overview of my past experience in embedded firmware and architecture. There are other aspects of the subject too but they depend a lot on your needs. You want to dive deeper into the subject then I would recommend you check out "Embedded Systems Architecture" by Tammy Noergaard and "Making Embedded Systems" by Elecia White. Those books are a good starting point and helped me a lot when i was starting in the field.

Now there is this joke that I always tell my friends. You know you have become an embedded developer when debugging a race condition becomes more exciting than a thriller movie and i am not kidding.

So yeah thats it. Its a complex field but a very exciting one you will see I am sure. Remember start simple and iterate and dont try to make it perfect in the first try. Good luck and happy coding.
