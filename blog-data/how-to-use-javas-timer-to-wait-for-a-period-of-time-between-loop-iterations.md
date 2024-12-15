---
title: "How to use Java's Timer to wait for a period of time between loop iterations?"
date: "2024-12-15"
id: "how-to-use-javas-timer-to-wait-for-a-period-of-time-between-loop-iterations"
---

i've seen this question pop up countless times, and it’s a really common hurdle when you start dabbling with background tasks or simulations in java. you want to execute something repeatedly, but not too fast, and that's where `java.util.timer` can seem like a straightforward option. but let's look at why it is not ideal and how to do it correctly.

first off, i can tell you about a past experience. i was building this, let's call it, a ‘data processing pipeline’ about 10 years ago. it would pull data from a few apis, massage it a bit, then push it out to another service. i started using `timer` because i just wanted a delay between processing each batch of data. i figured, just schedule the task and each time it runs wait a second and everything would be fine and done.

the code looked something like this originally, and it's horrible now when i look at it. it almost makes me cringe:

```java
import java.util.Timer;
import java.util.TimerTask;

public class BadExampleTimer {
    public static void main(String[] args) {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                // do processing here
                System.out.println("processing data, should take about 1 sec");
                 try {
                    Thread.sleep(1000); // simulate processing
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                   }

            }
        }, 0, 1000);
    }
}

```

it seemed perfect initially. processing data, then wait a second, then process again. but what happened is that the api's started getting a bit slower. not much. just a little. and suddenly, the `timer` started queuing up tasks because the process was taking longer than a second. it was a mess. the next thing i saw was memory exhaustion and complete failure of the whole application and not a good time. because of the `scheduleAtFixedRate`, the `timer` doesn't wait for the previous task to finish before scheduling the next. and if your task takes longer than the specified period (in my case one second), you end up with overlapping executions, like the code i just showed you. this is not what i wanted.

so, the big takeaway here is that `java.util.timer` with `scheduleAtFixedRate` is not a good idea for waiting between iterations of a loop if the loop’s content can take different amount of time. `scheduleAtFixedRate` will try to execute your code at a fixed rate, regardless of how long your previous execution takes, which results in overlapping. and even if you use `schedule` instead, which is better, you might still encounter problems if your task throws an exception, for example. that could silently cancel the timer. i lost many hours looking for that issue.

the proper way to handle waiting between iterations in a loop? well, it depends on what you’re doing, but here's a cleaner approach. instead of a timer, use a loop with `thread.sleep()` method. and be aware of the interruption issues which i did not.

let's modify the code i shown before to something better:

```java
public class BetterLoopWithSleep {

    public static void main(String[] args) {
        while (true) {
            System.out.println("processing data, should take about 1 sec");

           try {
               Thread.sleep(1000);
               // do processing here
            } catch (InterruptedException e) {
                 Thread.currentThread().interrupt();
                 System.err.println("task interrupted");
                 break; // end the execution correctly.
            }

        }
    }
}

```

this is much simpler and safer. i learned a lot of patience debugging the first code. in this new code we are using a `while (true)` loop and then using `thread.sleep` to pause for one second. then, if the thread is interrupted which could happen if the application is shutting down, we do not loop again. we break.

now, a slight improvement. let's add a delay that is not that precise. let's say you need to have a variable delay between 1 and 2 seconds. and still use the `thread.sleep` but make it variable and not fixed.

```java
import java.util.Random;

public class VariableDelayLoop {
    public static void main(String[] args) {
        Random random = new Random();
        while (true) {
            System.out.println("processing data, might take between 1 and 2 seconds");

            try {
                int delay = 1000 + random.nextInt(1000); // random delay between 1000 and 2000
                Thread.sleep(delay);
                // do processing here
            } catch (InterruptedException e) {
                 Thread.currentThread().interrupt();
                 System.err.println("task interrupted");
                 break; // end the execution correctly.
             }

        }
    }
}
```

this gives you a bit more control. you can adjust the logic in the loop depending on your requirements. and as a fun side note, if you put a breakpoint in the line of the `thread.sleep` call, you can pretend you are doing some serious debugging while just having coffee.

for more advanced scenarios, or more complex scheduling, you might want to look into `java.util.concurrent`. things like `scheduledthreadpoolexecutor`. you can have much more fine-grained control over thread execution and pooling. the `scheduledthreadpoolexecutor` gives you similar `scheduleatfixedrate` behavior that i mention at the beginning. but the big difference is that it is designed to be much more thread-safe. it's part of the concurrency package, which was built to resolve problems like those and much more. `timer` was added in java 1.3, but `scheduledthreadpoolexecutor` was added in java 1.5 as part of that concurrent library, which says something about the improvements and learnings from java.

regarding resources, i would recommend reading “java concurrency in practice” by brian goetz, or the java concurrency tutorial available online. the official java docs are also gold, of course. especially when you’re trying to figure out the finer details of the various concurrency constructs. and just as a general tip, it’s a good idea to avoid creating new threads if you don't really need to because there is a cost to thread creation, it's not just free, so try reusing them via thread pools. also, always catch `interruptedexception` and handle it by setting the interrupted flag. don't just swallow the exception and continue looping.

in summary: avoid `java.util.timer` and prefer using `thread.sleep` within a loop for basic delays between iterations. and if you require complex scheduling use the utilities available in the concurrency library. these will serve you much better in the long run.
