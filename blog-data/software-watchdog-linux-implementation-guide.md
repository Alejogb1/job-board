---
title: "software watchdog linux implementation guide?"
date: "2024-12-13"
id: "software-watchdog-linux-implementation-guide"
---

Okay so you're asking about software watchdogs on Linux right I've been around the block a few times with those things lets see if I can get you up to speed and help you navigate what is often a surprisingly tricky landscape

Alright lets dive in We are talking about software watchdogs here not those silly hardware things people use on embedded systems though the principle is similar The idea is pretty basic you have a process that's supposed to periodically nudge a kernel module and if it doesn't do that within a specific time period then the kernel module takes action usually a system reboot but you can also set it up to do other stuff like maybe trigger a debug dump

I've dealt with watchdog implementations for years and I've seen pretty much every variation out there from simple single threaded applications to more complex distributed ones Let me give you some pointers based on the messes I've gotten myself into

First off the kernel side you're going to be using `/dev/watchdog` or `/dev/watchdog0` typically they might be located somewhere else depending on how your distro is configured but that is their usual location and its important to verify that this device exists otherwise you wont be able to use it it is a character device you need to open it write to it to keep the timer alive and if you dont write to it the system reboots its pretty simple in practice it can be tricky if you get the timeouts wrong or have an unstable system

Here is an example of a very simple c program to ping the kernel watchdog

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/watchdog.h>

int main() {
    int fd;
    int timeout = 30; // example timeout in seconds
    int flags;


    fd = open("/dev/watchdog", O_WRONLY);
    if (fd < 0) {
        perror("Error opening watchdog device");
        return 1;
    }

     //Disable the reboot option
    flags = WDIOS_DISABLECARD;
	if (ioctl(fd, WDIOC_SETOPTIONS, &flags) < 0){
        perror("Error setting watchdog options");
		close(fd);
		return 1;
    }


    if (ioctl(fd, WDIOC_SETTIMEOUT, &timeout) < 0){
        perror("Error setting watchdog timeout");
        close(fd);
		return 1;
    }

    while (1) {
        if (write(fd, "\0", 1) != 1) {
            perror("Error writing to watchdog");
            close(fd);
            return 1;
        }
         sleep(10); // Ping every 10 seconds, make sure it is shorter than timeout
    }

    close(fd);
    return 0;
}
```

This program opens the watchdog device sets the timeout which is mandatory and then goes into a loop and writes a byte to the device this write operation keeps the watchdog timer alive and prevent a reboot from happening The 10 second sleep is just an example you should tune that based on your application

Now in the real world the program isn't just going to be a while loop with a sleep in it right you would probably have a main thread doing its stuff and a separate thread taking care of the watchdog process The main thread should notify the watchdog thread that it is doing what is it supposed to do and that thread is responsible for the watchdog pinging Here is an example in python that does that using threads:

```python
import threading
import time
import os
import fcntl
import struct
import errno

# Watchdog IOCTL constants
WDIOC_GETSUPPORT= 0x40087700
WDIOC_GETSTATUS=  0x40047701
WDIOC_SETOPTIONS= 0x80047702
WDIOC_KEEPALIVE=  0x00007705
WDIOC_SETTIMEOUT= 0x40047706
WDIOC_GETTIMEOUT= 0x40047707
WDIOC_GETPRETIMEOUT= 0x40047708
WDIOC_SETPRETIMEOUT= 0x80047709

WDIOS_DISABLECARD  = 0x0001
WDIOS_ENABLECARD   = 0x0002

WATCHDOG_DEVICE = "/dev/watchdog"

class Watchdog:
    def __init__(self, timeout=30):
        self.timeout = timeout
        self._fd = None
        self._running = False
        self._keepalive_event = threading.Event()
        self._watchdog_thread = None
        self.reset_signal = False


    def _open_device(self):
        try:
            self._fd = os.open(WATCHDOG_DEVICE, os.O_WRONLY)
            return True
        except OSError as e:
            print(f"Error opening watchdog device: {e}")
            return False

    def _disable_reboot(self):
        flags = WDIOS_DISABLECARD
        try:
           fcntl.ioctl(self._fd, WDIOC_SETOPTIONS,struct.pack('i', flags))
           return True
        except OSError as e:
            print(f"Error disabling reboot option: {e}")
            return False
    

    def _set_timeout(self):
        try:
            fcntl.ioctl(self._fd, WDIOC_SETTIMEOUT,struct.pack('i', self.timeout))
            return True
        except OSError as e:
            print(f"Error setting watchdog timeout: {e}")
            return False



    def _keepalive_loop(self):
            while self._running:
                self._keepalive_event.wait()
                if not self._running:
                    break
                try:
                     os.write(self._fd, b'\0')

                except OSError as e:
                    print(f"Error writing to watchdog: {e}")
                    self.stop()

                self._keepalive_event.clear()
                time.sleep(1)




    def start(self):
        if self._fd is not None:
           print("Watchdog already started")
           return

        if not self._open_device():
           return

        if not self._disable_reboot():
            self._close_device()
            return
        
        if not self._set_timeout():
            self._close_device()
            return

        self._running = True
        self._watchdog_thread = threading.Thread(target=self._keepalive_loop)
        self._watchdog_thread.start()
        print("Watchdog thread started.")



    def ping(self):
            if not self._running:
                print("Watchdog not running")
                return
            self._keepalive_event.set()


    def stop(self):
            if self._running:
                self._running = False
                self._keepalive_event.set()
                if self._watchdog_thread is not None:
                    self._watchdog_thread.join()

            self._close_device()
            print("Watchdog stopped")
    
    def _close_device(self):
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


def main_process():
    watchdog = Watchdog(timeout=30) # Timeout of 30 seconds
    watchdog.start()
    try:
      while True:
         #Simulate application work
         print("Doing main process work")
         time.sleep(10)
         # Send heartbeat to watchdog
         watchdog.ping()
    except KeyboardInterrupt:
        print("Exiting main process")
    finally:
        watchdog.stop()


if __name__ == "__main__":
    main_process()
```

This example provides a reusable watchdog class that you can initialize with a timeout and start it The ping method is used to keep the watchdog alive If the application freezes or dies the watchdog will not be pinged and the system will reboot after the set timeout it has a stop method as well for cleanups and graceful shutdown of the app

Now where things can get complex this is where you have to consider corner cases like what if the watchdog process itself hangs or does something unexpected it's not unheard of that a buggy watchdog process takes down a whole system because it's not pinging even though the main process is still alive so ideally this secondary process should be as robust as possible which usually means less features and more reliability the more complicated it is the more potential issues you will have

And what happens if your application takes too long to execute and does not notify the watchdog thread? well the system will reboot that's usually the desired behavior in those cases because the application is stuck in a loop and its unable to progress its internal state will be inconsistent and will probably cause more issues than a reboot ever will so its usually a good idea to just restart everything with a watchdog this means you need to measure your execution times and pick timeouts and frequencies that makes sense

I recall one time I was working on a distributed processing system and each node had its own watchdog setup The problem we had was we were sending a lot of data to different nodes and some of them ended up being overwhelmed during large transfers because we didn't have proper rate limiting at the application layer this resulted in watchdogs reboots and the worst thing was these reboots were happening at completely different times for each system making it a debugging nightmare in the end we ended up fixing the application but it's just a reminder that the watchdog is a fail-safe mechanism it is not a substitute for good software design and proper implementation.

Now about advanced use cases you might want to set up specific notifications on watchdog triggers for example you might want to trigger a memory dump or a debug log when the system reboots here is a simple systemd service unit example that can trigger some custom script before the reboot happens or after

```systemd
[Unit]
Description=My Watchdog Service
Requires=watchdog.service
After=watchdog.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/watchdog_trigger.sh

[Install]
WantedBy=multi-user.target
```

And the script itself `/usr/local/bin/watchdog_trigger.sh` might be something like this

```bash
#!/bin/bash

# Log the watchdog trigger with a timestamp
date >> /var/log/watchdog_trigger.log

# You can add other actions here such as collecting logs
# dumping memory etc.

# Then do nothing, let the reboot happen.
```

This way the script `/usr/local/bin/watchdog_trigger.sh` will run before a system reboot caused by the watchdog, you can extend this script to perform whatever action is required but the important thing is it is another system process doing it so it will be independent from your application process thus increasing the chances it will run even with the main application being down

Also when setting up your watchdogs it is useful to monitor the health of the watchdog from outside the application if something is failing and not triggering a watchdog you will have no idea of what is going on in the system so some form of remote health check is always a good idea because its not ideal to rely on the system process itself to tell you everything is fine so keep that in mind

Resources for a deep dive on watchdog implementations I'd recommend checking out the Linux Kernel documentation specifically the watchdog driver documentation in `Documentation/watchdog` it's surprisingly readable you will find the available ioctl definitions there and more background on how the device driver works also "Operating System Concepts" by Silberschatz et al is a good place to get a firm grasp on the theoretical side of these things although it does not cover specifically the linux kernel implementation

Also do you know why programmers prefer dark mode? Because light attracts bugs! I’m sorry I couldn’t help it

Anyway hopefully all of that is enough to get you started and avoid some of the pitfalls I've stumbled into over the years watchdogs are essential to embedded and production environments but they do need to be handled with care and a good dose of paranoia so good luck with your implementation and happy debugging
