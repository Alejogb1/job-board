---
title: "linux software watchdog usage explanation?"
date: "2024-12-13"
id: "linux-software-watchdog-usage-explanation"
---

Alright so software watchdogs on Linux huh I've been there trust me I've seen things. Like kernel panics happening at 3 am because a rogue thread decided to take an unplanned vacation in memory land.  Been wrestling with these little lifesavers for years feels like. It's a bit like having a responsible friend who checks on you periodically just to make sure you haven't gone off the rails. 

Let's break this down no fluff just the code and the concepts you actually need because I'm guessing you're probably in the middle of a debug session right now and the clock is ticking.

First off what's the deal with a software watchdog specifically Well think of it as a timer that your application or kernel code periodically kicks to say "hey I'm still alive I'm still doing my thing". If that kick doesn't happen within a certain timeframe then the watchdog assumes something is wrong and takes action usually that means a reboot.  Yeah a hard reboot but it's better than letting your embedded system get stuck in a loop spinning a cpu core to 100 percent forever.

Now why do we even use them? Well if you have embedded devices or critical systems software watchdogs are your best friend. Think of automated machines medical equipment or industrial controllers. Things that really cant get stuck or crash without causing real world problems. A regular process can crash but a process running a critical operation that's not restarted can be catastrophic.

Okay let's get practical because code talks. We have a couple of ways to implement watchdogs on Linux. One way is directly using the /dev/watchdog device the standard. You access this as a character device and there’s a ioctl system call that's the magic for peting the dog so to speak which means resetting the timer.

Here's some basic C code for this

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/watchdog.h>
#include <errno.h>

int main() {
    int fd;
    int timeout = 15; // Watchdog timeout in seconds you can configure this usually
    int ret;

    fd = open("/dev/watchdog", O_WRONLY);
    if (fd == -1) {
        perror("Error opening watchdog device");
        return 1;
    }


    // Set the timeout
    ret = ioctl(fd, WDIOC_SETTIMEOUT, &timeout);
     if (ret == -1) {
       perror("Error setting watchdog timeout");
        close(fd);
        return 1;
    }

    while (1) {
        // Pet the watchdog
        ret = ioctl(fd, WDIOC_KEEPALIVE, 0);
          if (ret == -1) {
          perror("Error petting the watchdog");
           close(fd);
           return 1;
          }

        sleep(10); // Sleep a bit longer than we pet the dog to make sure it happens before timeout
    }


    close(fd); // Never gets here but it's good habit
    return 0;
}
```

Compile this with gcc your_file_name.c -o your_executable. Now run it with root permissions because you need that to access the device. You will have to make sure the watchdog kernel module is loaded I think modprobe wdt is what you want for most default kernels. Also make sure your kernel is compiled with watchdog support enabled which is a big deal of its own. And if you're running this on a virtual machine some cloud providers might not give you access to the watchdog so watch out for that.

I spent days debugging a similar issue on a custom embedded device the watchdog would trigger randomly. Turned out some external communication hardware was locking up and blocking the watchdog feed from the application. It felt like the dog was barking at me and I didn't know why. It was all the hardware's fault that time.

Now there's a different approach using the `systemd` watchdog integration which is more elegant if you are running `systemd` which most of the modern Linux distros are using these days. `systemd` provides this mechanism that allows services to notify `systemd` about their "aliveness" with periodic messages and `systemd` can restart the service or reboot if it detects a failure. We use this mostly on servers.

Here's how you would implement watchdog integration with `systemd`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <systemd/sd-daemon.h>

int main() {
  int ret;

  while (1) {
      // Perform your application's task here
      printf("My Application is still alive\n");

      // Notify systemd that we are still alive
      ret = sd_notify(0, "WATCHDOG=1");
         if (ret < 0) {
          perror("Error Notifying systemd");
           return 1;
         }

      sleep(10); // Sleep a bit to ensure we check alive before timeout
  }

  return 0;
}
```

Compile this as well `gcc your_file_name.c -o your_executable -lsystemd` also run it like before with root or sudo or as a systemd service. You need to tell `systemd` about this application in a systemd service file like this which you can create with a name like `yourapp.service` and add in your `/etc/systemd/system` directory

```systemd
[Unit]
Description=My Watchdog Example Application
After=network.target

[Service]
Type=notify
ExecStart=/path/to/your/executable
WatchdogSec=15
Restart=always

[Install]
WantedBy=multi-user.target
```

This part is crucial you have to tell systemd how to treat your application as a service. `WatchdogSec` is crucial its the same as the time in the `/dev/watchdog` example. Make sure it matches with the timing of your `sd_notify` calls. If your service doesn’t notify `systemd` within this specified time `systemd` will restart it and that is the goal.

The `Type=notify` is also important. This tells systemd that the application is going to send notification messages via `sd_notify`.

I had a situation where we weren't restarting the service properly with systemd due to a misconfiguration of the service file itself. Systemd does its own internal checks before restarting a service and if one fails the service wont restart even if it does not call sd_notify it will not restart. You could make the service restart forever if you configure correctly a service with `Restart=always` and that's exactly what we want for critical stuff.

Okay one more example if your watchdog is kernel based you can configure it directly on the kernel's command line at the boot time. I'm assuming you're using grub here but other boot loaders should be similar.

In your `/etc/default/grub` look for the `GRUB_CMDLINE_LINUX_DEFAULT` entry and append to it the following:

```
watchdog_nowayout=1
```

Now run `sudo update-grub`. This ensures that the watchdog module is loaded early at the boot stage of the kernel. If you don’t specify this the watchdog is only loaded when you start to use `/dev/watchdog` device. And that's not enough for critical systems.

`watchdog_nowayout=1` is critical this prevents applications or other processes to close the device or disable the watchdog timer in a way that leaves the watchdog unusable. There is a watchdog_disable in the ioctl but it’s disabled by `watchdog_nowayout`.

Now the joke part. Why did the Linux admin cross the road? To get to the other side… which had a stable kernel. Okay maybe that wasn't funny.

Okay let's talk resources. For deeper understanding of the Linux watchdog subsystem I highly recommend the Linux Kernel Documentation in the source code itself. In your kernel source navigate to `Documentation/watchdog` and read `watchdog-api.txt`. And `Documentation/admin-guide/kernel-parameters.txt` is your go to reference for the kernel parameters including the watchdog ones. I've spend countless hours reading that.

Also look for embedded linux books for more context such as "Building Embedded Linux Systems" by Karim Yaghmour if you want to go deeper in this topic. It's a bit older now but a lot of these kernel level stuff is just the same old story. 

Finally always test your watchdogs thoroughly. Do a controlled failure of your application and observe if the watchdog reboots the system. A system that is not rebooting when it should is worse than a system not even trying to reboot it is creating a false sense of security. Your watchdog is only useful if it actually works when things go wrong. That's all I can think of for now hope it helps. I gotta go back to my debug session now so good luck and stay safe.
