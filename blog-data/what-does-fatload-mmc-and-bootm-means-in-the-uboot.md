---
title: "what does fatload mmc and bootm means in the uboot?"
date: "2024-12-13"
id: "what-does-fatload-mmc-and-bootm-means-in-the-uboot"
---

 so you're asking about `fatload mmc` and `bootm` in U-Boot right? Yeah I've been down that rabbit hole more times than I care to remember lets break it down in a way that makes sense.

First off U-Boot its a universal bootloader basically the first software that runs when your embedded system powers on or restarts It takes care of initial hardware initialization setting up memory and most importantly loading the operating system kernel.

Now `fatload mmc` specifically it's a command to load files from a FAT formatted filesystem on an MMC or SD card. Think of MMC or SD card as a little hard drive for your embedded system that can be used to store stuff like the kernel operating system image or the device tree blob or initramfs you know that kinda thing. The command structure is typically like this `fatload mmc <interface>:<partition> <load address> <filename>`

Let me show you a real-world example I had in past project oh man this was back when I was working with a Freescale i.MX6 board yeah old hardware but the principles the same. I had my kernel in a file named `zImage` and a device tree blob called `imx6-myboard.dtb` those were on an SD card. So the command I used was something like this to load the kernel in RAM

```
fatload mmc 0:1 0x10800000 zImage
```

And then another one to load the device tree blob

```
fatload mmc 0:1 0x18000000 imx6-myboard.dtb
```

Here `mmc 0:1` it means the first MMC/SD card and the first partition. `0x10800000` and `0x18000000` those are memory addresses in RAM where I wanted to load them it is important to chose right addresses because some regions are reserved and that could be cause of a no boot situation. Finally `zImage` and `imx6-myboard.dtb` are the filenames.

The important detail is that `fatload` needs a working mmc driver in uboot and that needs to be compiled in the uboot build phase that can cause you anoying debugging session if you didnt compile it in you just get "mmc not found" or something like that and you start chasing the wrong bug.

 now onto `bootm`. This command its responsible for actually starting the kernel. Its the moment all your work will pay off. In its simple use `bootm` takes a memory address argument where a valid kernel image is and then starts executing the code from that address. The kernel is a compressed image so `bootm` also handles the decompressing of the kernel so there is no need to manually do that.

And here's the thing `bootm` isn't just for raw kernels. It also works with what's called a "uImage" format which is a kernel wrapped with some metadata and checksums. It is the most used format in the uboot world. This metadata is important because it indicates to uboot information about what to load or how to load it if it's a kernel or if it is some other kind of archive.

So continuing with the previous example in our Freescale board after loading the kernel and dtb to the RAM i had something like this:

```
bootm 0x10800000 - 0x18000000
```

Here `0x10800000` is the address where the kernel was loaded and `0x18000000` it is the address of device tree blob. Uboot will pass this dtb address to kernel boot process so it can load the hardware configurations.

Now lets say you have an uImage instead of the raw zImage in that case the command would be something like this:

```
bootm 0x10800000
```

Because the uImage format contains the address of the DTB inside it so no need to provide it in arguments.

So a common workflow looks like this: `fatload mmc ...` to load kernel or uimage and device tree to specific addressses and then `bootm ...` to actually start the booting process.

A very common error you can encounter here is when the `bootm` command gives an error like "image checksum error" or "image invalid". That usually means the image file is corrupted or there is some issue with the way the file is created. The way to debug this error is comparing the checksum from the image in the uboot enviroment with the checksum of the original file in your computer this will give you a lead to solve it and it is important to understand that the checksum of a uImage can change with even small modifications.

Now a thing that many people get wrong is that the memory address arguments are in hexadecimal representation. If you provide a memory address that is not in hexadecimal you just get a uboot error and your debugging session just became much longer.

And of course these addresses need to be valid memory locations accessible by uboot some memory regions are reserved for internal uboot operations and trying to load anything there will lead to a system lock.

One time I was pulling my hair out for days until I realized my load addresses were overlapping with some other memory areas used by some uboot peripherals I was using I mean it was so frustratring and dumb at the same time. It's amazing what happens when you don't read the documentation properly. And speaking of that remember to double check you addresses because U-Boot sometimes will print the address as decimal numbers instead of hexadecimals its a thing many uboot users overlook. It is a good thing to remember that uboot is a very versatile tool but it demands precision.

And now the joke. Why did the embedded engineer bring a ladder to the debug session? Because he heard the code was running on a different level!  I'll get my coat.

To make a better understanding of the concepts the book "Embedded Linux Primer" by Christopher Hallinan is a great read for people trying to figure out low-level stuff like this. Also "Understanding the Linux Kernel" by Daniel P. Bovet and Marco Cesati will help you understanding the kernel booting process. There are a lot of things going on in the background at the startup of the kernel that many people overlook.

In conclusion: `fatload mmc` is like your loading truck grabbing files from a storage device and putting them in memory and `bootm` is the ignition key starting the machine (kernel). You need both and they need to be correct to start your system. So next time you get stuck in uboot don't just blindly copy and paste check your arguments check your addresses and check your file integrity that will save you a lot of headaches. And if you dont remember any of that just check your notes and the documentation. That's it for now happy booting.
