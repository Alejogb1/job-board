---
title: "Why isn't portrait screen touch rotation working on Ubuntu 20.04?"
date: "2024-12-23"
id: "why-isnt-portrait-screen-touch-rotation-working-on-ubuntu-2004"
---

,  I remember a particularly frustrating project back in 2021 where we were deploying interactive kiosks with portrait-oriented touchscreens running Ubuntu 20.04. It seemed straightforward enough, but the touch input was consistently misaligned – a classic case of the screen rotation not playing nice with the touch input. It’s a problem that has a few layers to it, and there are specific reasons why it might not just “work” out of the box.

The core issue lies in the way Ubuntu, and linux systems in general, handle display rotation and touch input separately. The rotation you set through, say, the gnome settings or via `xrandr`, affects how the graphics are rendered on the screen. However, the touch input, handled by the kernel through the input subsystem (think drivers and the like), often doesn’t automatically re-calibrate to the rotated coordinates. It's essentially sending the coordinates based on the *physical* orientation of the device, not how the display content is *rotated*. This discrepancy leads to that frustrating off-by-a-factor-of-90-degrees or mirrored feeling when you try to interact with a rotated portrait screen.

This misinterpretation isn't necessarily a bug in Ubuntu; rather, it's a lack of automatic synchronization between these two distinct systems. The operating system needs to be explicitly told to transform touch coordinates to match the rotated display. This is typically accomplished through configuration files or commands that map the physical touch coordinates to the logical display coordinates.

Now, you might be thinking, "why isn't this handled automatically?". Well, the complexity comes from the sheer variety of touch devices and display setups out there. Universal handling would either be too specific, leading to incompatibility with other hardware, or too vague to be useful in all edge cases. A generalized solution is difficult to achieve universally, hence the common manual intervention.

There are a few common culprits in this scenario, and understanding them is key to finding a fix. Firstly, the xorg configuration may not be set up to handle transformations correctly. Xorg, the display server used in most ubuntu setups of that era, uses configuration files to manage how input devices are handled. If the touch device is recognized but not configured to apply rotation transforms, we're going to see misaligned touch input.

Secondly, wayland, if you’ve moved to it, which is unlikely in an old 20.04 system, has its own set of challenges and may not handle older configurations correctly. I'll focus more on Xorg configuration as it was the most common in 20.04.

Let's look at some specific strategies. The first approach I’d recommend is to explicitly define a touch matrix transformation in the xorg configuration file. You’ll typically find this in `/etc/X11/xorg.conf.d/`. If it doesn't exist, you may need to create a new `.conf` file, perhaps naming it `10-touchscreen.conf`. Here’s a potential snippet illustrating a clockwise 90-degree rotation:

```
Section "InputClass"
        Identifier "touchscreen-rotation"
        MatchIsTouchscreen "on"
        MatchDevicePath "/dev/input/event*"
        Option "TransformationMatrix" "0 1 0 -1 0 1 0 0 1"
EndSection
```

This code snippet applies a transformation matrix that effectively rotates the touch coordinates 90 degrees clockwise. The `MatchIsTouchscreen "on"` line ensures this transformation is only applied to detected touchscreen devices and the `MatchDevicePath` line helps with matching devices. The key is the `TransformationMatrix` itself. To understand this specific matrix ( "0 1 0 -1 0 1 0 0 1"), you might need to brush up on linear algebra – specifically matrix transformations. The first two rows control scaling, rotation, and skew, and the third row dictates translation or shifting of coordinates. You can research more on “affine transformations” to understand how these matrices work.

If a 90-degree clockwise rotation isn't what you needed, you might need different matrices. A 90-degree counter-clockwise rotation can be achieved with:

```
Section "InputClass"
        Identifier "touchscreen-rotation"
        MatchIsTouchscreen "on"
        MatchDevicePath "/dev/input/event*"
        Option "TransformationMatrix" "0 -1 1 1 0 0 0 0 1"
EndSection
```

And for a 180-degree rotation, try this:

```
Section "InputClass"
        Identifier "touchscreen-rotation"
        MatchIsTouchscreen "on"
        MatchDevicePath "/dev/input/event*"
        Option "TransformationMatrix" "-1 0 1 0 -1 1 0 0 1"
EndSection
```

Each configuration file is a snippet that can be combined with other configurations. After adding these `.conf` files, you'll need to restart the X server to ensure changes are loaded. This is usually done by logging out and back in or restarting the display manager (like `sudo systemctl restart gdm3` for gnome).

Sometimes, however, even specifying the transformation matrix isn't sufficient. I’ve seen cases where the specific input driver for the touchscreen might need adjustment. Some drivers might have their own quirks or not implement transformation correctly. You can attempt to identify your touchscreen device and its driver using `xinput --list` and `lsusb` and then you may need to research and potentially modify the driver’s source code. However, that’s advanced and hopefully these matrix solutions will help.

For a deeper dive into xorg configuration and input device management, I highly recommend examining the X.Org manual pages, accessible through `man xorg.conf`, and the documentation of the input driver for your device. Additionally, "The X Window System: Programming and Applications with Xt" by Douglas A. Young, though dated, offers valuable fundamental insights into the X system's architecture. Also the *X Input Extension* (XI2) documentation available through the freedesktop.org website will detail on various xinput tools.

Ultimately, the key to addressing touchscreen rotation issues is a methodical approach of identifying the discrepancy between display and touch coordinates, applying an appropriate transformation through the xorg configuration, and then verifying its efficacy. It’s rarely an automatic fix, but with a touch of understanding the underlying mechanics and a structured strategy, you can definitely achieve your desired results. This experience back then certainly hardened my understanding of low-level input systems, which I hope also offers a good starting point for your journey.
