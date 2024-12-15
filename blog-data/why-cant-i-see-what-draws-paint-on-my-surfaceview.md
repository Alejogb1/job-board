---
title: "Why can't I see what draws paint on my SurfaceView?"
date: "2024-12-15"
id: "why-cant-i-see-what-draws-paint-on-my-surfaceview"
---

so, you're not seeing the paint calls on your surfaceview, huh? been there, done that, got the t-shirt. let me tell you, it's a classic problem and there are multiple places where things can go sideways. it's less about the drawing itself and more about the lifecycle of the surfaceview and the thread handling the drawing. let's break it down like a good old bug hunt.

first off, surfaceview is not your regular view. it operates on a separate surface, which is kind of like a blank canvas managed by the os. drawing to this surface needs to happen on a background thread, or you'll lock up the ui thread, and nobody wants that. we're not drawing directly on the view in the same way we would with a regular view's ondraw().

the most common pitfall is that the surface itself might not be valid or ready yet when you try to draw on it. this happens when you start the drawing thread too early, before the surfaceview is properly created. surfaceview has a surfaceholder which tells you about the surface's state. you need to wait for the surfaceholder to tell you that its surface has been created with the `surfacecreated` callback before attempting any drawing operation. this is paramount.

i remember one time, way back when i was coding an early android game, i was so convinced my drawing code was broken. spent hours refactoring, optimizing what not. then a colleague pointed out i was kicking off the drawing loop in `oncreate` of my activity, and the surface was not initialized yet. it was embarrassing, i tell you. rookie mistake, but we've all done something similar. the fix was to wait for the `surfacecreated` call, and it worked like a charm.

another frequent culprit is forgetting to properly lock the canvas before drawing and unlock it afterward. the surfaceview canvas is a shared resource, and multiple threads could attempt to access it at the same time. this would lead to data corruption. you need to lock the canvas before you draw on it, then unlock the canvas afterwards, posting your changes to the screen.

here is some basic structure:

```java
public class mysurfaceview extends surfaceview implements surfaceholder.callback {

    private drawthread drawthread;
    private surfaceholder surfaceholder;
    private boolean isrunning = false;


    public mysurfaceview(context context, attributeset attrs) {
        super(context, attrs);
        surfaceholder = getsurfaceholder();
        surfaceholder.addcallback(this);

    }

    @override
    public void surfacecreated(surfaceholder holder) {
        drawthread = new drawthread(holder);
        isrunning = true;
        drawthread.start();

    }

    @override
    public void surfacechanged(surfaceholder holder, int format, int width, int height) {

    }

    @override
    public void surfacedestroyed(surfaceholder holder) {
       isrunning = false;
       try {
            drawthread.join();
       } catch (interruptedexception e){
           e.printstacktrace();
       }

    }

    public void pause() {
        isrunning = false;
        while (true) {
            try {
                drawthread.join();
            } catch (interruptedexception e) {
               // retry
               continue;
            }
            break;
        }
        drawthread = null;
    }


    class drawthread extends thread {
        private surfaceholder surfaceholder;

        public drawthread(surfaceholder surfaceholder) {
            this.surfaceholder = surfaceholder;
        }

        @override
        public void run() {
            while (isrunning) {
                canvas canvas = null;
                try {
                    canvas = surfaceholder.lockcanvas();
                    if (canvas != null){
                        synchronized (surfaceholder) {
                            //your drawing calls should be placed here
                            canvas.drawcolor(color.black);
                            canvas.drawcircle(100, 100, 50, new paint());
                        }
                    }
                } finally {
                    if (canvas != null) {
                        surfaceholder.unlockcanvasandpost(canvas);
                    }
                }
            }
        }
    }
}

```

in this example, `drawthread` manages drawing. it waits for a valid canvas, locks it, does some drawing (a black background with a circle in this case), unlocks it, and then post the canvas to the screen. notice how we are using the `surfacecreated` callback to initialize the thread, and `surfacedestroyed` to stop it. the `synchronized` block in the thread could be used to ensure that only one thread is drawing at a time, but in this example, we are already executing the code inside a single thread.

now, let's take a step back and consider how your drawing is done. is it inside the draw loop? are you using basic paint operations or something more complex like bitmaps? often, issues are not directly related to surfaceview itself, but rather to mistakes in how drawing calls are handled. i've spent hours debugging why a bitmap was showing as black, only to find out it was the colorspace format in the bitmap itself was not set properly or the bitmap was not properly loaded.

sometimes, if your drawing is very intensive, you might not even see what you are drawing because the frames are just getting dropped, especially in slower devices. frame skipping is common if you are not careful about optimization. consider using `game` or `rendering` patterns (like a single game loop) with fixed time step, rather than just constantly redrawing the view with every small change. this can be achieved by using a `fixed framerate` using `systemclock.uptimeMillis()`. something like this:

```java
public class surfaceviewextended extends surfaceview implements surfaceholder.callback {

    private drawthread drawthread;
    private surfaceholder surfaceholder;
    private boolean isrunning = false;
    private final long target_fps = 60;
    private final long ms_per_frame = 1000 / target_fps;


    public surfaceviewextended(context context, attributeset attrs) {
        super(context, attrs);
        surfaceholder = getsurfaceholder();
        surfaceholder.addcallback(this);

    }

    @override
    public void surfacecreated(surfaceholder holder) {
        drawthread = new drawthread(holder);
        isrunning = true;
        drawthread.start();

    }

    @override
    public void surfacechanged(surfaceholder holder, int format, int width, int height) {

    }

    @override
    public void surfacedestroyed(surfaceholder holder) {
        isrunning = false;
        try {
            drawthread.join();
        } catch (interruptedexception e){
            e.printstacktrace();
        }
    }

    public void pause() {
        isrunning = false;
        while (true) {
            try {
                drawthread.join();
            } catch (interruptedexception e) {
                // retry
                continue;
            }
            break;
        }
        drawthread = null;
    }


    class drawthread extends thread {
        private surfaceholder surfaceholder;

        public drawthread(surfaceholder surfaceholder) {
            this.surfaceholder = surfaceholder;
        }

        @override
        public void run() {
           long start_time;
           long time_diff;
           long sleep_time;
           while (isrunning) {
               start_time = systemclock.uptimemillis();
                canvas canvas = null;
                try {
                    canvas = surfaceholder.lockcanvas();
                    if (canvas != null){
                        synchronized (surfaceholder) {
                            //your drawing calls should be placed here
                            canvas.drawcolor(color.black);
                            canvas.drawcircle(100, 100, 50, new paint());
                        }
                    }
                } finally {
                    if (canvas != null) {
                        surfaceholder.unlockcanvasandpost(canvas);
                    }
                }
                time_diff = systemclock.uptimemillis() - start_time;
                sleep_time = ms_per_frame - time_diff;
                if (sleep_time > 0) {
                   try {
                       sleep(sleep_time);
                   } catch (interruptedexception e) {
                       e.printstacktrace();
                   }
                }
            }
        }
    }
}
```
this code adds a fixed framerate of 60 fps. not something you would need most of the time, but it shows how you could control how many frames are drawn in a given time interval. note that this could introduce some stuttering depending on how much is being drawn, and that the sleep can be imprecise.

if you are still having trouble, add some logging to your drawing loop. print out the canvas instance before you attempt to lock it and after. see if it is null or valid. that can save a lot of time. another thing you might want to check is if there are any other views obscuring or on top of your surfaceview, if your surfaceview is translucent or semi-transparent, or if there are transformations applied in any view above your surface view.

now, here’s one of my favourite silly tech jokes: why did the programmer quit his job? because he didn't get arrays. (sorry, i couldn't resist).

debugging drawing issues on surfaceview can be a pain. but if you follow the steps above, you will find the culprit easily. and, it is something that many developers stumble upon. also, reading into more advanced opengl programming (which surfaceview does under the hood) could provide more insight. for advanced graphics rendering, i would also recommend reading into "opengl es 3.0 programming guide" by david shreiner. for android specific info about surfaceview, the android developers documentation is a treasure trove (even if sometimes a bit verbose).

here is a more complex example of drawing using bitmaps, and moving objects around the surface. i've added some comments to the code to guide you. this is a classic example that has all the basic features we were talking about:

```java

import android.content.context;
import android.graphics.bitmap;
import android.graphics.bitmapfactory;
import android.graphics.canvas;
import android.graphics.color;
import android.graphics.paint;
import android.os.systemclock;
import android.util.attributeset;
import android.view.surfaceholder;
import android.view.surfaceview;
import com.example.myproject.r; // replace with your r class path

public class gameview extends surfaceview implements surfaceholder.callback {

    private drawthread drawthread;
    private surfaceholder surfaceholder;
    private boolean isrunning = false;
    private final long target_fps = 60;
    private final long ms_per_frame = 1000 / target_fps;
    private bitmap mybitmap;
    private float x = 100; // x position of bitmap
    private float y = 100; // y position of bitmap
    private float dx = 5;  // x speed
    private float dy = 3; // y speed

    public gameview(context context, attributeset attrs) {
        super(context, attrs);
        surfaceholder = getsurfaceholder();
        surfaceholder.addcallback(this);
        mybitmap = bitmapfactory.decoderesource(getresources(), r.drawable.my_image); //replace with your image resource
    }

    @override
    public void surfacecreated(surfaceholder holder) {
        drawthread = new drawthread(holder);
        isrunning = true;
        drawthread.start();
    }

    @override
    public void surfacechanged(surfaceholder holder, int format, int width, int height) {
    }

    @override
    public void surfacedestroyed(surfaceholder holder) {
        isrunning = false;
        try {
            drawthread.join();
        } catch (interruptedexception e) {
            e.printstacktrace();
        }
        if (mybitmap != null) mybitmap.recycle();
    }

    public void pause() {
        isrunning = false;
        while (true) {
            try {
                drawthread.join();
            } catch (interruptedexception e) {
                continue;
            }
            break;
        }
        drawthread = null;
    }


    class drawthread extends thread {
        private surfaceholder surfaceholder;

        public drawthread(surfaceholder surfaceholder) {
            this.surfaceholder = surfaceholder;
        }

        @override
        public void run() {
           long start_time;
           long time_diff;
           long sleep_time;
            while (isrunning) {
                start_time = systemclock.uptimemillis();
                canvas canvas = null;
                try {
                    canvas = surfaceholder.lockcanvas();
                    if (canvas != null) {
                        synchronized (surfaceholder) {
                           canvas.drawcolor(color.black); // clear canvas every frame.
                            canvas.drawbitmap(mybitmap, x, y, new paint()); //draw bitmap at position x,y
                            //update position
                            x += dx;
                            y += dy;
                            // bounce when touching edges
                            if (x + mybitmap.getwidth() > canvas.getwidth() || x < 0) dx = -dx;
                            if (y + mybitmap.getheight() > canvas.getheight() || y < 0) dy = -dy;
                        }
                    }
                } finally {
                    if (canvas != null) {
                        surfaceholder.unlockcanvasandpost(canvas);
                    }
                }
                time_diff = systemclock.uptimemillis() - start_time;
                sleep_time = ms_per_frame - time_diff;
                if (sleep_time > 0) {
                    try {
                        sleep(sleep_time);
                    } catch (interruptedexception e) {
                        e.printstacktrace();
                    }
                }
            }
        }
    }
}
```
this example also shows how to load bitmaps and move them around the screen. also how to clean them up when the surfaceview is destroyed. it is pretty much a simple moving sprite example.

finally, remember to always recycle your bitmaps when they are no longer needed. android resources are limited and you don't want to run out of memory. there are many ways to optimize your code, but these examples should help you start on the right track.
