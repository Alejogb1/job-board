---
title: "How to make a background transparent using a mask in a java canvas and paint Android - Is it possible to declare an alpha mask?"
date: "2024-12-15"
id: "how-to-make-a-background-transparent-using-a-mask-in-a-java-canvas-and-paint-android---is-it-possible-to-declare-an-alpha-mask"
---

alright, so you're looking at making a background transparent using a mask on an android canvas, and wondering about alpha masks specifically. i've been there, wrestled with this beast a few times myself. it's definitely doable, and yes, you can declare an alpha mask, but it's not *quite* as straightforward as just setting a transparency value on a single paint object.

first off, let's talk about what's actually happening. a canvas in android is essentially a surface on which you draw. think of it like a digital piece of paper. when you draw something, you're essentially layering paint onto that paper. what we want to achieve is selectively make parts of that paper (the canvas) transparent based on a mask.

the way you do this involves using the `porterduffxfermode` class, which is part of the android graphics api. this class provides a bunch of different blending modes, each describing how a source and destination paint should interact. in our case, we're interested in using a mask, that we will paint on the canvas and then blend. it's not a simple alpha channel applied to the background of the view itself like some people think.

now, let's walk through how i typically handle this. i remember back when i was working on this drawing app, i needed to let users use a brush with a specific shape but then i needed the background to show and also see the brush shape. that's when i really got into masks. i initially started by trying to directly affect the view, what a mess. later i started with a layer, a bitmap with the mask and then painting over the original canvas, it was still a disaster until i came across porterduffxfermode.

let's start by breaking down the code, imagine you have a bitmap `sourceBitmap`, which is what you want to apply the mask to. let's say it is your background image. and another bitmap, `maskBitmap`, which represents our mask. the mask should ideally be black where you want the source to be transparent and white where you want the source to show. remember to avoid colors when you make your mask bitmap, grayscale, not rgb.

first we need to create a paint instance and set it to a mode where the source color will replace what's below, if this was your base canvas, it would be the same thing as putting your base canvas as the background.

```java
    Paint paint = new Paint();
    paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC));
```

then you have to create a bitmap that will be used as a layer, this is where the masking and painting will happen.

```java
    Bitmap resultBitmap = Bitmap.createBitmap(sourceBitmap.getWidth(), sourceBitmap.getHeight(), Bitmap.Config.ARGB_8888);
    Canvas resultCanvas = new Canvas(resultBitmap);
    resultCanvas.drawBitmap(sourceBitmap, 0, 0, paint);
```

then you change the paint mode, to the mask mode where only the destination is shown where the source is not transparent, in this case is where the mask bitmap is black.

```java
    paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_IN));
    resultCanvas.drawBitmap(maskBitmap, 0, 0, paint);
```

and finally you draw the result on your canvas.

```java
    canvas.drawBitmap(resultBitmap, 0, 0, null);
```

here's a full method that encapsulates this process:

```java
    public static Bitmap applyAlphaMask(Bitmap sourceBitmap, Bitmap maskBitmap) {
        Bitmap resultBitmap = Bitmap.createBitmap(sourceBitmap.getWidth(), sourceBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas resultCanvas = new Canvas(resultBitmap);

        Paint paint = new Paint();
        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC));
        resultCanvas.drawBitmap(sourceBitmap, 0, 0, paint);

        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_IN));
        resultCanvas.drawBitmap(maskBitmap, 0, 0, paint);

        return resultBitmap;
    }
```

a common mistake people have is not making sure that the size of both bitmaps are the same. if you are experiencing weird artifacts, the bitmaps are the first thing you should check.

now, regarding declaring an alpha mask, the mask itself is essentially a bitmap, which is just a matrix of pixel data. the alpha channel in a bitmap defines the transparency of each pixel. so, when you create `maskBitmap`, you can programmatically control which pixels are opaque and which ones are transparent by controlling the alpha channel of each pixel. but this is not the alpha you are setting on a paint.

here's how you might generate a simple alpha mask with a shape, imagine a circle, a mask of circle in the center of the mask bitmap:

```java
    public static Bitmap generateCircularMask(int width, int height, int radius) {
        Bitmap maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(maskBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.WHITE);
        canvas.drawCircle(width / 2f, height / 2f, radius, paint);
        return maskBitmap;
    }
```

in this snippet, the area inside the circle will be white (opaque), and the remaining area of the `maskBitmap` will be black (transparent because its default pixel value is 0), resulting in a circle shape.

when i was building that drawing app i had to create all those masks from shapes, from user drawn paths and more. i even had to deal with smoothing and antialising problems so i ended up implementing a small graphics engine on my own. i learned a lot in the process.

another thing i want to mention, is that `porterduffxfermode` uses source and destination. source is usually the bitmap drawn later and destination the bitmap already present on canvas. the modes are the operation between the source and the destination. so always keep in mind the order you draw.

also something that might catch someone by surprise is that it might have some performance penalty. the reason for this is that it needs to calculate pixel by pixel the blending operation. you should avoid it if you need to do it in a very long list of items, since it can affect the ui performance. i even had to add some threading to offload that work on a thread, the app almost crashed due to that masking process.

the `porterduff.mode.dst_in` is the key here: it only shows the area of the destination where the source is not transparent, effectively using the source as a mask on the destination.

a good book i would recommend to understand more about graphic engines, is "computer graphics with opengl" by hearne and baker. even though it uses open gl, the theory and concepts are really useful.

remember that while i showed you with bitmaps you can use also use shapes, text, images, whatever you want, on the source or on the mask, just as long as you set the paint correctly and in the end just draw it on the right place. the canvas in android is like that, a blank paper, just waiting for you to paint something. just don't use a potato to draw because the results are going to be… well, not the best.

let me know if you have more questions or encounter specific issues; i've spent too much time on graphics rendering, feel like i've seen it all.
