---
title: "How do I make an erasing tool in an Android drawing app?"
date: "2024-12-23"
id: "how-do-i-make-an-erasing-tool-in-an-android-drawing-app"
---

Alright,  Implementing an erasing tool in an Android drawing application might seem straightforward at first glance, but as anyone who's tried it knows, it requires a bit more consideration than simply changing the paint color to white. It's about carefully managing how we modify the underlying bitmap and how we reflect those changes on the user interface. I remember way back in my early Android development days, working on a prototype for a collaborative sketching app, that's where I really hammered out the details of this issue. The challenges weren't necessarily complex individually, but getting all the pieces to play nice with smooth, performant results took some dedicated work. Let's break down the key aspects.

Essentially, the core of the erasing function boils down to modifying the pixel data of the canvas, typically represented as a bitmap. We don't technically "erase," but instead, we *overwrite* the existing pixels with transparent pixels (or white pixels if you prefer to emulate an "eraser" look). There are a few approaches to achieve this, and the correct one really depends on your needs regarding performance and visual quality.

One straightforward method, especially useful for simpler drawing applications, is to simulate erasing by drawing a path with a transparent paint object over the existing content. This works by setting the `PorterDuff.Mode` of the paint to `CLEAR`. Essentially, wherever you draw with this paint, the underlying pixels are cleared.

Here’s a code snippet demonstrating that approach:

```java
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.view.MotionEvent;
import android.view.View;

public class EraserView extends View {

    private Bitmap bitmap;
    private Canvas canvas;
    private Paint paint;
    private Path path;
    private float lastX, lastY;

    public EraserView(Context context) {
        super(context);
        init();
    }

    private void init() {
        paint = new Paint();
        paint.setAntiAlias(true);
        paint.setDither(true);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeWidth(12); // Adjust as needed
        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR)); // The key here!

        path = new Path();
    }

    public void setBitmap(Bitmap bitmap){
       this.bitmap = bitmap;
       this.canvas = new Canvas(bitmap);
       invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        if (bitmap != null) {
            canvas.drawBitmap(bitmap, 0, 0, null);
        }

        if (path != null){
            canvas.drawPath(path, paint);
        }


    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                lastX = x;
                lastY = y;
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                float dx = Math.abs(x - lastX);
                float dy = Math.abs(y - lastY);
                if (dx >= 4 || dy >= 4) {
                    path.quadTo(lastX, lastY, (x + lastX)/2, (y + lastY)/2);
                    lastX = x;
                    lastY = y;
                    invalidate();
                }
                break;
            case MotionEvent.ACTION_UP:
                path.lineTo(x,y);
                canvas.drawPath(path, paint); // Final path draw to the bitmap canvas
                path.reset(); // Clear the path
                invalidate();
                break;

        }
        return true;
    }
}

```

This snippet sets up a basic `EraserView` that uses `PorterDuff.Mode.CLEAR` to remove content. You'll need to pass the bitmap you're drawing on to this view to make it work.

Another method, potentially more flexible depending on the granularity you need, involves directly manipulating the bitmap's pixel array. This isn't something you'd want to do for every single point in the path during an onTouch move event as it will kill performance, but it works for when you have discrete eraser shapes (for example, a square eraser).

Here’s a sample showcasing that:

```java
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.MotionEvent;
import android.view.View;

public class BitmapEraserView extends View {

    private Bitmap bitmap;
    private Canvas canvas;
    private Paint paint;
    private int eraserSize = 20; // Adjust size as needed

    public BitmapEraserView(Context context) {
        super(context);
        init();
    }

    private void init() {
        paint = new Paint();
    }

    public void setBitmap(Bitmap bitmap){
       this.bitmap = bitmap;
       this.canvas = new Canvas(bitmap);
       invalidate();
    }


    @Override
    protected void onDraw(Canvas canvas) {
        if (bitmap != null) {
            canvas.drawBitmap(bitmap, 0, 0, null);
        }
    }


    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if(bitmap == null) return true;

        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_MOVE:
            case MotionEvent.ACTION_DOWN:
                eraseArea(x,y);
                 invalidate();
                break;
        }
        return true;
    }

   private void eraseArea(float x, float y){
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        int startX = (int) Math.max(0, x - eraserSize / 2);
        int startY = (int) Math.max(0, y - eraserSize / 2);
        int endX = (int) Math.min(width, x + eraserSize / 2);
        int endY = (int) Math.min(height, y + eraserSize / 2);

       for (int i = startY; i < endY; i++) {
            for (int j = startX; j < endX; j++) {
                pixels[i * width + j] = Color.TRANSPARENT;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
   }
}
```

This example reads the entire pixel array, modifies the relevant section, and then sets it back. While this provides a lot of control, direct manipulation is significantly less efficient. It’s suitable for situations where you don’t need to erase very complex or dynamic shapes, such as a pre-defined square or circular eraser.

Finally, for those aiming for very complex drawing apps with very performant erasing, consider leveraging hardware acceleration and low level libraries like OpenGL or Vulkan if you plan to do it at scale, especially with very large bitmaps or high resolution drawings. This approach can give you near-native performance with pixel manipulation, and its especially useful when you want to do things like feather the edges of the eraser stroke. It’s far more complex though and probably outside the scope of this immediate discussion.

For additional reading, I recommend looking into "Android Graphics Architecture" documentation and sample code. Further, exploring the 'Advanced Graphics Techniques' section of a standard Android development textbook can provide deeper insights into bitmap manipulation and performance optimization. Specifically, understand the differences between `Bitmap.getPixels()` and `Bitmap.setPixels()` is critical when using this bitmap manipulation approach. Lastly, study OpenGL and Vulkan tutorials if you are planning for complex, high-performance drawing applications; this can enhance your rendering pipelines.

Ultimately, selecting the "best" technique for implementing an eraser depends entirely on the requirements of your application. The `PorterDuff` method with a `CLEAR` operation is generally simpler and more efficient for real-time drawing, but direct bitmap manipulation offers maximum control. Remember to always consider performance implications when working with pixel data, particularly on lower-end devices. My experience has taught me that iterative testing and targeted optimization are essential for delivering a smooth and satisfying user experience.
