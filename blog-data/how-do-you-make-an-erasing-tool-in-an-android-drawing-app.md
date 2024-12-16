---
title: "How do you make an erasing tool in an Android drawing app?"
date: "2024-12-16"
id: "how-do-you-make-an-erasing-tool-in-an-android-drawing-app"
---

Alright, let's talk about implementing an eraser tool in an Android drawing application. I’ve dealt with this specific issue a few times over the years, and it’s more nuanced than it might initially appear. The core concept is quite straightforward: instead of drawing with a color, you're drawing with a ‘transparent’ brush, effectively removing existing content. However, the implementation details can greatly impact performance and user experience.

First, let's break down the fundamental approaches. The common method involves using `PorterDuff.Mode.CLEAR` in your `Paint` object. This mode, when applied in conjunction with `Canvas.drawPath` or similar drawing operations, replaces any existing pixel with a transparent pixel. I find it's the workhorse method for most cases. It's generally efficient, leveraging hardware acceleration when available, and offers decent visual results. The primary gotcha to watch out for here is your drawing surface. Are you drawing directly onto a `Bitmap` managed by a `View`? Are you using layers? Or are you doing something more complex like using a multi-layer canvas structure? The way you manage your canvas heavily dictates how these operations will interact.

My past experience working on a collaboratively edited whiteboard application showed me the challenges when dealing with such a seemingly simple task. There, performance became the biggest concern. Erasing operations weren't simply single-stroke, often involving several overlapped sections that required constant refreshing. Managing the paint operations effectively was key to providing a responsive UI. This involved making sure the erase action itself was done on an off-thread, and the results were properly synchronized with the ui thread using the `postInvalidate()` mechanism or similar view update notifications.

Let’s take a look at some code snippets. Firstly, a basic implementation of the eraser with `PorterDuff.Mode.CLEAR`. We'll assume a simple drawing `View` where `canvas` is the drawing canvas, `paint` is our brush configuration, and `path` is the user's finger movement being converted to a drawing path:

```java
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.view.MotionEvent;
import android.view.View;

public class DrawingView extends View {
    private Bitmap bitmap;
    private Canvas canvas;
    private Paint paint;
    private Path path;
    private float previousX, previousY;

    public DrawingView(Context context) {
        super(context);
        init();
    }

    private void init() {
         paint = new Paint();
        paint.setAntiAlias(true);
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeWidth(5f);

        path = new Path();
    }
    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(bitmap);
        canvas.drawColor(Color.WHITE);
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawBitmap(bitmap, 0, 0, null);
        //optional path preview during draw operation
        canvas.drawPath(path, paint);
    }


    @Override
    public boolean onTouchEvent(MotionEvent event) {
      float x = event.getX();
      float y = event.getY();

        switch (event.getAction()){
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                previousX = x;
                previousY=y;
                return true;

            case MotionEvent.ACTION_MOVE:
                float dx = Math.abs(x - previousX);
                float dy = Math.abs(y - previousY);
                if (dx >= 4 || dy >= 4) {
                  path.quadTo(previousX, previousY, (x + previousX) / 2, (y + previousY) / 2);
                  previousX = x;
                  previousY = y;
                }
                break;
            case MotionEvent.ACTION_UP:
                 path.lineTo(x,y);
                 canvas.drawPath(path,paint);
                 path.reset();
                 break;
        }
        invalidate();
        return true;
    }
   public void useEraser(){
        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
        paint.setStrokeWidth(20f);
        paint.setStyle(Paint.Style.STROKE);

    }
    public void useBrush(){
        paint.setXfermode(null);
        paint.setStrokeWidth(5f);
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
    }

}
```

In this example, I have included the basic touch event capture to handle motion events, a canvas creation in onSizeChanged, and a method called `useEraser()` to switch our tool to an eraser. The `PorterDuffXfermode` and `PorterDuff.Mode.CLEAR` is where the erasing happens. The `useBrush` method will reset the `paint` so we can start drawing again. You can call these methods when the user clicks a button to switch between the modes.
This is the basic setup that most implementations follow, and it works pretty well for simple drawing.

Now, let’s consider a more complex example where you might want to have a different eraser shape, or a smoother erase effect. You can achieve this by drawing a blurred circle, using it as a mask to affect transparency with  `PorterDuff.Mode.DST_OUT`. This offers a more natural, gradual edge:

```java

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.BlurMaskFilter;
import android.view.MotionEvent;
import android.view.View;

public class SoftEraserView extends View {
    private Bitmap bitmap;
    private Canvas canvas;
    private Paint paint;
    private Paint eraserPaint;
    private Path path;
    private float previousX, previousY;
    private float eraserSize = 30f;


    public SoftEraserView(Context context) {
        super(context);
        init();
    }

    private void init() {
        paint = new Paint();
        paint.setAntiAlias(true);
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeWidth(5f);

        eraserPaint = new Paint();
        eraserPaint.setAntiAlias(true);
        eraserPaint.setColor(Color.TRANSPARENT);
        eraserPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_OUT));
        eraserPaint.setStyle(Paint.Style.FILL);
        eraserPaint.setMaskFilter(new BlurMaskFilter(eraserSize / 2f, BlurMaskFilter.Blur.NORMAL));



        path = new Path();
    }
    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(bitmap);
        canvas.drawColor(Color.WHITE);
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawBitmap(bitmap, 0, 0, null);
        //optional path preview during draw operation
        canvas.drawPath(path, paint);

    }


    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()){
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                previousX = x;
                previousY=y;
                return true;

            case MotionEvent.ACTION_MOVE:
                float dx = Math.abs(x - previousX);
                float dy = Math.abs(y - previousY);
                if (dx >= 4 || dy >= 4) {
                    path.quadTo(previousX, previousY, (x + previousX) / 2, (y + previousY) / 2);
                    previousX = x;
                    previousY = y;
                }
                break;
            case MotionEvent.ACTION_UP:
                path.lineTo(x,y);
                canvas.drawPath(path,paint);
                path.reset();
                break;
        }
        invalidate();
        return true;
    }

    public void useEraser(){
        paint.setXfermode(null);
    }
    public void useBrush(){
        paint.setXfermode(null);
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5f);
    }

    public void erase(float x, float y){
        canvas.drawCircle(x, y, eraserSize , eraserPaint );
        invalidate();
    }

    public void setEraserSize(float size){
        eraserSize = size;
        eraserPaint.setMaskFilter(new BlurMaskFilter(eraserSize / 2f, BlurMaskFilter.Blur.NORMAL));
    }

}
```

Here, we have the `SoftEraserView` class. Instead of applying the erase mode directly to the paint, the `erase` method will draw a blurred transparent circle at the touch point on the canvas. This circle is drawn with `DST_OUT` mode on the `eraserPaint` and will create the soft erase effect. The `setEraserSize` method lets the user control the size. This technique provides a different visual appearance than the first, giving you a softer erasure style.

Finally, sometimes, particularly when handling large, complex drawings, erasing individual strokes might not be the most efficient approach. If you manage your drawing data as a series of geometric shapes (lines, curves, etc.), you can implement a vector eraser. Instead of erasing pixels on the canvas, you would calculate which shapes intersect with the current eraser path and remove those shapes from the drawing data. This can provide better performance and allows for potentially more advanced features such as undo/redo functionalities.

As for resources, I would recommend diving into the Android Graphics API documentation thoroughly. Specifically, the sections on `Canvas`, `Paint`, `Path`, and `PorterDuff` modes. Also, a deeper study of the `android.graphics` package, including resources like the “Graphics Architecture” section in the Android documentation, can help understand hardware acceleration limitations and optimizations. “Android Graphics Programming” by Robert M. Love provides a comprehensive overview of these concepts. Beyond that, "OpenGL ES 2.0 Programming Guide" by Dave Shreiner can be a useful resource if you are considering pushing the graphic rendering to lower level libraries for complex scenarios. Understanding graphic pipelines is key when going to low-level graphics. Understanding when to offload canvas manipulation on non ui threads is paramount when dealing with performance issues. Finally, a review of common 2D graphic algorithms and optimizations can also greatly improve the overall implementation.

These examples and resources should give you a solid foundation for creating a powerful eraser tool in your application. Remember that the best approach ultimately depends on your specific requirements, performance targets, and design preferences.
