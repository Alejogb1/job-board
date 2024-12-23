---
title: "How can I create an erasing tool in a drawing application on Android?"
date: "2024-12-23"
id: "how-can-i-create-an-erasing-tool-in-a-drawing-application-on-android"
---

Alright, let's talk about implementing an eraser tool in an android drawing application. I've tackled this specific challenge a few times over the years, and it always boils down to understanding how you're managing your drawing data and the underlying canvas interactions. It's not simply about coloring pixels white; it's much more nuanced.

The core concept is straightforward enough: instead of drawing *on* the existing drawing, you're effectively *removing* sections of it. This removal can be achieved through a few primary methods, each with their own trade-offs concerning performance and complexity. You could consider the canvas itself as a series of layers; what we're doing with the eraser is essentially modifying these layers in a targeted manner. The approach I favor involves utilizing the 'Porter-Duff' composition modes offered by the android graphics system, specifically `PorterDuff.Mode.CLEAR`. These modes allow you to control how drawing operations affect the underlying canvas pixels, moving beyond just direct color replacement.

Let's dig into how this operates. When using `PorterDuff.Mode.CLEAR`, any pixels drawn during a canvas operation will essentially be made transparent. Imagine the eraser tool as a brush that leaves behind a transparent trail, removing whatever was present before. This, unlike simply drawing a white color over the existing pixels, retains the underlying transparency of a canvas if there is any. This leads to a more natural erasing behaviour, especially when dealing with drawings that have partial transparency.

Here's the first code snippet, illustrating how to set up an eraser tool within a custom `View`, specifically, in its `onDraw` and `onTouchEvent` methods:

```java
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class DrawingView extends View {

    private Paint paint;
    private Path path;
    private boolean isErasing = false; //Flag to indicate eraser mode

    public DrawingView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
         paint = new Paint();
        paint.setAntiAlias(true);
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeWidth(10f); //Default brush width

        path = new Path();
    }

    @Override
    protected void onDraw(Canvas canvas) {
         if(isErasing){
             paint.setXfermode(new android.graphics.PorterDuffXfermode(PorterDuff.Mode.CLEAR)); //Set erase mode.
        } else {
             paint.setXfermode(null); //Reset to normal draw mode
        }
        canvas.drawPath(path, paint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                break;
            case MotionEvent.ACTION_MOVE:
                path.lineTo(x, y);
                break;
            case MotionEvent.ACTION_UP:
                break;
            default:
                return false;
        }

        invalidate(); //Redraw view with the changes
        return true;
    }

    public void setErasing(boolean erasing){
        isErasing = erasing;
        //Change paint properties if needed, e.g., width.
        if(isErasing){
              paint.setStrokeWidth(25f); //Set eraser width
        } else {
              paint.setStrokeWidth(10f); // Reset to brush width
        }

    }


    public boolean isErasing() {
       return isErasing;
    }
    public void clear(){
         path.reset();
         invalidate();
    }
}
```

This snippet shows the basic structure. Notice the use of a boolean flag `isErasing` and the setting of the `Xfermode` in the `onDraw` method. When `isErasing` is true, the paint uses `PorterDuff.Mode.CLEAR`, otherwise, we simply set the `Xfermode` to null, and we have a normal drawing mode. The `onTouchEvent` method captures touch inputs and updates the `Path`. Also, it's important to reset xfermode to null after the drawing is complete; otherwise, you will experience drawing and UI issues. The `setErasing` method controls the state of the eraser and can, for example, change the size of the eraser.

Now, for a slightly more complex scenario. Instead of simply setting the `Xfermode`, imagine that your drawing application needs to support multiple layers. This is where using a bitmap as an intermediary step can be invaluable. You can draw on a bitmap backed `Canvas` and then compose this bitmap onto your main canvas in `onDraw()`. This allows you to selectively apply the erase effect onto specific layers.

Here's a second example, showcasing how you could incorporate a bitmap-backed `Canvas`:

```java
//Same imports as before + android.graphics.Bitmap

public class DrawingView extends View {

    private Paint paint;
    private Path path;
    private boolean isErasing = false; //Flag to indicate eraser mode
    private Bitmap drawingBitmap; // Bitmap used to draw onto, acts as our drawing layer.
    private Canvas bitmapCanvas; // Canvas backed by bitmap

    public DrawingView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        paint = new Paint();
        paint.setAntiAlias(true);
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeWidth(10f);

        path = new Path();
    }
     @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);

         //Initialize or resize the bitmap on view dimensions change.
        drawingBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        bitmapCanvas = new Canvas(drawingBitmap);
    }


    @Override
    protected void onDraw(Canvas canvas) {
         if(isErasing){
            paint.setXfermode(new android.graphics.PorterDuffXfermode(PorterDuff.Mode.CLEAR)); //Set erase mode.
        } else {
              paint.setXfermode(null); //Reset to normal draw mode
        }
        bitmapCanvas.drawPath(path, paint);
        canvas.drawBitmap(drawingBitmap, 0, 0, null);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                break;
            case MotionEvent.ACTION_MOVE:
                path.lineTo(x, y);
                break;
            case MotionEvent.ACTION_UP:
                break;
           default:
             return false;
        }

        invalidate();
        return true;
    }
      public void setErasing(boolean erasing){
        isErasing = erasing;
        if(isErasing){
              paint.setStrokeWidth(25f);
        } else {
             paint.setStrokeWidth(10f);
        }

    }


    public boolean isErasing() {
       return isErasing;
    }
    public void clear(){
         bitmapCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
         invalidate();
         path.reset();
    }

}

```

Here, the core drawing operations now occur on `bitmapCanvas`, backed by `drawingBitmap`. The `onDraw` method simply draws the `drawingBitmap` onto the main canvas, applying our `Xfermode` operations as needed. This separation gives you fine-grained control over erasing operations. Notice that we're drawing to the bitmap and then to the canvas which leads to a better performance.

For an even more sophisticated application, let's consider a scenario where you want to implement pressure sensitivity for the eraser, perhaps using the stylus API. The idea is to adjust the stroke width of the eraser based on the pressure applied by the user. Here's the third code snippet adding the pressure handling:

```java

//Same imports as before + android.view.MotionEvent
public class DrawingView extends View {

    private Paint paint;
    private Path path;
    private boolean isErasing = false;
    private Bitmap drawingBitmap;
    private Canvas bitmapCanvas;
    private float currentWidth = 10f;

    public DrawingView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
         paint = new Paint();
        paint.setAntiAlias(true);
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeWidth(currentWidth);

        path = new Path();
    }

       @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);

        drawingBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        bitmapCanvas = new Canvas(drawingBitmap);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        if(isErasing){
             paint.setXfermode(new android.graphics.PorterDuffXfermode(PorterDuff.Mode.CLEAR));
        } else {
            paint.setXfermode(null);
        }
         bitmapCanvas.drawPath(path, paint);
        canvas.drawBitmap(drawingBitmap, 0, 0, null);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                updateStrokeWidth(event);
                break;
            case MotionEvent.ACTION_MOVE:
                path.lineTo(x, y);
                updateStrokeWidth(event);
                break;
            case MotionEvent.ACTION_UP:
                break;
            default:
                return false;
        }
        invalidate();
        return true;
    }

     private void updateStrokeWidth(MotionEvent event){
          if(event.getToolType(0) == MotionEvent.TOOL_TYPE_STYLUS && isErasing){
               float pressure = event.getPressure();
               currentWidth = 25f * pressure; //Adjust size based on pressure
               paint.setStrokeWidth(currentWidth);
          } else {
                paint.setStrokeWidth(isErasing ? 25f : 10f); //Default width if not stylus or not erasing

          }
     }

    public void setErasing(boolean erasing){
        isErasing = erasing;
           if(!isErasing){
               paint.setStrokeWidth(10f);
           }
    }
    public boolean isErasing() {
       return isErasing;
    }
     public void clear(){
          bitmapCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
         invalidate();
         path.reset();
    }

}
```

This last snippet adds a simple `updateStrokeWidth()` method to check for stylus input and pressure, and adjust the stroke width based on that pressure.  This example demonstrates how pressure from a stylus can dynamically influence the size of the eraser for better precision. It's good to have a default brush size when using touch or when the user is not erasing.

For further study, I'd strongly suggest reading up on Porter-Duff compositing modes from the official Android documentation (search for `android.graphics.PorterDuff.Mode` within Android developer docs) and the graphics pipeline in android; Understanding how `Canvas`, `Bitmap`, and `Paint` interact is key here. Also, a great resource is "Filthy Rich Clients" by Chet Haase and Romain Guy; while a bit older now, its chapters on Android graphics remain very relevant. There's also plenty of valuable information to be gained from exploring the official android developer samples on github, as well. These resources will provide a more foundational understanding of graphics and performance considerations. Remember to always test thoroughly, especially on varied devices, since performance and subtle differences in rendering can appear in different android versions. This should provide you a solid base to build from when creating your eraser tool.
