---
title: "How to create an erasing tool in an Android drawing application?"
date: "2024-12-16"
id: "how-to-create-an-erasing-tool-in-an-android-drawing-application"
---

Let's tackle the concept of creating an erasing tool in an Android drawing application. It’s a problem I’ve certainly grappled with in the past, specifically during the development of a collaborative sketching app for tablet users, a project that turned out to be more intricate than initially anticipated, and it was all because of how we handled the erase functionality. So, from my experience, there isn’t just one single "best" way, but a few reliable methods, each with its own set of tradeoffs and nuances.

At the heart of it, when we're talking about erasing in a digital drawing app, we're essentially manipulating the visual representation of the artwork. This usually involves either overwriting the existing pixels or simulating an erase effect using a clever mix of blend modes and drawing techniques. Let’s break this down.

The first, and perhaps most straightforward approach, is to use the canvas's drawing commands. Essentially, we're not really "erasing"; we're drawing over existing content with the background color. Think of it like painting over a mistake with white paint on a traditional canvas. This method relies on setting the paint's color to match the background color and then drawing a path (or a series of connected paths) over the area intended for 'erasing.' It’s efficient for simple use cases, but when you need finer control or want to erase more complex elements, this approach might feel crude.

Here’s a basic code example illustrating this concept:

```java
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.MotionEvent;
import android.view.View;

public class EraserCanvasView extends View {

    private Paint eraserPaint;
    private Path eraserPath;
    private float lastX, lastY;

    public EraserCanvasView(Context context) {
        super(context);
        init();
    }

    private void init() {
        eraserPaint = new Paint();
        eraserPaint.setColor(android.graphics.Color.WHITE); // Set to background color
        eraserPaint.setStyle(Paint.Style.STROKE);
        eraserPaint.setStrokeWidth(20f); // Adjust eraser size
        eraserPaint.setAntiAlias(true);
        eraserPaint.setStrokeJoin(Paint.Join.ROUND);
        eraserPaint.setStrokeCap(Paint.Cap.ROUND);

        eraserPath = new Path();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawPath(eraserPath, eraserPaint);
    }


    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                eraserPath.moveTo(x, y);
                lastX = x;
                lastY = y;
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                float dx = Math.abs(x - lastX);
                float dy = Math.abs(y - lastY);
                if (dx >= 4 || dy >= 4) {  // Adjust this threshold for smoother paths
                     eraserPath.quadTo(lastX, lastY, (x + lastX)/2, (y+lastY)/2);
                     lastX = x;
                     lastY = y;
                     invalidate();
                 }
                 break;

            case MotionEvent.ACTION_UP:
                 eraserPath.lineTo(x,y);
                 invalidate();
                 break;
             default:
                 return false;
        }
        return true;
    }

    public void setEraserColor(int color){
       eraserPaint.setColor(color);
    }
}
```

In this first example, the `EraserCanvasView` creates a basic eraser using the background color. It reacts to touch events, creating a path that mimics a finger drawing with the selected 'erase' color. Setting `eraserPaint.setColor()` allows this to be modified in the code, allowing a variety of different colors to be used.

The second approach involves utilizing `PorterDuff` blend modes, particularly `PorterDuff.Mode.CLEAR`. This is a very powerful method and provides a more genuine 'erasing' effect, creating transparent areas on the canvas, instead of using the background color. Instead of overwriting, it makes pixels transparent, revealing anything below the current layer. This approach is extremely handy when you have multiple layers or want the eraser to truly remove the underlying content.

Here’s a code snippet demonstrating the use of `PorterDuff.Mode.CLEAR`:

```java
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.view.MotionEvent;
import android.view.View;

public class ClearEraserCanvasView extends View {

     private Paint eraserPaint;
     private Path eraserPath;
    private float lastX, lastY;

    public ClearEraserCanvasView(Context context) {
        super(context);
         init();
    }

    private void init() {
         eraserPaint = new Paint();
         eraserPaint.setAntiAlias(true);
         eraserPaint.setStyle(Paint.Style.STROKE);
         eraserPaint.setStrokeJoin(Paint.Join.ROUND);
        eraserPaint.setStrokeCap(Paint.Cap.ROUND);
         eraserPaint.setStrokeWidth(20f);  // Adjust eraser size
         eraserPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR)); // Enable clear mode
        eraserPath = new Path();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawPath(eraserPath, eraserPaint);
    }


    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                eraserPath.moveTo(x, y);
                lastX = x;
                lastY = y;
                invalidate();
                 return true;
            case MotionEvent.ACTION_MOVE:
                 float dx = Math.abs(x - lastX);
                 float dy = Math.abs(y - lastY);
                 if (dx >= 4 || dy >= 4) {  // Adjust this threshold for smoother paths
                      eraserPath.quadTo(lastX, lastY, (x + lastX)/2, (y+lastY)/2);
                      lastX = x;
                      lastY = y;
                      invalidate();
                 }
                break;
            case MotionEvent.ACTION_UP:
                  eraserPath.lineTo(x,y);
                  invalidate();
                  break;
             default:
                 return false;
        }
        return true;
    }
}

```

Here the `ClearEraserCanvasView` utilizes `PorterDuff.Mode.CLEAR`. This creates actual transparency where the user 'erases,' showing whatever is below the current drawing. This is more powerful than simply drawing the background color, as it completely removes the underlying pixel data.

Finally, a third method, which I found particularly useful in the collaborative app I mentioned, involves working with bitmaps. Instead of drawing directly onto the canvas, the drawing operations are initially performed onto a bitmap and then the bitmap itself is drawn onto the canvas. When it's time to erase, we don't directly modify the displayed canvas. Instead, we draw the "erase" path using `PorterDuff.Mode.CLEAR` directly onto the underlying bitmap. This offers more control and flexibility, specifically with undo/redo features and dealing with different layers.

Here's a simplified code example:

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

public class BitmapEraserCanvasView extends View {

    private Bitmap drawBitmap;
    private Canvas drawCanvas;
    private Paint drawPaint;
     private Paint eraserPaint;
     private Path eraserPath;
    private float lastX, lastY;


    public BitmapEraserCanvasView(Context context) {
        super(context);
        init();
    }


    private void init() {
        drawPaint = new Paint();
        drawPaint.setColor(android.graphics.Color.BLACK);
        drawPaint.setStyle(Paint.Style.STROKE);
        drawPaint.setStrokeWidth(5f);
        drawPaint.setAntiAlias(true);
         drawPaint.setStrokeJoin(Paint.Join.ROUND);
        drawPaint.setStrokeCap(Paint.Cap.ROUND);

         eraserPaint = new Paint();
         eraserPaint.setAntiAlias(true);
         eraserPaint.setStyle(Paint.Style.STROKE);
         eraserPaint.setStrokeJoin(Paint.Join.ROUND);
        eraserPaint.setStrokeCap(Paint.Cap.ROUND);
         eraserPaint.setStrokeWidth(20f);
        eraserPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
         eraserPath = new Path();
    }



    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        drawBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        drawCanvas = new Canvas(drawBitmap);
        drawCanvas.drawColor(android.graphics.Color.WHITE);  // Initial background
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawBitmap(drawBitmap, 0, 0, null);
        canvas.drawPath(eraserPath, eraserPaint);
    }


   @Override
   public boolean onTouchEvent(MotionEvent event) {
       float x = event.getX();
        float y = event.getY();

       switch (event.getAction()) {
          case MotionEvent.ACTION_DOWN:
               eraserPath.moveTo(x, y);
                lastX = x;
                lastY = y;
               invalidate();
                return true;
           case MotionEvent.ACTION_MOVE:
              float dx = Math.abs(x - lastX);
                float dy = Math.abs(y - lastY);
                if (dx >= 4 || dy >= 4) {
                    eraserPath.quadTo(lastX, lastY, (x + lastX)/2, (y+lastY)/2);
                    lastX = x;
                    lastY = y;
                   drawCanvas.drawPath(eraserPath, eraserPaint);  // Draw erase on bitmap
                    eraserPath.reset();
                   invalidate();
                }
               break;
            case MotionEvent.ACTION_UP:
                 eraserPath.lineTo(x,y);
                drawCanvas.drawPath(eraserPath, eraserPaint); // Draw final erase path on bitmap
                 eraserPath.reset();
                 invalidate();
                 break;
           default:
               return false;
      }
        return true;
    }
    public void drawSomeLine(float x1, float y1, float x2, float y2){
        drawCanvas.drawLine(x1, y1, x2, y2, drawPaint);
        invalidate();
    }

}

```

In this `BitmapEraserCanvasView`, we create the `drawBitmap` and `drawCanvas` when the view is created. Draw operations and eraser operations are performed to the bitmap, which is then rendered on the view’s canvas. This separation makes it significantly easier to manage different drawing states and provides a far superior method of managing the erase functionality. A simple method to test the draw functionality is provided in the `drawSomeLine()` function.

To further enhance your understanding of these concepts, I recommend delving into publications like "Android Graphics and Animation" by Jerome DiMarzio for a strong foundation in graphics concepts on Android. Additionally, papers that discuss techniques for real-time canvas manipulation, especially those focusing on `PorterDuff` modes, will prove invaluable. Look for resources that cover concepts like double-buffering, which enhances the performance of these drawing methods. Lastly, a deep dive into `Android.graphics` package documentation will be vital.

In conclusion, creating an effective erase tool in an Android drawing application involves understanding canvas operations, blend modes, and bitmap manipulation. The best approach will often depend on the specific requirements of the project, including complexity of the artwork, performance targets, and desired user experience. What I’ve found is, regardless of which method you use, the key is to understand the core principles behind how Android renders graphics, and how we can leverage this to deliver an engaging user experience.
