---
title: "How to change the stroke colour programmatically in Android?"
date: "2024-12-15"
id: "how-to-change-the-stroke-colour-programmatically-in-android"
---

alright, so you're looking at programmatically changing the stroke color of something in android, eh? i've been there, done that, got the t-shirt, and probably a few stress-induced grey hairs to show for it. it sounds like a pretty straightforward thing but trust me it can lead to some very confusing moments when you first try it. let me walk you through how i've done it and some of the gotchas i've seen.

basically, you're working with either a drawable, usually something like a shape drawable (a `gradientdrawable` or `shapeshape`) or maybe a vector drawable. the method changes a bit depending on what you have.

let’s start with the case that you have a shape drawable that has an xml definition in your res folder. i've seen the most cases with those. now, if you want to change that stroke color dynamically you can not change the drawable itself that lives in the resource because this drawable object is shared across all instances of your app. thus, changes to it would change for all elements that are using this specific resource. you have to create a copy and then do the changes. the general idea is to grab the drawable from resources, create a mutable version of it, and then use `setstrokecolor`.

so let's say you have a drawable in `res/drawable/my_shape.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<shape xmlns:android="http://schemas.android.com/apk/res/android"
    android:shape="rectangle">
    <solid android:color="#FFFFFF" />
    <stroke
        android:width="2dp"
        android:color="#000000" />
    <corners android:radius="8dp" />
</shape>
```

to change that black stroke color to something else at runtime you would do this in your activity or fragment:

```java
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;

public class MyActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ImageView myImageView = findViewById(R.id.my_image_view); //assuming you have an imageview

        Drawable originalDrawable = getResources().getDrawable(R.drawable.my_shape, getTheme());

        if(originalDrawable instanceof GradientDrawable){ //check if it is a gradient drawable

            GradientDrawable shapeDrawable = (GradientDrawable) originalDrawable.mutate();
             shapeDrawable.setStroke(4, Color.RED); //red color example 

            myImageView.setImageDrawable(shapeDrawable);

        }
    }
}
```

a few things here. first, `getresources().getdrawable()` gets your drawable. i've used `gettheme()` here to make it work correctly in all versions, especially those below android 6.0 (api 23). then, the important part: `.mutate()` creates a mutable copy of the drawable. without this, any changes you make will affect all instances of that drawable throughout your app. it’s a classic error that can cause many headaches especially if you are using the same drawable in multiple places and then suddenly all of them change colors. i've spend hours with that one. the other important part is the check with `instanceof GradientDrawable`. this is a sanity check and to ensure that the type of drawable we are getting back from the resources is the same as the type we expect. i made the assumption here that you are using a shape drawable, but this check can help you prevent crashes in runtime.

then, `setstroke` is the method that does the actual changing of the color. i've used `color.red` here to get the red color. you can set it to whatever you need to and specify also the stroke width. in this specific case i am changing it to a 4dp width. finally, `setimagedrawable()` sets that modified drawable to the image view.

now, if you're dealing with a vector drawable, it's a little different. vector drawables are defined in xml using paths and colors, so you need to access the paths and change their paint's color.

let's imagine you have a `my_vector.xml` like this:

```xml
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="24dp"
    android:height="24dp"
    android:viewportWidth="24"
    android:viewportHeight="24">
  <path
      android:pathData="M12,2 L12,22 M2,12 L22,12"
      android:strokeColor="#FF000000"
      android:strokeWidth="2"/>
</vector>
```

this simple vector is a cross. to change the stroke color to say blue you would do it like this:

```java
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.VectorDrawable;
import android.os.Bundle;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.drawable.DrawableCompat;

public class MyActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    
        ImageView myImageView = findViewById(R.id.my_image_view);
    
         Drawable originalDrawable = getResources().getDrawable(R.drawable.my_vector, getTheme());

        if (originalDrawable instanceof VectorDrawable) {
            VectorDrawable vectorDrawable = (VectorDrawable) originalDrawable.mutate();
            
            //this is the important part, we are grabbing the first path
            DrawableCompat.setTint(vectorDrawable, Color.BLUE);
            myImageView.setImageDrawable(vectorDrawable);
        }
    }
}
```

here we are using `drawablecompat.settint()` which is the easiest way of setting the tint to the whole vector drawable. this can be good enough for simple cases, but if you want more granular control over it you would need to iterate through the paths and subpaths and change their respective paint parameters. if you want a full example of how to do that just let me know i have code for it somewhere. i've had to deal with very complex vector drawables before and it is not trivial to understand each parameter at first. the `.mutate()` is again needed.

another common case is when you are creating a drawable directly via the code, like this example:

```java
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;

public class MyActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    
        ImageView myImageView = findViewById(R.id.my_image_view);
    
        GradientDrawable shapeDrawable = new GradientDrawable();
        shapeDrawable.setShape(GradientDrawable.RECTANGLE);
        shapeDrawable.setColor(Color.WHITE); // Set the solid color if necessary
        shapeDrawable.setStroke(4, Color.GREEN); // Setting the stroke here directly

        myImageView.setImageDrawable(shapeDrawable);
    }
}
```

this creates a gradientdrawable object directly in code. i like this method because it is simple and you do not need to create xml files. in this case setting the stroke color is as simple as using the `setStroke()` method. no need for `mutate()` here since you are creating it directly. this is generally what i try to do now as it is cleaner and the code is more readable.

for further information i would suggest looking into the official android documentation about drawables. also, the "android graphics architecture" paper by romain guy from 2011 is a great read, although quite old. it does give the fundamentals of how android drawables work. it is not very easy to find now since it is very old but it still contains fundamental knowledge about how this all works behind the scenes. if you want to go into a very deep dive the book "android graphics with opengl es and ndk" will be useful. i read it a long time ago and it does explain how the android graphics subsystem works. that can be useful if you are having very low level performance issues with drawables in your apps.

oh, and one more thing: when changing colors dynamically remember that sometimes, you might be running into situations where you want to change that color for accessibility purposes, such as inverting colors, if you need to do this then make sure to test for that edge case. i've spend hours with that one too. a good test is to just enable the "color inversion" or "color correction" from the android system and see if your app is behaving correctly. its a kind of testing that very few developers ever do. also, a word of warning here if you are planning to make animations with drawables: animated vector drawables can be a pain sometimes. its just something to take into consideration. it reminds me of the time i tried to debug a rendering issue on a game on android, turned out the solution was that i had a typo on a variable name that was only causing problems on android because of how opengl handles its contexts. sometimes i feel like my job is to be a professional code spelunker trying to find all the obscure bugs in the deep depths of code... but in a good way. that's it, i hope this helps!
