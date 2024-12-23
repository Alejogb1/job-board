---
title: "how to start an app with activity1 or activity2?"
date: "2024-12-13"
id: "how-to-start-an-app-with-activity1-or-activity2"
---

 so you need to figure out how to launch your app sometimes with Activity1 and sometimes with Activity2 right got it I've been there man so many times it feels like every project has this weird edge case I think I even got nightmares about intent flags for a month once anyways lets unpack this

First thing is you can’t use the manifest default activity for this at least not in the straightforward sense because it specifies a single entry point you want flexibility a choice at runtime you gotta ditch the manifest's fixed idea and become the master of your app's entry point

Basically you need a decision point something that decides when to go to Activity1 or Activity2 I usually use what I call a "Launcher Activity" this isn't an official android thing just a convention I use all the time. It's an empty activity that figures out where you need to go and then pushes you there. Think of it like a bouncer at a club but instead of checking IDs it checks some flags or settings for your app

Now the actual implementation is simple enough. I'll show you a basic example with a shared preference its not the best way for complex apps but its great to illustrate the point and it helps to make some tests and quickly understand the core concept.

First this is what your LauncherActivity looks like :

```java
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

public class LauncherActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        SharedPreferences prefs = getSharedPreferences("MyPrefs", MODE_PRIVATE);
        boolean useActivity2 = prefs.getBoolean("useActivity2", false);


        Intent intent;
        if (useActivity2) {
            intent = new Intent(this, Activity2.class);
        } else {
            intent = new Intent(this, Activity1.class);
        }
        startActivity(intent);
        finish();
    }
}
```

so what's happening here is this we read from `SharedPreferences` a boolean flag called `useActivity2` if its true we start Activity2 if its false we start Activity1 and finish our LauncherActivity so it's not hanging around in the back stack you need to finish here always or you will end up with multiples LauncherActivity in the back stack and that is a recipe for disaster trust me.

Now you need to remember to set this LauncherActivity as your main activity in the manifest the one with the `LAUNCHER` intent filter so something like this:

```xml
<activity
    android:name=".LauncherActivity"
    android:exported="true">
    <intent-filter>
        <action android:name="android.intent.action.MAIN" />
        <category android:name="android.intent.category.LAUNCHER" />
    </intent-filter>
</activity>

 <activity
    android:name=".Activity1"
    android:exported="false"/>

 <activity
    android:name=".Activity2"
    android:exported="false"/>

```

So you can see here we have our LauncherActivity set to launch on startup which is what we want but Activities 1 and 2 are not entry points no `LAUNCHER` or `MAIN` activity filter on them.

Now you need to decide when to set that boolean in my experience it varies wildly you can do it on the first run of the app maybe based on some user selection in the settings or some internal app logic or even from a push notification a lot of flexibility here so its up to your app design.

To show how you set the preference here is an example you might do this somewhere else in your code like the first time you are opening the app after install for example you might do this with some boolean that you read on the first time the app is opened or if the user clicks on settings to change that behavior or if you got a push message:

```java
    SharedPreferences prefs = getSharedPreferences("MyPrefs", MODE_PRIVATE);
    SharedPreferences.Editor editor = prefs.edit();
    editor.putBoolean("useActivity2", true);
    editor.apply(); // Use apply for async commits

```

This will set the flag to true for subsequent launches when you start the app it will go directly to `Activity2` instead of `Activity1`.
Remember to use `apply` for async updates instead of `commit` if you are not doing other operations that need to have the result synchronously you can check the differences of these two in documentation online but usually apply is enough.

There is also another approach that i have seen sometimes but is less elegant in my opinion since it mixes the logic of starting the app directly in the activity itself. In this case you would still have the launcher activity be the activity1 and in the `onCreate` of the Activity1 you would implement the decision logic to check if you should start Activity2 or not. For this you would have activity1 doing something like this

```java
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

public class Activity1 extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        SharedPreferences prefs = getSharedPreferences("MyPrefs", MODE_PRIVATE);
        boolean useActivity2 = prefs.getBoolean("useActivity2", false);


        if (useActivity2) {
            Intent intent = new Intent(this, Activity2.class);
            startActivity(intent);
            finish();
        } else {
            // Continue with Activity1's normal logic
            setContentView(R.layout.activity_activity1); // You would have this here to show activity1 content
        }
    }
}
```

This is not my favorite because now activity 1 is doing something that it is not really its responsability it is acting like a launcher now and that is bad for separation of concerns but it might be useful if you only need some simple logic.

Now I know what you are thinking is this a lot for just picking which activity to start it's not the most elegant solution but it works and that's what matters it gives you complete control over the starting point of your app. Plus you will most likely need this flexibility in a lot of apps and in my experience it's good to have this type of pattern always in the back of your head.

I've spent countless hours trying to do this with just intent flags and it's never worked as reliably as this more explicit approach. There was one time I actually broke the manifest parser or thought I had and started panicking for 3 days it was a real nightmare I am pretty sure it was just my lack of sleep though lol. I remember once a colleague said something funny about me trying to solve this issue something about Intents being not my forte or something I don't know I can't remember it that well.

Anyway for further reading you might want to explore the android documentation on intents and activity lifecycle it really helps in understanding what happens behind the scenes when you do `startActivity` for example the android dev documentation is quite useful for this also for more advanced uses check out the android source code for the `ActivityManager` class for some extra details it is really a rabbit hole though I really dont recommend it unless you have a lot of spare time that you dont know what to do with. Also the book "Android Programming The Big Nerd Ranch Guide" really helped me in some cases with this and other issues it has a very practical approach that I think you will like.

Remember always start with the simple solution and go up in complexity don't try to over engineer something before you have it working correctly that is my best advice for this issue also use tests that is very very important you should always test your code before deploying it especially in this case where you are deciding which of the activities to start.

And that's it let me know if you need anything else I've been down this road so many times it's almost like I have a map of the place.
