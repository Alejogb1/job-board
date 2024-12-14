---
title: "How to Test an Android application with a stylus?"
date: "2024-12-14"
id: "how-to-test-an-android-application-with-a-stylus"
---

alright, so you’re looking into testing your android app with a stylus, huh? i’ve been down that road, more times than i care to remember, and it’s definitely not as straightforward as just poking around with a finger. believe me. it has its own quirks, and getting it to play nice with your testing framework takes some wrangling. i'm not gonna pretend it's all sunshine and rainbows, but here's a breakdown of how i've tackled it in the past, along with some tips i've picked up along the way.

first off, the problem. fingers are large. styluses are pointy. android treats them, and their respective inputs, differently. your app might be perfectly responsive to a finger tap, but completely miss a stylus tap, or misinterpret its position, causing chaos. if your ui relies on precise input, like drawing apps or anything involving small interactive elements, then stylus testing is not a luxury, it’s a necessity. think of it as the difference between operating a crane with a sledgehammer, and using the designed control panel; accuracy matters.

now, from personal experience, i once worked on an annotation app. it was all smooth sailing during our initial user testing sessions, all performed with fingers. but the moment we handed the app out to users with styluses, things started to fall apart; lines were jagged, selections were imprecise, and some users had the pleasure of creating random strokes when they only meant to tap a button, that was a funny day, at least for me, since i wasn't the one doing the testing. it was a brutal reminder that assuming finger input behavior will match stylus input behavior is a recipe for disaster. that’s when i realized i had to change how i worked.

let’s get into specifics of how i approach testing an app with a stylus. we have to consider two main areas: unit testing and ui testing.

for unit tests, the core of it is mocking, if you can mock correctly, you are good to go, believe me, we are talking about simulated touch events, but when it comes to stylus, it adds more pressure details, pressure sensitivity and angles. that's where you will have to ensure your input logic handles those values with resilience. here’s a java-based example, using mockito:

```java
import org.junit.Test;
import static org.mockito.Mockito.*;
import android.view.MotionEvent;
import android.view.InputDevice;

public class StylusInputTest {

  @Test
  public void testStylusInput() {
    MotionEvent mockEvent = mock(MotionEvent.class);
    when(mockEvent.getToolType()).thenReturn(MotionEvent.TOOL_TYPE_STYLUS);
    when(mockEvent.getAction()).thenReturn(MotionEvent.ACTION_DOWN);
    when(mockEvent.getX()).thenReturn(100f);
    when(mockEvent.getY()).thenReturn(200f);
    when(mockEvent.getPressure()).thenReturn(0.8f);
    when(mockEvent.getAxisValue(MotionEvent.AXIS_TILT)).thenReturn(15f);
    when(mockEvent.getAxisValue(MotionEvent.AXIS_ORIENTATION)).thenReturn(45f);

    // now pass the mocked event to the method under test
    InputHandler handler = new InputHandler();
    handler.handleInput(mockEvent);

    // assert on the outcome
    // verify that the handler correctly processed the simulated stylus input
    // example:
    verify(mockEvent, times(1)).getPressure();
    verify(mockEvent, times(1)).getAxisValue(MotionEvent.AXIS_TILT);
  }
}


//dummy handler class
class InputHandler {
    public void handleInput(MotionEvent event) {
        float pressure = event.getPressure();
        float tilt = event.getAxisValue(MotionEvent.AXIS_TILT);
        float orientation = event.getAxisValue(MotionEvent.AXIS_ORIENTATION);
        //do something with values
    }
}

```

here, we create a mock event, set the tool type to stylus, set values for pressure, tilt, and orientation. we use mockito to verify that the right methods are called and verify the values. this approach has saved my skin countless times, allowing me to quickly pinpoint logic errors that would have otherwise slipped through.

however, unit tests are not enough. they don’t capture user experience. that’s where ui testing comes in, and things get more complicated because we are testing actual scenarios. for this you will need to use espresso or ui automator which allows you to programmatically send events to the app. here is a espresso example:

```java
import androidx.test.espresso.action.GeneralClickAction;
import androidx.test.espresso.action.Press;
import androidx.test.espresso.action.Tap;
import androidx.test.espresso.UiController;
import androidx.test.espresso.ViewAction;
import androidx.test.espresso.matcher.ViewMatchers;
import androidx.test.espresso.Espresso;
import androidx.test.ext.junit.rules.ActivityScenarioRule;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import android.view.MotionEvent;
import android.view.InputDevice;
import android.view.View;
import com.example.yourapp.MainActivity;

import static androidx.test.espresso.Espresso.onView;
import static androidx.test.espresso.matcher.ViewMatchers.withId;

@RunWith(AndroidJUnit4.class)
public class StylusUiTest {
    @Rule
    public ActivityScenarioRule<MainActivity> activityRule =
            new ActivityScenarioRule<>(MainActivity.class);

    @Test
    public void testStylusTap() {
        // we use the custom action.
       onView(withId(R.id.targetView)).perform(stylusTapAction(100, 200, 0.8f, 15f, 45f));
       // add your asserts here. check the screen or UI changes.
    }

   private static ViewAction stylusTapAction(final float x, final float y, final float pressure, final float tilt, final float orientation) {
        return new GeneralClickAction(
            Tap.SINGLE,
            action -> {
                MotionEvent event = MotionEvent.obtain(
                    System.currentTimeMillis(),
                    System.currentTimeMillis(),
                    MotionEvent.ACTION_DOWN,
                    x,
                    y,
                    pressure,
                    1,
                    0,
                    0,
                    0,
                    InputDevice.SOURCE_STYLUS,
                    0
                );
                event.setAxisValue(MotionEvent.AXIS_TILT, tilt);
                event.setAxisValue(MotionEvent.AXIS_ORIENTATION, orientation);
                 action.injectEvent(event);

                 event = MotionEvent.obtain(
                    System.currentTimeMillis(),
                    System.currentTimeMillis(),
                    MotionEvent.ACTION_UP,
                    x,
                    y,
                    pressure,
                    1,
                    0,
                    0,
                    0,
                    InputDevice.SOURCE_STYLUS,
                    0
                 );
                 event.setAxisValue(MotionEvent.AXIS_TILT, tilt);
                event.setAxisValue(MotionEvent.AXIS_ORIENTATION, orientation);
                 action.injectEvent(event);
            },
            Press.FINGER
        );
    }
}
```

in this example, we define a custom `stylusTapAction` that creates `motionEvent` objects mimicking stylus inputs with specified coordinates, pressure, tilt, and orientation and we use espresso `onView()` method with a matcher to find an `id`. this action is performed on that view. again, this approach is very robust once it is fully set up; at first it will look like an extra setup that adds more complexity but this is a necessary step to create a reliable test bed.

but sometimes, the best test is with the real deal: a real stylus and a real device. in this case, you need to implement a way to check for stylus input while using the application. one quick way to log all input parameters is by creating a log handler for stylus interactions as a debugging tool. here is an example, you can add this to your view's touch listener:

```java
import android.view.MotionEvent;
import android.view.View;
import android.util.Log;

public class StylusInputLogger implements View.OnTouchListener {
    private static final String TAG = "StylusInputLogger";
    @Override
    public boolean onTouch(View view, MotionEvent event) {
        if (event.getToolType() == MotionEvent.TOOL_TYPE_STYLUS) {
            Log.d(TAG, "Stylus Input Detected:");
            Log.d(TAG, "Action: " + event.getAction());
            Log.d(TAG, "X: " + event.getX());
            Log.d(TAG, "Y: " + event.getY());
            Log.d(TAG, "Pressure: " + event.getPressure());
            Log.d(TAG, "Tilt: " + event.getAxisValue(MotionEvent.AXIS_TILT));
             Log.d(TAG, "Orientation: " + event.getAxisValue(MotionEvent.AXIS_ORIENTATION));
            Log.d(TAG, "----------");
        }
        return false;
    }
}
```
this logger is very simple to include in a view. all you have to do is set the `onTouchListener` of your view, like this `myView.setOnTouchListener(new StylusInputLogger());`. and remember to set your `minSdkVersion` to at least api level 21. with this you can inspect the raw parameters that the device reports which is very handy when you are trying to troubleshoot. this kind of logger is like a personal debug inspector.

now, for the resources. i wouldn’t really recommend generic online tutorials for this particular problem. you have to get into deeper specifics, and that’s where academic papers and in-depth android books become your best friend.

*   "android programming: the big nerd ranch guide" is a very complete book that can be helpful for learning to interact with user inputs, it will also teach you some good practices of android development in general.
*   "the art of unit testing" this book might not be specific to android but it will give you very good fundamentals about how to structure your unit test. in particular, for mocking events.

and that's pretty much it, it’s a bit of an involved process, but focusing on unit tests, custom ui tests and a runtime log handler has always worked for me. never underestimate the value of real-device testing to gain an understanding of how the user interacts with the application. keep an eye on those pressure, tilt, and orientation values; they're where the real magic (or problems) lies when it comes to stylus interactions.
