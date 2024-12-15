---
title: "How to play a video in fullscreen in landscape with ExoPlayer?"
date: "2024-12-15"
id: "how-to-play-a-video-in-fullscreen-in-landscape-with-exoplayer"
---

alright, so fullscreen landscape with exoplayer, yeah i’ve been down that road a few times. it always seems like it should be a one-liner, but it usually isn’t. i remember my first real encounter with this, back in the pre-android jetpack days, it was a mess of manual layout calculations and handling rotation myself. i’m talking about pre-2015, using the old support library. we had to juggle ui flags, view groups, and calculate dimensions by hand it felt like i was performing a pixel dance every time the orientation changed. let's just say, it made me appreciate the api updates that came later.

anyway, the basic idea involves a few core steps. firstly, you need to toggle the system ui flags, basically telling android to go fullscreen and get rid of the navigation and status bars. then, your player view has to adapt to the fullscreen dimensions, which is usually the parent view's dimensions. finally, rotation to landscape needs to be done programmatically.

here’s how i usually tackle it. i'll start by assuming you have your exoplayer instance and player view all set up. if you don’t, i recommend checking out the exoplayer documentation, it's pretty solid. also, the google codelabs for exoplayer are gold, if you're just getting started i can't recommend them enough.

so, first, the fullscreen toggle method:

```java
    private void toggleFullscreen(boolean fullscreen) {
        if (fullscreen) {
           // set flags for immersive fullscreen
            getWindow().getDecorView().setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            );
            // request landscape orientation
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        } else {
            // restore ui flags
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_VISIBLE);

            // return to portrait mode, if that's what you need
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        }
    }

```

this little function does the heavy lifting. it takes a boolean which decides whether to go fullscreen or restore the ui. the `setSystemUiVisibility` calls, they're key, if you don't use those android will keep showing things that you don't want to see in fullscreen mode like the status bar and navigation buttons. the `ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE` forces the activity into landscape and the other to portrait again.

now, the trick here is to know when to call it. usually, i would bind this to a button click, something like that. or if you have a double tap or gesture you can use that too. lets imagine we have a button id called `fullscreen_button`:

```java
       Button fullscreenButton = findViewById(R.id.fullscreen_button);
        fullscreenButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // toggle the fullscreen state, you need to have a boolean declared on your activity class
                isFullscreen = !isFullscreen;
                toggleFullscreen(isFullscreen);
                updateLayoutParams(); //important call
            }
        });
```

now, as you can see, there is an important `updateLayoutParams()` call. that’s because after making the system go full screen, we must change the player view’s size to match the full size of the display and here how you make the `updateLayoutParams()` function look like:

```java
   private void updateLayoutParams(){
        PlayerView playerView = findViewById(R.id.player_view);

        // gets the parent view
        ViewGroup.LayoutParams params = playerView.getLayoutParams();

        if(isFullscreen){
            params.width = ViewGroup.LayoutParams.MATCH_PARENT;
            params.height = ViewGroup.LayoutParams.MATCH_PARENT;
        }
        else{ // we restore to the default dimensions, probably in this situation we have some aspect ratio in mind so we get the layout from a attribute set in the xml layout for example
             params.width = ViewGroup.LayoutParams.WRAP_CONTENT;
             params.height= ViewGroup.LayoutParams.WRAP_CONTENT;
        }

        playerView.setLayoutParams(params); //apply the layout
    }
```

this function retrieves the player view which id is `player_view` then updates it’s layout parameters to be fill parent in both dimensions if we’re in fullscreen or reset it to wrap content if not. the view dimensions should fit the parent view.

a common pitfall i've seen is not properly handling configuration changes, especially when you rotate the device. the default behavior of an android activity is to restart when it's orientation changes so you can handle the changes yourself or avoid them all together to avoid a flicker by doing the following:

```xml
    <activity
            android:name=".YourActivity"
            android:configChanges="orientation|screenSize|screenLayout|keyboardHidden"
             ...>
    </activity>
```

adding this to your `AndroidManifest.xml` tells android that you will handle the configuration changes yourself in code, avoiding the activity recreation. remember when you call the toggle method you are already changing the orientation programatically so the config change is already going to happen there so you dont need to care about handling it twice.

also keep in mind that if you're using any custom ui over your player view, you may also need to update it's layout params as well. and sometimes, weird behaviors can arise with different phones or android versions so it's better to test in several devices. i recall once debugging an issue for hours just to discover that a specific device was not correctly setting the immersive flag and that i needed to change to another one. it was really irritating. so my advice is, if possible, try your implementation in multiple real devices not just emulators.

if you want to go deeper, the android developer documentation on system ui visibility is a good resource. and if you want to understand the ui lifecycle better, books like “android programming: the big nerd ranch guide” are pretty much the bible of android development and worth checking out.

and yeah that's pretty much it for full-screen landscape exoplayer playing, i hope this helps. and i also hope that this time, everything runs smoothly without any unexpected bugs from another dimension. sometimes android feels like a magical box that is also unpredictable.
