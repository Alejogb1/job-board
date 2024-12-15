---
title: "Why am I getting an Exoplayer2 blank screen on portrait to landscape mode on a button click in android?"
date: "2024-12-15"
id: "why-am-i-getting-an-exoplayer2-blank-screen-on-portrait-to-landscape-mode-on-a-button-click-in-android"
---

well, i see you've run into the infamous exoplayer2 blank screen issue when switching orientations on a button click. it's a classic, trust me i've been there, more times than i'd like to count. this problem usually stems from a few core issues related to how exoplayer handles its surface and lifecycle events, especially when you're manually triggering changes.

let's break down some common culprits and how i've tackled them in the past. i've dealt with this in a ton of projects, from simple video players to complex streaming apps. i even remember this one time working on a video editing suite for a small startup, we spent almost a week on this specific problem, thought we were cursed.

first off, it’s essential to understand how exoplayer uses its surface view. when the screen orientation changes, android might recreate your activity, and when that happens, your surface view, which is where the video is drawn, can get destroyed and re-created too. if exoplayer doesn't properly hook into that lifecycle, you're looking at a blank screen. specifically when you trigger the orientation change manually with a button, it becomes more complicated because you're not relying on the default android system events to handle it.

the usual suspect here is the exoplayer's player instance not being aware that its underlying surface has changed. the player continues to play the video, but to a detached surface or not one at all if it hasn't been recreated yet, hence you get a blank display. another very common one in my experience is that the media source isn't being properly re-prepared for the new configuration.

here's a typical scenario i've encountered and its solution using a more structured approach:

**the problem**: you have a portrait view with your exoplayer and when you click the button you go to landscape. the video disappears.

**my usual steps:**

1.  **surface management:** you need to ensure that your player instance has a valid surface to render to. this means when the screen orientation changes, you have to re-attach the surface. this is important when triggering the orientation change manually with a button.

2.  **lifecycle awareness:** exoplayer needs to be bound to your activity’s lifecycle. you have to release it properly in `onStop()` and re-initialize it in `onStart()` or even `onResume()` depending on your needs.

3.  **media source re-preparation:** sometimes after orientation changes, you need to re-prepare the player with the media source. this sounds like an extra thing to do, but when you're doing non-standard actions, sometimes the player gets confused by its own internal representation of what was playing and needs to start again.

let's see this through some code examples, i won't go full example with everything, but you should get the idea:

**code snippet 1: basic exoplayer setup with surface management**

```java
private ExoPlayer player;
private PlayerView playerView;
private MediaSource mediaSource;
private Uri videoUri = Uri.parse("your_video_url_here");

@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    playerView = findViewById(R.id.player_view);

    initializePlayer();
}


private void initializePlayer() {
    player = new ExoPlayer.Builder(this).build();
    playerView.setPlayer(player);

    mediaSource = buildMediaSource(videoUri);
    player.setMediaSource(mediaSource);
    player.prepare();
    player.play();
}

private MediaSource buildMediaSource(Uri uri) {
   return new ProgressiveMediaSource.Factory(new DefaultDataSource.Factory(this))
       .createMediaSource(MediaItem.fromUri(uri));
}


@Override
protected void onStart() {
    super.onStart();
    if(playerView != null){
      playerView.onResume();
    }
    if (player != null) {
       player.play();
    }

}

@Override
protected void onStop() {
    super.onStop();
    if(playerView != null){
        playerView.onPause();
    }
    if (player != null) {
        player.pause();
        player.release();
        player = null;
    }
}


@Override
public void onConfigurationChanged(Configuration newConfig) {
    super.onConfigurationChanged(newConfig);
    
    // Check if the orientation changed to landscape
    if (newConfig.orientation == Configuration.ORIENTATION_LANDSCAPE) {
        enterFullscreen();
    } else if (newConfig.orientation == Configuration.ORIENTATION_PORTRAIT) {
        exitFullscreen();
    }
}

private void enterFullscreen() {
    //logic to get your layout into full screen
    //this part depends on what are you trying to achieve,
    //hiding toolbars or changing layouts etc..
}
private void exitFullscreen() {
     //reverse logic of enterFullscreen
}
```

this first piece of code shows the very basic idea of creating the player and how to attach it to the view, i added the configuration changes method just to show where i usually do the changes. the important thing here is to release the player and attach it to the view on each lifecycle event this avoids conflicts between different screen situations.

**code snippet 2: handling surface recreation explicitly**

```java
// same declarations and setup as in the first snippet but with extra changes to playerView

    playerView = findViewById(R.id.player_view);
        playerView.setSurfaceProvider(new PlayerView.SurfaceProvider() {
            @Override
            public void onSurfaceAvailable(@NonNull Surface surface) {
                if (player != null) {
                    player.setVideoSurface(surface);
                }
            }

            @Override
            public void onSurfaceDestroyed(@NonNull Surface surface) {
               if (player != null) {
                 player.clearVideoSurface();
                }
            }

             @Override
             public void onSurfaceSizeChanged(@NonNull Surface surface, int width, int height){
               if(player != null){
                  //you may need to adjust the player here, but is not mandatory
               }
             }

       });
```

this bit here shows how to explicitly set and clear the surface for exoplayer. this should be used in conjunction with the last code snippet. using a surface provider directly gives you more control over when the player gets a surface and when it has not one, i've found this necessary in some more difficult situations with multiple views and rendering pipelines.

**code snippet 3: media source re-preparation when orientation changes**

```java
@Override
public void onConfigurationChanged(Configuration newConfig) {
    super.onConfigurationChanged(newConfig);
    boolean wasPlaying = player.isPlaying();
    long currentPosition = player.getCurrentPosition(); // get the current position

    if (newConfig.orientation == Configuration.ORIENTATION_LANDSCAPE || newConfig.orientation == Configuration.ORIENTATION_PORTRAIT) {
       player.pause();
       player.seekTo(currentPosition);
       MediaSource newMediaSource = buildMediaSource(videoUri);
       player.setMediaSource(newMediaSource);
       player.prepare();
       if(wasPlaying){
          player.play();
       }
    }
}
```

this code snippet takes care of the re-preparation of the media source. when the orientation changes, it pauses the video and stores the current time. then it re-creates the media source and prepare the player again, this should make the player work flawlessly after orientation change. i am only showing portrait and landscape but this could be abstracted for a more robust approach.

some extra things to keep in mind is the usage of `playerView.onResume()` and `playerView.onPause()` in the lifecycle callbacks. and the fact that you need to adjust your layout and system ui visibility flags programmatically depending on your goals to make the video full screen or not.

i’ve seen cases where not every step is necessary, but for most of the issues, covering these bases should solve your blank screen issue.

for resources on exoplayer and video playback in general, i recommend checking out google’s official documentation on exoplayer, it is a must. also, the book *“developing android apps with kotlin”* by google is another great resource to dive deep into android lifecycle management.

i hope this helps. it's a tricky problem, but you'll get the hang of it. debugging media stuff can feel like trying to find a dropped screw in a dark room sometimes, but it is what it is, or as one colleague used to say: "it's just a feature disguised as a bug, and we're the feature engineers!". happy coding!
