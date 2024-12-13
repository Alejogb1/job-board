---
title: "orthographic camera react three fiber config?"
date: "2024-12-13"
id: "orthographic-camera-react-three-fiber-config"
---

Alright so orthographic camera react three fiber config huh I've been down that rabbit hole before like way more times than I care to admit it’s one of those things that sounds simple in theory but then you actually try to implement it and suddenly you're staring at a black screen or something wildly distorted

Let me tell you about the first time I dealt with this it was back in my early days using threejs I was building this architectural visualization app nothing fancy just a simple model viewer but the client insisted everything had to be perfectly aligned no perspective distortion at all naturally my noob self thought "oh easy just switch to orthographic camera" famous last words right

I spent a whole day banging my head against the wall trying to figure out why my model was either super zoomed in or completely out of view Turns out the configuration for an orthographic camera is a whole different ball game compared to its perspective sibling The projection parameters the frustum settings all that jazz its crucial to get them right

Okay so lets dive into some specifics using react-three-fiber makes it a little nicer but the underlying concepts are the same We're essentially manipulating a threejs camera object through the react declarative approach

First things first you need a camera component in your scene this is how a basic one looks

```jsx
import { OrthographicCamera } from '@react-three/drei'
import { useRef } from 'react'

function CameraComponent({ zoom = 1, position = [0, 0, 10] }) {
  const camera = useRef()

  return (
      <OrthographicCamera
        ref={camera}
        zoom={zoom}
        position={position}
        near={0.1}
        far={1000}
        left={-10}
        right={10}
        top={10}
        bottom={-10}
      />
  );
}

export default CameraComponent
```

Couple things to point out here `OrthographicCamera` is from `@react-three/drei` which is a useful library that gives us pre built components and utilities for threejs  `zoom` controls the level of magnification think of it as how much of the world fits into your view `position` is the cameras location in 3D space and `near` and `far` define the near and far clip planes everything within this range will be rendered `left` `right` `top` `bottom` these are the crucial bits for orthographic projection they define the viewport dimensions in camera space. If you get these wrong you'll see the issues I described earlier like your objects appearing stretched or outside the view frustum.

Now in my initial struggle i was just messing with zoom and position thinking i could make it work like a perspective camera it didn't work obviously i then started thinking maybe it was a problem with the models units nah it was the frustum parameters not being properly configured these `left` `right` `top` and `bottom` values need to correspond to the dimensions of your scene

So if your scene is say 20 units wide and 20 units tall you would need these values set to something like `-10, 10, 10, -10` or if it’s bigger `-50,50,50,-50` and so on The zoom level essentially scales the size of the virtual viewport if you zoom out the viewport looks like it gets wider

This was probably my mistake for my architectural visualization app I was thinking it was like a zoom in camera like my phone camera was working I didn't realized that is an actual "optical" zoom and that it was completely different

Now another thing to keep in mind is the aspect ratio In a perspective camera the aspect ratio is handled automatically using the cameras `fov` property and window dimensions but for the orthographic projection the aspect ratio is not automatically computed by the camera

Here is a common issue people get into when using a window resize and the aspect ratio for their camera

```jsx
import { useState, useEffect } from 'react';
import { useThree } from '@react-three/fiber';

function CameraComponent({ zoom = 1, position = [0, 0, 10] }) {
  const cameraRef = useRef();
  const { viewport } = useThree();
  const [aspect, setAspect] = useState(viewport.aspect);


  useEffect(() => {
    const handleResize = () => {
      setAspect(viewport.aspect);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [viewport.aspect]);


  const left = -10 * aspect;
  const right = 10 * aspect;
  const top = 10;
  const bottom = -10;

    return (
        <OrthographicCamera
          ref={cameraRef}
          zoom={zoom}
          position={position}
          near={0.1}
          far={1000}
          left={left}
          right={right}
          top={top}
          bottom={bottom}
        />
    );
}

export default CameraComponent;
```

What we do here we are using the `useThree` hook to get the current aspect ratio from react-three-fiber `viewport` and then we adjust the left and right properties of our camera accordingly This ensures that your objects don’t look stretched when the window resizes

If you were using a canvas component or a container that handles the resizing you might need to adjust it based on how you are getting the dimensions of your container

So far this might seem like an issue with configuration but this can also be an issue if we are trying to simulate a 2d camera with a "2d" world I remember I was working on this isometric game for a portfolio project i wanted the camera to look like old school pixel games no camera perspective at all i spent an entire weekend trying to calculate these bounds based on object sizes in the scene it was a wild goose chase

Here is a different case where you don't want to specify the frustum manually

```jsx
import { OrthographicCamera, useHelper } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import { useRef, useMemo } from 'react';
import * as THREE from 'three';

function CameraComponent({ zoom = 1, position = [0, 0, 10], bounds = [-10, -10, 10, 10], children }) {

  const cameraRef = useRef();
  const { size } = useThree();

  const frustumSize = useMemo(() => {
    const width = bounds[2] - bounds[0];
    const height = bounds[3] - bounds[1];

    const maxDimension = Math.max(width, height);
    const newFrustumSize = maxDimension / 2
    return newFrustumSize
  },[bounds])

  const adjustedBounds = useMemo(() => {
    const newBounds = [
      bounds[0] ,
      bounds[1],
      bounds[2],
      bounds[3],
    ];
      const width = newBounds[2] - newBounds[0];
      const height = newBounds[3] - newBounds[1];
      const aspectRatio = size.width / size.height;

    if(width > height){
      const diff = width - height;
      const diffByTwo = diff/2
        newBounds[1] -= diffByTwo
      newBounds[3] += diffByTwo
    }else if(height > width){
      const diff = height - width;
      const diffByTwo = diff/2
      newBounds[0] -= diffByTwo
      newBounds[2] += diffByTwo
    }
    return newBounds
  },[bounds, size])



  useHelper(cameraRef, THREE.CameraHelper, 1, 'red')

  return (
    <>
          <OrthographicCamera
            ref={cameraRef}
            zoom={zoom}
            position={position}
            near={0.1}
            far={1000}
            left={adjustedBounds[0]}
            right={adjustedBounds[2]}
            top={adjustedBounds[3]}
            bottom={adjustedBounds[1]}
          />
          {children}
    </>
  );
}

export default CameraComponent;
```

This component dynamically calculates the bounds based on the scene object by specifying the boundaries of the scene itself `bounds`  and it uses `useHelper` from drei to render a camera helper (useful for debugging). This version has two important `useMemo` hooks that calculates the bounds based on the given parameters and the current viewport size it scales the bounds of the camera to make sure that the entire scene is visible and makes sure the camera is never distorted. If the scene has a different width than height we scale the bounds to make sure it becomes a square this is usually what people do for isometric games

I mean if you think about it why did the orthographic camera cross the road? To get to the other side with no perspective of course

Anyways that's the gist of it configuring an orthographic camera in react-three-fiber isn't rocket science but it needs some attention to detail Especially frustum parameters and aspect ratios that are not automatically managed by react-three-fiber. There are a lot of ways you could do it but these are usually the main things to look at

For more in-depth stuff I would recommend looking at these

*   **"Mathematics for 3D Game Programming and Computer Graphics" by Eric Lengyel** this book dives deep into projection matrices which are the core of how cameras work
*   **"Real-Time Rendering" by Tomas Akenine-Möller, Eric Haines and Naty Hoffman** this is a bible for rendering a lot of information about camera models and projection and other rendering stuff is in this book
*   The **threejs documentation** itself is also a very good resource specifically for the camera objects you can also find very useful examples in their documentation

Hope that clears some things up feel free to ask if you have other questions and good luck with your scene.
