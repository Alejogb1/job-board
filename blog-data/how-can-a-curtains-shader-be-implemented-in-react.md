---
title: "How can a curtains shader be implemented in React?"
date: "2024-12-23"
id: "how-can-a-curtains-shader-be-implemented-in-react"
---

Alright,  Implementing a curtains shader in React isn't as straightforward as, say, rendering a basic div, but it's absolutely achievable, and frankly, quite rewarding. I've seen this requirement pop up on several projects – once, it was for an e-commerce site aiming for a smooth, animated transition between product pages, and another time for a portfolio wanting a more engaging navigation experience. In both cases, a standard component transition just wouldn't cut it. The key here lies in combining React's component lifecycle with a shader library that can handle the heavy lifting of the fragment shader calculations, and in managing the animation loop effectively.

The fundamental concept is to use WebGL to draw our curtain effect. We won't be directly manipulating WebGL in React, which can get messy and cumbersome. Instead, we leverage a library – such as *three.js* or *pixi.js* – that abstracts away the low-level WebGL interactions. I lean towards *three.js* for this scenario because it provides a robust set of tools for handling 3d scenes, which our curtain effect will effectively be a part of.

The typical approach involves a few steps. First, we set up the *three.js* scene with a plane mesh that's mapped with textures (or a color if it's solid). Then, and this is crucial, we craft our custom fragment shader that takes input parameters like the progress of our curtain animation, and calculates the color output for each pixel based on that progress. Finally, we integrate this *three.js* scene within our React component and animate the shader uniform (the progress value) using React's component state and potentially requestAnimationFrame to handle frame updates.

Let’s get into some code. For our demonstration, I'll assume you have *three.js* installed (`npm install three` or `yarn add three`).

**Example 1: Basic Setup with Uniform Animation**

Here’s the initial setup, creating a basic plane with a simple shader that transitions from one color to another. This example sets the stage for the more complex implementations later on.

```javascript
import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

const CurtainShader = () => {
  const mountRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const shaderMaterialRef = useRef(null);

  useEffect(() => {
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    // Scene setup
    sceneRef.current = new THREE.Scene();
    cameraRef.current = new THREE.OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, 0.1, 1000);
    cameraRef.current.position.z = 10;
    rendererRef.current = new THREE.WebGLRenderer({ alpha: true });
    rendererRef.current.setSize(width, height);
    mountRef.current.appendChild(rendererRef.current.domElement);

    // Shader setup
    const vertexShader = `
      void main() {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const fragmentShader = `
      uniform float u_progress;
      void main() {
          vec3 color1 = vec3(0.0, 0.0, 1.0);
          vec3 color2 = vec3(1.0, 0.0, 0.0);
          gl_FragColor = vec4(mix(color1, color2, u_progress), 1.0);
      }
    `;

    shaderMaterialRef.current = new THREE.ShaderMaterial({
      vertexShader: vertexShader,
      fragmentShader: fragmentShader,
      uniforms: {
          u_progress: {value: 0}
      }
    });


    // Plane setup
    const planeGeometry = new THREE.PlaneGeometry(width, height);
    const planeMesh = new THREE.Mesh(planeGeometry, shaderMaterialRef.current);
    sceneRef.current.add(planeMesh);


    // Animation loop setup
    const animate = () => {
      if (shaderMaterialRef.current) {
        shaderMaterialRef.current.uniforms.u_progress.value = progress; // updates the shader's progress
      }
        rendererRef.current.render(sceneRef.current, cameraRef.current);
        requestAnimationFrame(animate);

    };
    animate();
      return () => {
          if(mountRef.current) {
            mountRef.current.removeChild(rendererRef.current.domElement);
          }
          rendererRef.current.dispose();
          shaderMaterialRef.current.dispose();
          planeGeometry.dispose();
        // any other cleanup like stopping animations can go here.
      };


    }, []); // Empty dependency array means this runs only once on component mount


    useEffect(() => {
      if (shaderMaterialRef.current) {
        shaderMaterialRef.current.uniforms.u_progress.value = progress; // updates the shader's progress
      }
    }, [progress])


    const handleButtonClick = () => {
        //animate to close
      if(progress < 1){
        let currentProgress = progress;
        const updateAnimation = () => {
            if(currentProgress < 1){
                currentProgress =  Math.min(1, currentProgress + 0.025);
                setProgress(currentProgress);
                requestAnimationFrame(updateAnimation)
            }
        }
        updateAnimation();
      }else {
        //animate to open
        let currentProgress = progress;
        const updateAnimation = () => {
          if(currentProgress > 0){
              currentProgress =  Math.max(0, currentProgress - 0.025);
              setProgress(currentProgress);
              requestAnimationFrame(updateAnimation)
          }
      }
      updateAnimation();
      }

  };

  return (
    <div>
      <div ref={mountRef} style={{ width: '400px', height: '300px' }} />
      <button onClick={handleButtonClick}>Animate Curtain</button>
    </div>
  );
};

export default CurtainShader;

```

**Example 2: Implementing a basic 'curtain' wipe**

This extends the previous example by using a simple step function within the fragment shader to create a vertical wipe effect, commonly seen with curtain transitions.

```javascript
// Same import as above
import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';


const CurtainShader = () => {
  const mountRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const shaderMaterialRef = useRef(null);


  useEffect(() => {
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;


    // Scene setup
    sceneRef.current = new THREE.Scene();
    cameraRef.current = new THREE.OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, 0.1, 1000);
    cameraRef.current.position.z = 10;
    rendererRef.current = new THREE.WebGLRenderer({ alpha: true });
    rendererRef.current.setSize(width, height);
    mountRef.current.appendChild(rendererRef.current.domElement);


    // Shader setup
     const vertexShader = `
      void main() {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;
    const fragmentShader = `
      uniform float u_progress;
       varying vec2 vUv;

      void main() {
         vec3 color1 = vec3(0.0, 0.0, 1.0);
         vec3 color2 = vec3(1.0, 0.0, 0.0);
        float mask = step(vUv.x, u_progress);
         gl_FragColor = vec4(mix(color1, color2, mask), 1.0);
      }
    `;


    shaderMaterialRef.current = new THREE.ShaderMaterial({
      vertexShader: vertexShader,
      fragmentShader: fragmentShader,
      uniforms: {
        u_progress: { value: 0 },
      },
    });


    // Plane setup
    const planeGeometry = new THREE.PlaneGeometry(width, height);
    const planeMesh = new THREE.Mesh(planeGeometry, shaderMaterialRef.current);
    sceneRef.current.add(planeMesh);


    // Animation loop setup
    const animate = () => {
      if (shaderMaterialRef.current) {
        shaderMaterialRef.current.uniforms.u_progress.value = progress;
      }
      rendererRef.current.render(sceneRef.current, cameraRef.current);
      requestAnimationFrame(animate);
    };
    animate();
     return () => {
        if(mountRef.current) {
            mountRef.current.removeChild(rendererRef.current.domElement);
        }
        rendererRef.current.dispose();
        shaderMaterialRef.current.dispose();
        planeGeometry.dispose();
      };


  }, []);


  useEffect(() => {
    if (shaderMaterialRef.current) {
      shaderMaterialRef.current.uniforms.u_progress.value = progress;
    }
  }, [progress])


  const handleButtonClick = () => {
      //animate to close
    if(progress < 1){
        let currentProgress = progress;
        const updateAnimation = () => {
            if(currentProgress < 1){
                currentProgress =  Math.min(1, currentProgress + 0.025);
                setProgress(currentProgress);
                requestAnimationFrame(updateAnimation)
            }
        }
        updateAnimation();
    }else {
        //animate to open
        let currentProgress = progress;
        const updateAnimation = () => {
            if(currentProgress > 0){
                currentProgress =  Math.max(0, currentProgress - 0.025);
                setProgress(currentProgress);
                requestAnimationFrame(updateAnimation)
            }
        }
        updateAnimation();
    }


  };


  return (
    <div>
      <div ref={mountRef} style={{ width: '400px', height: '300px' }} />
      <button onClick={handleButtonClick}>Animate Curtain</button>
    </div>
  );
};


export default CurtainShader;
```

**Example 3: Incorporating Texture and UVs**

Here we get closer to a practical use case by mapping a texture to our plane and animating the curtain wipe based on UV coordinates. The key here is how we're using `vUv` in the fragment shader, which is passed from the vertex shader. This represents the texture coordinates for each pixel.

```javascript
// Same imports as above
import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';


const CurtainShader = () => {
  const mountRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const shaderMaterialRef = useRef(null);
  const textureRef = useRef(null);


  useEffect(() => {
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;

       // Scene setup
    sceneRef.current = new THREE.Scene();
    cameraRef.current = new THREE.OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, 0.1, 1000);
    cameraRef.current.position.z = 10;
    rendererRef.current = new THREE.WebGLRenderer({ alpha: true });
    rendererRef.current.setSize(width, height);
    mountRef.current.appendChild(rendererRef.current.domElement);

    const textureLoader = new THREE.TextureLoader();
    textureRef.current = textureLoader.load('path/to/your/image.jpg'); // Replace with your image
    textureRef.current.flipY = false;


     // Shader setup
     const vertexShader = `
     varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const fragmentShader = `
      uniform float u_progress;
      uniform sampler2D u_texture;
      varying vec2 vUv;


      void main() {
        vec4 texColor = texture2D(u_texture, vUv);
        float mask = step(vUv.x, u_progress); // vertical wipe
        if( mask > 0.5){
            gl_FragColor = texColor;
        }else {
            gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }


      }
    `;


    shaderMaterialRef.current = new THREE.ShaderMaterial({
      vertexShader: vertexShader,
      fragmentShader: fragmentShader,
        uniforms: {
          u_progress: { value: 0 },
           u_texture: { value: textureRef.current },
        },
    });


    // Plane setup
    const planeGeometry = new THREE.PlaneGeometry(width, height);
    const planeMesh = new THREE.Mesh(planeGeometry, shaderMaterialRef.current);
    sceneRef.current.add(planeMesh);


     // Animation loop setup
     const animate = () => {
        if(shaderMaterialRef.current) {
            shaderMaterialRef.current.uniforms.u_progress.value = progress;
        }
        rendererRef.current.render(sceneRef.current, cameraRef.current);
        requestAnimationFrame(animate);
      };
      animate();
      return () => {
        if(mountRef.current) {
            mountRef.current.removeChild(rendererRef.current.domElement);
        }
        rendererRef.current.dispose();
        shaderMaterialRef.current.dispose();
        planeGeometry.dispose();
        textureRef.current.dispose();
      };


  }, []);

    useEffect(() => {
        if (shaderMaterialRef.current) {
        shaderMaterialRef.current.uniforms.u_progress.value = progress;
        }
    }, [progress])


  const handleButtonClick = () => {
      //animate to close
      if(progress < 1){
          let currentProgress = progress;
          const updateAnimation = () => {
              if(currentProgress < 1){
                  currentProgress =  Math.min(1, currentProgress + 0.025);
                  setProgress(currentProgress);
                  requestAnimationFrame(updateAnimation)
              }
          }
          updateAnimation();
        }else {
            //animate to open
            let currentProgress = progress;
            const updateAnimation = () => {
                if(currentProgress > 0){
                    currentProgress =  Math.max(0, currentProgress - 0.025);
                    setProgress(currentProgress);
                    requestAnimationFrame(updateAnimation)
                }
            }
            updateAnimation();
        }


  };


  return (
    <div>
      <div ref={mountRef} style={{ width: '400px', height: '300px' }} />
      <button onClick={handleButtonClick}>Animate Curtain</button>
    </div>
  );
};


export default CurtainShader;
```

This is where the real magic starts to happen. The shader now uses a texture and applies a curtain effect based on the `u_progress` uniform.

**Key Considerations:**

*   **Performance:** Avoid complex calculations within your fragment shader for mobile devices. Keep it lean and effective.
*   **Responsiveness:** Handle window resizing appropriately to prevent stretching. You'll have to update the renderer size and camera properties.
*   **Texture Loading:** Ensure textures are loaded before the shader attempts to use them. Use asynchronous image loading, such as Promises with texture loader to handle loading textures safely.
*   **Shader Code:** GLSL is different from JavaScript. It’s recommended to learn the basics. *The Book of Shaders* by Patricio Gonzalez Vivo is a fantastic resource for that. Additionally, for general WebGL knowledge, I'd suggest starting with *WebGL Programming Guide* by Kouichi Matsuda and Rodger Lea.
*   **Three.js:** *Three.js Journey* by Bruno Simon is a detailed and great learning resource. *three.js* documentation is also essential for understanding the api's.

This approach, combining React's component lifecycle with *three.js* and custom shaders, provides a powerful way to create engaging, visually striking transitions. Remember to fine-tune the fragment shader for the specific curtain effect you're aiming for. And as with any graphics-intensive application, performance is paramount, so always be mindful of keeping your shader calculations simple and efficient. It's a journey with a learning curve but leads to very interesting results.
