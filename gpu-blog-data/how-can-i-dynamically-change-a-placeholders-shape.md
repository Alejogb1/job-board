---
title: "How can I dynamically change a placeholder's shape?"
date: "2025-01-30"
id: "how-can-i-dynamically-change-a-placeholders-shape"
---
The challenge of dynamically altering a placeholder’s shape primarily arises within the context of interactive user interfaces, specifically those employing graphics libraries or technologies where visual elements are often rendered programmatically. Simply swapping placeholder images won’t suffice if the requirement is for a transition or for a shape morph, requiring manipulation of the underlying graphic primitives or the application of specific effects. In my experience developing custom UI components for data visualization dashboards, I frequently encountered this need, prompting a deeper exploration of shape manipulation techniques.

To achieve this, one must interact with the drawing mechanism of the framework or library being used, rather than relying solely on static image replacement. The methods available vary significantly depending on the tools selected, ranging from manipulation of vectors to controlling parameters on procedural drawing functions. The goal is to create a placeholder, often a simple shape like a rectangle or a circle, and then define the desired shape change and animate the transition if necessary. This dynamic adjustment typically involves updating the shape's properties—its vertices, radius, or any defining parameters—within the render cycle.

Let’s consider some implementations, assuming different common scenarios and tools:

**1. Using HTML5 Canvas**

The HTML5 Canvas element provides a powerful pixel-based drawing API. While it doesn’t have predefined shapes that can be directly morphed, we can achieve the illusion of shape change by redrawing the canvas with updated parameters in each animation frame. This approach requires manual calculation of intermediate shapes during the transition.

```javascript
// Example: morphing a circle into a square
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

let progress = 0; // Animation progress (0-1)
const duration = 1000; // Animation duration in milliseconds
const startTime = null;
let centerX = canvas.width / 2;
let centerY = canvas.height / 2;
let radius = 50;

function drawShape(progressValue){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();

    const cornerRadius = radius * (1 - progressValue)
    const squareSide = radius * 2;

    if (progressValue < 0.5){
      // Draw circle
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    } else {
      // Draw square with rounded corners
      const adjustedProgress = (progressValue - 0.5) * 2; // Convert 0.5-1 to 0-1
        const x = centerX - squareSide / 2;
        const y = centerY - squareSide / 2;

        ctx.moveTo(x + cornerRadius, y);
        ctx.lineTo(x + squareSide - cornerRadius, y);
        ctx.arcTo(x + squareSide, y, x + squareSide, y + cornerRadius, cornerRadius);
        ctx.lineTo(x + squareSide, y + squareSide - cornerRadius);
        ctx.arcTo(x + squareSide, y + squareSide, x + squareSide - cornerRadius, y + squareSide, cornerRadius);
        ctx.lineTo(x + cornerRadius, y + squareSide);
        ctx.arcTo(x, y + squareSide, x, y + squareSide - cornerRadius, cornerRadius);
        ctx.lineTo(x, y + cornerRadius);
        ctx.arcTo(x, y, x + cornerRadius, y, cornerRadius);

    }
    ctx.fillStyle = 'blue';
    ctx.fill();
}

function animate(timestamp) {
    if (!startTime) startTime = timestamp;
    const elapsed = timestamp - startTime;
    progress = Math.min(elapsed / duration, 1);
    drawShape(progress);
    if (progress < 1) {
        requestAnimationFrame(animate);
    }
}

requestAnimationFrame(animate);

```

*Commentary*:  This example uses the canvas API. The `drawShape` function dynamically determines whether to draw a circle or a square with rounded corners based on an animation `progress` variable. The `animate` function manages the timing and updates the drawing. We transition from a circle to a square by gradually altering the drawing parameters. The first 50% of the animation renders the circle; after that, we draw a square with increasingly less rounded corners until the corners are sharp. We calculate a progress variable to interpolate between circle and square drawing parameters smoothly.

**2. Using Scalable Vector Graphics (SVG)**

SVG offers a more declarative way to handle shapes.  SVG paths, in particular, can be animated by modifying their “d” attribute, which defines the path's drawing instructions. This allows for more precise and often more performant shape changes compared to repeatedly redrawing a canvas.

```html
<svg width="200" height="200">
  <path id="myPath" d="M 100 50 a 50 50 0 1 1 0 100 a 50 50 0 1 1 0 -100" fill="green" />
</svg>
<script>
  const path = document.getElementById('myPath');
  let progress = 0;
  const duration = 1000;
  let startTime = null;
  const startPath = "M 100 50 a 50 50 0 1 1 0 100 a 50 50 0 1 1 0 -100"; // circle
  const endPath = "M 50 50 l 100 0 l 0 100 l -100 0 z" // square

  function morphPath(progressValue) {
      const currentPath = interpolatePath(startPath, endPath, progressValue);
      path.setAttribute('d', currentPath);
  }

  function interpolatePath(startPath, endPath, progressValue) {
    const startCommands = extractPathCommands(startPath);
    const endCommands = extractPathCommands(endPath);
    if (startCommands.length !== endCommands.length) { return startPath;} // Ensure they match

    let newPath = ""
    for (let i = 0; i < startCommands.length; i++) {
      const startCmd = startCommands[i];
      const endCmd = endCommands[i];

      newPath += startCmd[0]; // Copy the Command type (M,L,A etc.)

       for (let j = 1; j < startCmd.length; j++) {
          const startVal = parseFloat(startCmd[j]);
          const endVal = parseFloat(endCmd[j]);
           const interpolatedValue = startVal + (endVal - startVal) * progressValue
           newPath += " " + interpolatedValue;
       }
    }
    return newPath
}
  function extractPathCommands(pathString) {
    const commands = [];
    let currentCommand = [];
    for (const part of pathString.trim().split(/([MLHVCSQTAZmlhvcsqtaz])/)) {
        if (part.match(/([MLHVCSQTAZmlhvcsqtaz])/)) {
            if (currentCommand.length > 0) {
                commands.push(currentCommand);
            }
            currentCommand = [part];
        }
         else if (part.trim()) {
            currentCommand.push(...part.trim().split(/[\s,]+/));
        }
    }
    if (currentCommand.length > 0) {
          commands.push(currentCommand);
      }
    return commands;
}


  function animate(timestamp) {
    if (!startTime) startTime = timestamp;
    const elapsed = timestamp - startTime;
    progress = Math.min(elapsed / duration, 1);
    morphPath(progress);

    if (progress < 1) {
        requestAnimationFrame(animate);
    }
  }
  requestAnimationFrame(animate);
</script>

```

*Commentary*:  This code defines an SVG path element, initially shaped as a circle. The Javascript then interpolates between this circular path and a square path, updating the "d" attribute in each animation frame. The `interpolatePath` function ensures the paths have matching commands for a smooth transition, and if commands mismatch returns the start path. We also convert all numbers within commands to interpolated values based on the `progressValue`. The animation loop then adjusts this attribute, morphing the shape.

**3. Using a Library Like Three.js**

For 3D or more complex shape manipulation, a library like Three.js is often beneficial. Three.js handles the heavy lifting of rendering and provides a variety of geometry manipulation capabilities.  In this example, we would modify vertex data for mesh manipulation, demonstrating that even complex shapes can be transformed dynamically.

```javascript
import * as THREE from 'three';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const geometry = new THREE.SphereGeometry(1, 32, 32);
const material = new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true });
const sphere = new THREE.Mesh(geometry, material);
scene.add(sphere);
camera.position.z = 5;

let progress = 0;
const duration = 2000;
let startTime = null;
let originalVertices = [];
for (let i=0; i < geometry.attributes.position.count; i++){
    originalVertices.push(geometry.attributes.position.array[i*3],geometry.attributes.position.array[i*3+1],geometry.attributes.position.array[i*3+2]);
}


function morphShape(progressValue){
  for (let i=0; i < geometry.attributes.position.count; i++){
        const x = originalVertices[i*3];
        const y = originalVertices[i*3+1];
        const z = originalVertices[i*3+2];
      const newX =  x * (1 - progressValue) + (x * 1.5)*progressValue;
      const newY =  y * (1 - progressValue) + (y * 0.5)*progressValue;
        const newZ =  z * (1 - progressValue);


        geometry.attributes.position.array[i*3] = newX
        geometry.attributes.position.array[i*3+1] = newY
        geometry.attributes.position.array[i*3+2] = newZ;
  }
  geometry.attributes.position.needsUpdate = true;
}

function animate(timestamp) {
    if (!startTime) startTime = timestamp;
    const elapsed = timestamp - startTime;
    progress = Math.min(elapsed / duration, 1);
    morphShape(progress);

    renderer.render(scene, camera);

    if (progress < 1) {
        requestAnimationFrame(animate);
    }
}

requestAnimationFrame(animate);
```

*Commentary*:  This Three.js example creates a sphere. Instead of using a predefined square, I chose to stretch this sphere, demonstrating more advanced manipulation. The vertex positions of the sphere are saved initially. In the `morphShape` function we manipulate the vertex coordinates based on the animation progress. In this case we are stretching the sphere in x, y directions and moving it on z direction. This shows the process of directly manipulating geometry properties to achieve a morphing effect.

In summary, changing a placeholder’s shape dynamically requires direct interaction with the drawing or rendering system you are using, whether it is a low-level API like the canvas, a declarative approach like SVG, or a high-level framework like Three.js.

For further exploration, I recommend reviewing resources on:

1. **HTML5 Canvas API** documentation which provides methods for drawing shapes, setting colors, and managing animation.
2. **SVG path animation** techniques, which describe how to manipulate the path data for transitions.
3. **Graphics libraries like Three.js** which document functions for accessing and manipulating mesh geometry.
4. **Animation timing and easing** functions, which assist with creating smooth transitions between shapes.
5. **Vector math fundamentals** for manipulating vertex data.
Exploring these areas would provide a solid foundation for implementing complex dynamic shape manipulation requirements.
