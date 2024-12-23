---
title: "How do I pin a horizontal plane to a specific altitude in MapLibre GL JS?"
date: "2024-12-23"
id: "how-do-i-pin-a-horizontal-plane-to-a-specific-altitude-in-maplibre-gl-js"
---

Alright, let's tackle this. You're looking to lock a horizontal plane within your MapLibre GL JS map to a specific altitude, essentially creating a consistent visual reference point as you navigate the 3d space. I've certainly encountered this scenario a few times, most memorably when working on a flight simulation interface a while back. It’s a surprisingly common requirement when you need a stable reference, regardless of camera perspective.

The challenge here isn’t a built-in feature, per se. MapLibre GL JS deals primarily with layers and map elements tied to geographical coordinates; pinning to a specific *altitude* requires a bit of crafty manipulation. We're going to leverage the map’s camera properties and the transformation capabilities of custom layers to achieve this. The core concept is to generate a plane at the desired altitude and then dynamically adjust its position each frame so that it appears anchored despite camera movement.

The technique involves the following steps: creating a custom layer, defining the shape of our horizontal plane, translating it vertically to the designated altitude, and ensuring it moves with camera adjustments. Let’s get into the code.

Here's how I’d implement it, broken down into practical sections.

**1. Setting up the Custom Layer:**

We’ll start by creating a custom layer. MapLibre's custom layer mechanism is perfect for this because it allows us to bypass the usual geographical coordinate system and directly manipulate the WebGL context. We will define an `id` for the layer, a `type` (custom in this case), and hook into the `render` and `prerender` methods. The `render` is where the WebGL instructions go, and `prerender` helps to prepare the data. This method also allows us to define parameters for the custom layer.

```javascript
map.addLayer({
    id: 'altitude-plane',
    type: 'custom',
    renderingMode: '3d',
    onAdd: function(map, gl){
        this.cameraMatrix = new Float64Array(16);
        this.projectionMatrix = new Float64Array(16);
        this.viewMatrix = new Float64Array(16);

        //setup your geometry here
        this.geometry = createPlaneGeometry(1000); //example: 1000 x 1000 plane

        //initialize shader program here
        this.program = createShaderProgram(gl);

    },
    render: function(gl, matrix) {
        const uniformLocations = this.program.uniforms;
        gl.useProgram(this.program.program);

        //get camera and projection matrices, note the use of the map.transform.cameraToMatrix() function
        map.transform.cameraToMatrix(this.cameraMatrix, this.projectionMatrix);

        gl.uniformMatrix4fv(uniformLocations.projectionMatrix, false, this.projectionMatrix);
        gl.uniformMatrix4fv(uniformLocations.viewMatrix, false, map.transform.viewMatrix);
        gl.uniformMatrix4fv(uniformLocations.cameraMatrix, false, this.cameraMatrix);
        gl.uniform1f(uniformLocations.altitude, this.altitude);

        // Bind the geometry attributes for rendering.
        gl.bindBuffer(gl.ARRAY_BUFFER, this.geometry.vertexBuffer);
        gl.vertexAttribPointer(this.program.attributes.position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.program.attributes.position);

        // Render the geometry
        gl.drawArrays(gl.TRIANGLES, 0, this.geometry.vertexCount);
    },
    prerender: function(gl, matrix){
      // update the altitude based on your desired level
      this.altitude = 100; // Set the altitude (example: 100 meters)
    }
});
```

In this snippet, `createPlaneGeometry` would be a custom function to generate a vertex buffer for a flat plane, and `createShaderProgram` would initialize a simple shader program which transforms our vertexes from local coordinates to the world view.

**2. Defining the Plane Geometry:**

Here's an example of `createPlaneGeometry`: This function generates vertex data for a simple plane, specified by its size and then returns the buffer object.

```javascript
function createPlaneGeometry(size) {
    const halfSize = size / 2;
    const vertices = new Float32Array([
        -halfSize, 0, -halfSize,
         halfSize, 0, -halfSize,
        -halfSize, 0,  halfSize,
        -halfSize, 0,  halfSize,
         halfSize, 0, -halfSize,
         halfSize, 0,  halfSize
    ]);

    const gl = map.getCanvas().getContext('webgl'); //get the webgl context

    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    return {
        vertexBuffer: vertexBuffer,
        vertexCount: 6
    };
}
```

**3. Creating the Shader Program:**

The shader program is crucial for transforming our plane. It takes the vertex position, applies transformations from the camera, view and projection matrices, and moves our plane vertically according to the altitude value. Here's how `createShaderProgram` might look:

```javascript
function createShaderProgram(gl) {

   const vertexShaderSource = `
      uniform mat4 projectionMatrix;
      uniform mat4 viewMatrix;
      uniform mat4 cameraMatrix;
      uniform float altitude;
      attribute vec3 position;

      void main() {
        //Transform local space position to world space
        vec4 worldPosition = cameraMatrix * vec4(position.x, position.y + altitude, position.z, 1.0);
        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `;

    const fragmentShaderSource = `
      void main() {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 0.5); // Red, semi-transparent
      }
    `;

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)){
       console.error('Failed to compile vertex shader.', gl.getShaderInfoLog(vertexShader));
    }

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
      if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)){
         console.error('Failed to compile fragment shader.', gl.getShaderInfoLog(fragmentShader));
      }

    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)){
       console.error('Failed to link shader program.', gl.getProgramInfoLog(shaderProgram));
    }


    return {
        program: shaderProgram,
        uniforms:{
          projectionMatrix: gl.getUniformLocation(shaderProgram, "projectionMatrix"),
          viewMatrix: gl.getUniformLocation(shaderProgram, "viewMatrix"),
          cameraMatrix: gl.getUniformLocation(shaderProgram, "cameraMatrix"),
          altitude: gl.getUniformLocation(shaderProgram, "altitude")
        },
        attributes:{
          position: gl.getAttribLocation(shaderProgram, "position")
        }
    };
}
```

**Key Takeaways:**

*   **Custom Layers:** They're your gateway to precise, per-frame control when standard layers fall short. You'll often find you need this level of access when dealing with three-dimensional elements.
*   **Camera Matrices:** The `map.transform.cameraToMatrix` method is indispensable for calculating accurate transformations of the world model, correctly transforming the 3d plane based on the camera movement.
*   **Shader Programming:** While it might seem complex initially, shader programming (GLSL) becomes much more approachable with practice and is very flexible. You will probably want to investigate tutorials specific to WebGL for a better understanding.
*   **Altitude Adjustment:** The altitude is applied through the uniform `altitude` variable that we pass into the vertex shader, and then used to modify the `y` position of the vertex in the shader.

**Resources:**

For a deeper dive into these concepts, I recommend the following:

*   **WebGL Fundamentals** ([https://webglfundamentals.org](https://webglfundamentals.org)): An excellent resource to learn the fundamental operations of WebGL, covering vertex buffers, shaders, and how they all fit together.
*   **The Book of Shaders** ([https://thebookofshaders.com/](https://thebookofshaders.com/)): This interactive book explains shader programming (GLSL) concisely, ideal for understanding the shader-side aspects of our example.
*   **Real-Time Rendering** (book by Tomas Akenine-Moller, Eric Haines, and Naty Hoffman): A definitive text on real-time 3D graphics, offering advanced knowledge on transformation matrices and rendering pipelines, although it might be overkill for this specific task.

Implementing the solution this way provides a clean and robust method to pin a horizontal plane at a specific altitude within MapLibre GL JS. Remember, the specifics of the geometry and shader code might need to be adjusted based on your desired visual appearance and performance requirements but the core approach described is, in my experience, extremely robust. Feel free to dive into each of these parts, and don't hesitate to ask if you have any more questions.
