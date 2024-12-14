---
title: "How to cut a part of a shape with the help of another painted shape above?"
date: "2024-12-14"
id: "how-to-cut-a-part-of-a-shape-with-the-help-of-another-painted-shape-above"
---

alright, so you're trying to chop out a section of a shape using another shape that's painted on top, huh? yeah, i've been there, spent a good couple of late nights debugging this kind of stuff. it seems pretty straightforward at first, but the devil is always in the details, especially when dealing with complex shapes and rendering. let me tell you about the time i was working on this interactive map application, and i needed to create these cut-out areas based on user-drawn polygons. it was a nightmare until i found a proper method. so i'll share what i learned.

first off, what you're describing is fundamentally a boolean operation, specifically a difference operation. we're subtracting the area of the 'upper' shape from the 'lower' shape. the hard part is doing this on the pixel level, efficiently, and also dealing with all the edge cases you will encounter.

the basic approach involves using what's called a stencil buffer (or a similar concept depending on your framework/api). a stencil buffer is essentially an extra layer in your graphics pipeline where you can store information about which parts of the screen should or shouldn't be rendered based on a defined operation. here's how this typically works:

1.  **render the 'upper' shape to the stencil buffer:**
    instead of writing to the main color buffer, we configure the rendering pipeline to write to the stencil buffer. we tell the system to increment the stencil value wherever a pixel from the 'upper' shape is drawn. crucially, we don't render the shape to the main color buffer *yet*.

2.  **set stencil test rules:** we define a rule to only render pixels that *do not* have a non-zero stencil value. in other words, we're essentially saying "only draw where the 'upper' shape wasn't drawn."

3.  **render the 'lower' shape:** now we render the 'lower' shape to the main color buffer. only the pixels of this shape where the stencil rule allows will be visible since we are only rendering the portion not covered by the 'upper' shape.

now this explanation is theory, and we always need to get our hands dirty. let's dive into some code snippets using different libraries/approaches, they’re not entirely complete, you will need to adapt them for your context, consider these as a starting point. i also assume the shapes are filled and not just outlined, that simplifies a lot of things.

**example 1: using canvas 2d api (javascript)**

```javascript
function cutShape(baseCanvas, cuttingCanvas, resultCanvas) {
  const baseCtx = baseCanvas.getContext('2d');
  const cuttingCtx = cuttingCanvas.getContext('2d');
  const resultCtx = resultCanvas.getContext('2d');

  if (!baseCtx || !cuttingCtx || !resultCtx) {
    console.error("could not get 2d context, check the elements!");
    return;
  }


  resultCanvas.width = baseCanvas.width;
  resultCanvas.height = baseCanvas.height;

  // draw the base shape on the result
  resultCtx.drawImage(baseCanvas, 0,0);

  // use the 'destination-out' compositing mode
  resultCtx.globalCompositeOperation = 'destination-out';
  // draw the cutting shape
  resultCtx.drawImage(cuttingCanvas, 0, 0);
  // reset the compositing
  resultCtx.globalCompositeOperation = 'source-over';


}

// example how to call it

const baseCanvas = document.getElementById('baseCanvas'); // base shape canvas
const cuttingCanvas = document.getElementById('cuttingCanvas'); // cutting shape canvas
const resultCanvas = document.getElementById('resultCanvas'); // result canvas

cutShape(baseCanvas, cuttingCanvas, resultCanvas);

```

**explanation:** this example leverages canvas compositing operations, specifically `destination-out`. the `destination-out` operation removes the overlapping pixels of the drawing on the `destination`, basically cutting out the shape. this approach, while it might look the easiest to implement and can work in simple cases, has some limitations since it doesn't really use stencils and is raster-based instead of being a true boolean operation using vectors. you can't do complex shapes or multiple subtractions with this method easily. also the performance of this kind of approach is not ideal.

**example 2: using opengl/webgl (javascript)**

this example is more verbose because it involves setting up the stencil buffer and the shader program. please note that i am only showing a relevant portion and assuming you have some basic gl knowledge.

```javascript
function cutShapeGl(gl, baseTexture, cuttingTexture, outputTexture, program){

  // assuming program is a compiled gl program with uniforms for our textures
    gl.useProgram(program);

  // bind framebuffer for rendering
    gl.bindFramebuffer(gl.FRAMEBUFFER, outputTexture.framebuffer);


    // enable stencil buffer operations
    gl.enable(gl.STENCIL_TEST);

  // configure stencil buffer to write only in areas where the cutting shape is drawn
    gl.stencilFunc(gl.ALWAYS, 1, 0xff);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);

    // render cutting shape, writing to the stencil buffer and not the color buffer
    gl.colorMask(false, false, false, false); //disable writing color buffer
    cuttingTexture.bind(0);
    gl.uniform1i(gl.getUniformLocation(program, 'texture0'), 0); //send texture 0 to fragment shader
    gl.drawArrays(gl.TRIANGLES, 0, 6); // assume there is a plane ready


    // configure stencil test to only render where the stencil is 0, meaning areas not covered by the cutting shape
    gl.colorMask(true, true, true, true); //re-enable writing to color buffer
    gl.stencilFunc(gl.EQUAL, 0, 0xff);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);

    // render base shape, only in areas where the stencil test passes (areas not occupied by the cutting shape)
    baseTexture.bind(0);
    gl.uniform1i(gl.getUniformLocation(program, 'texture0'), 0); //send texture 0 to fragment shader
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    //disable stencil test
    gl.disable(gl.STENCIL_TEST);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null); // render the resulting texture to the canvas
}


//example of shader code
// vertex shader
/*
    #version 300 es
    in vec3 a_position;
    out vec2 v_uv;

    void main() {
        gl_Position = vec4(a_position, 1.0);
	    v_uv = a_position.xy * 0.5 + 0.5;
    }
*/
//fragment shader
/*
    #version 300 es
    precision mediump float;
    in vec2 v_uv;
    uniform sampler2D texture0;
    out vec4 fragColor;

    void main() {
        fragColor = texture(texture0, v_uv);
    }
*/
```

**explanation:** this example is more involved, but it demonstrates how to perform the shape cutting using stencil buffers and a shader program in webgl. it uses webgl commands like `gl.stencilFunc`, `gl.stencilOp`, `gl.colorMask` to modify how the stencil buffer and color buffers are written. you can also achieve very powerful things if you understand the glsl shader language. you can add custom visual effects, and more. the shader code example is a very basic example just for showing how to do this with textures.

**example 3: using svg path operations**

if your shapes are vector-based and not rasterized images, you can leverage svg path operations. this approach allows you to do actual vector boolean operations. you'll need a library for handling svg paths. i'll use a library called `svg-path-boolean` for demonstration purposes:

```javascript
import { difference } from 'svg-path-boolean';

function cutShapeSvg(basePath, cuttingPath) {
    try {
      const result = difference(basePath, cuttingPath);
      return result;
    } catch (error){
      console.error("error happened with path operation, check your svgs:", error);
      return null;
    }
}


// example

const basePath = 'M10 10 L100 10 L100 100 L10 100 Z';
const cuttingPath = 'M30 30 L80 30 L80 80 L30 80 Z';

const resultPath = cutShapeSvg(basePath, cuttingPath);
// result path: 'M10 10 L100 10 L100 100 L10 100 L10 30 L30 30 L30 80 L10 80 L10 10 Z'

if (resultPath){
  const svgElement = document.getElementById('svg-container');
  svgElement.innerHTML = `<path d="${resultPath}" fill="blue" />`
}
```

**explanation:** with vector data, we can use dedicated libraries to compute the difference operation between the paths, the library `svg-path-boolean` takes two svg paths and return a new path with the subtracted shapes, this approach is the correct method to do boolean operations. it will return a path that can be rendered directly in an svg element. the downside, is that this is only usefull if you are dealing with vector graphics and svg data. and also that we are adding one extra library for this kind of operation. in the real world many applications often use a combination of these techniques based on their needs, rasterization with the `canvas api` or the graphics hardware, with `opengl` and also vector manipulation with `svg` paths.

now a little bit of advice based on my past experience with this type of stuff, if your shapes are simple and you are dealing with relatively small canvas/render target sizes, the `canvas api` approach can be ok and performant enough. but, if your shapes are complex, have many vertices and you need a lot of performance then you *should* explore methods that use stencil buffers such as `opengl`/`webgl`. for vector data, prefer library implementations that do actual boolean operations instead of rasterizing, this means libraries that can use the paths you provide as input to compute new paths. when using svg don't forget to install the necessary library.

also remember that there are a few tricky cases, like when shapes overlap exactly or when very thin lines are involved. you'll have to test your implementation with a wide variety of shapes to iron out all the kinks. and sometimes the only way to be sure is actually measuring performance, some operations like boolean subtraction are costly specially if done many times per second.

as for resources, i would recommend "real-time rendering" by tomas akenine-möller. it is a very exhaustive resource covering rendering and modern techniques for creating visualisations. also some papers on computational geometry, are usefull if your shapes are vectors, and want to do boolean operations with vector based data such as svg paths. some libraries like threejs or opengl can help a lot with the complexity of rendering and you might find some inspiration looking at how they do things. and always remember that nothing works at the first try, so be prepared to debug for hours :) this is how software engineering is. (sometimes you see developers trying to debug code and it looks like they are just staring at the screen. but i'll tell you a little secret: we are not staring we are actually debugging!). good luck and happy coding.
