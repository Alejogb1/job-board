---
title: "highp shader code precision meaning?"
date: "2024-12-13"
id: "highp-shader-code-precision-meaning"
---

 so highp shader code precision I've been down this rabbit hole a bunch let me tell you my personal experience it's not exactly a picnic

See back in my early days I was messing around with this mobile game project it involved some pretty intensive fragment shaders I was aiming for smooth gradients and some fancy post-processing effects everything looked fine on my development machine a beast of a workstation but then I deployed it on some older android phones it was an absolute dumpster fire Banding all over the place colors were visibly stepping and the overall look was just plain awful That's when I realized the precision of shader variables had significant consequences

So you're basically asking about `highp` right In GLSL or similar shading languages we have variable precision specifiers `lowp` `mediump` and `highp` These tell the compiler the minimum precision at which operations on that variable should be carried out Think of it like this `highp` variables are like using a super precise scientific scale versus `lowp` is more akin to estimating weight by hand

`lowp` uses the fewest bits to store a value which leads to memory savings and often faster processing particularly on mobile GPUs but that comes with the trade-off that the range of values that can be represented accurately is limited You can get precision issues especially when dealing with larger number values or subtle changes in values `mediump` is kind of the middle ground it offers a balance between accuracy and performance It's generally fine for many use cases like calculating texture coordinates and colors but can still run into trouble with specific computations Then we have `highp` this one uses the most bits to represent a value offering the highest accuracy and range I've found that `highp` floats are usually represented as 32-bit floating point numbers which means you have a ton of precision and you're less likely to encounter artifacts related to lack of precision but its more memory consumption and a little slower operations

 let's get to some code examples

Example 1: Simple Gradient issue without Highp

```glsl
#version 300 es
precision mediump float;
out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(600, 400);
    float gradient = uv.x;
    fragColor = vec4(gradient, gradient, gradient, 1.0);
}
```

Now try run this on older mobile devices you will see banding if not you might be a little luckier with the GPU that is in use this shader calculates a simple horizontal gradient based on the x-coordinate of the fragment Now with mediump its -ish on some devices but on a large scale or larger resolutions it can lead to noticeable banding This is because the available precision is insufficient to accurately represent the small changes in color intensity across the gradient that you intended

Example 2: The same example using highp

```glsl
#version 300 es
precision highp float;
out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(600, 400);
    float gradient = uv.x;
    fragColor = vec4(gradient, gradient, gradient, 1.0);
}
```

With highp you should be looking at a smooth and clean gradient without any stepping issues highp allows the shader to represent fine color changes without losing data on a higher bit size its a trade of performance and quality. This ensures that the changes are represented smoothly in the output

Example 3: Highp with complex operations

```glsl
#version 300 es
precision highp float;
out vec4 fragColor;

uniform float time;

float complexCalculation(float x, float y) {
    float a = sin(x * 20.0 + time);
    float b = cos(y * 15.0 - time * 0.5);
    return a * b;
}

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(600, 400);
    float result = complexCalculation(uv.x, uv.y);
    fragColor = vec4(vec3(result), 1.0);
}
```

This one does a bit more complex stuff involving sine and cosine functions these operations are prone to precision related problems in lower precision especially after a few calculations this is where highp is really critical for maintaining the quality of the output especially with the time variable in the calculations You should experiment with `mediump` and `lowp` and if you see any pixelation you will need `highp`

When to use what: `lowp` use for things like texture coordinates or color indices where minor inaccuracies aren't visually noticeable It is more performant so use them whenever it doesn't hurt the outcome `mediump` this is the default for many scenarios and great for texture color calculations and interpolations where a bit of precision loss is acceptable `highp` this one is vital for complex mathematical calculations large values or gradients where visual artifacts due to precision loss are unacceptable use it for all critical places in the shader

So what are the best practices Avoid doing expensive calculations in lower precision then converting to highp its generally more efficient to maintain the high precision for the entire pipeline When dealing with very large or very small values you should explicitly use highp Also testing on a range of devices is super important as precision support varies across different GPUs what looks fine on a high-end phone could be a disaster on a low-end one that's always fun

And also lets not forget about the number of calculations done in your shader if you use lowp and the result is used by a long chain of calculations that will introduce precision errors on the outcome and will create unexpected pixel artifacts so you should be looking on the whole calculations chain of variables too

Resource wise I'd recommend checking out some more serious resources instead of some random websites the OpenGL ES specification documentation is a great place for the official lowdown on precision qualifiers and their implications the Khronos website is your friend there you will find it you also could use the "OpenGL Shading Language" from Randy Fernando for a more in depth understanding on the nuances of GLSL and shader development especially this book delves into specific topics about shading including precision issues

Also while it's a bit old now "Real-Time Rendering" by Tomas Akenine-Moller et al is very handy for a broad understanding of graphics principles including practical implications of precision in rendering but be aware that these book might be more theory wise rather than directly practical use

hopefully that clears it up for you From my experience messing around with these settings can make a huge difference in terms of look and feel of your renders It’s not just about making it work; it’s about making it work well on all devices which can be hard to maintain but it's our job as dev

Oh and you know why I don't trust atoms They make up everything haha   i’ll get going
