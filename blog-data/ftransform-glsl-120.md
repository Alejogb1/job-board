---
title: "ftransform glsl 1.20?"
date: "2024-12-13"
id: "ftransform-glsl-120"
---

Okay so ftransform glsl 120 right Been there done that Got my share of headaches with that specific version of the old school GLSL pipeline Let me tell you it's a journey a trip down memory lane of fixed function rendering and quirks You're asking about ftransform which basically is the built-in that transforms vertex positions from object space to clip space It was the workhorse back in the days when vertex shaders were optional a quaint memory if you ask me

Back then ftransform was how you did your basic modelview projection transformations It was all matrix multiplications and not much wiggle room It was part of the vertex program and was handled behind the scenes No access to modify or tweak the process outside of matrix manipulation That's why when vertex shaders became more mainstream and fully customizable ftransform slowly faded out We're way past those times now almost like dinosaurs

I recall a project I did years back building a simple rendering engine using OpenGL 2.1 and GLSL 120 It was all about wireframe spheres and cubes but it taught me the ropes We started with the classic ftransform setup and boy was it a pain to debug It's tricky because you're working with matrices directly so any mistake there screws up everything Geometry goes off to neverland perspective projection looks like a mangled mess you name it

The typical setup looked something like this in the vertex shader part where you would be doing a bunch of things before the fragment shader takes over

```glsl
#version 120

void main() {
  gl_Position = ftransform();
  gl_FrontColor = gl_Color; // Pass through vertex color
}
```
See not much to it right It all happens behind the ftransform call The heavy lifting was all in setting up those matrices properly before submitting them to the shader That included modelview matrix projection matrix those had to be computed on the CPU and passed as uniform variables

Later we moved on to more sophisticated projects where ftransform was a bottleneck We needed per-vertex lighting complex deformations and other advanced stuff that the fixed pipeline simply couldn't handle That’s where custom vertex shaders really shine we got to explicitly define those matrix multiplications and had greater control

Now remember this is old school stuff So you wouldn't see ftransform in modern OpenGL or WebGL code for example But it's worth understanding for historical context and if you ever have to deal with legacy code

Here’s a more involved example a bit more realistic where we have color and a variable color based on distance from a point in space in a fixed-function emulation

```glsl
#version 120

uniform mat4 modelview_matrix;
uniform mat4 projection_matrix;
uniform vec3 light_position;
uniform float radius;

void main() {
  // Transform vertex to world space
  vec4 world_position = modelview_matrix * gl_Vertex;

  // Compute distance from light
  float distance = length(world_position.xyz - light_position);

  // Attenuation factor using a simple inverse-square approximation
  float attenuation = 1.0 / (1.0 + distance * distance);
  float clampedAttenuation = clamp(attenuation, 0.0, 1.0);

  // Calculate color
  vec4 finalColor =  clampedAttenuation * gl_Color ;

  // Setup fixed function inputs
  gl_Position = projection_matrix * world_position;
  gl_FrontColor = finalColor;
}
```
This code is a sort of emulation using matrix uniform to transform the vertex and perform color calculations Before calling ftransform you had to calculate the final position which was the key of ftransform as a black box

The matrix setup on the CPU side was crucial The projection matrix usually was set up once at the start if you are not playing with the camera perspective which usually uses functions like `glOrtho` or `glFrustum` for orthographic and perspective projections respectively The modelview matrix on the other hand was updated every frame usually using `gluLookAt` or a custom camera class you may have coded

The whole thing looked something like this CPU side:

```cpp
// Assuming you have a math library for matrices
glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f), 800.0f/ 600.0f, 0.1f, 100.0f);
glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
glm::mat4 modelMatrix = glm::mat4(1.0f);  // Identity matrix for the model
glm::mat4 modelViewMatrix = viewMatrix * modelMatrix;

// Setting uniform values to the shader
glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection_matrix"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));
glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "modelview_matrix"), 1, GL_FALSE, glm::value_ptr(modelViewMatrix));

// Uniform for light
glm::vec3 lightPosition = glm::vec3(2.0f, 2.0f, 2.0f);
glUniform3fv(glGetUniformLocation(shaderProgram, "light_position"), 1, glm::value_ptr(lightPosition));

// Uniform for the radius
float radius = 5.0f;
glUniform1f(glGetUniformLocation(shaderProgram, "radius"), radius);
```

This is a basic example with a view projection modelview transformations light position calculation and a radius parameter All of these were typically done before the shader call in the CPU using glOrtho glFrustum gluLookAt or similar functions which is the matrix transformation part you would be setting for ftransform to execute properly if it was available in this example

Now the fun part ftransform itself didn't have much room to be manipulated within GLSL 120 It was a fixed function no variables no modification That was a key point about fixed function pipeline vs the programmable one You just fed the matrices you got your result in clip space If something was wrong then 99 percent of the time it was the matrices calculation itself or the order of multiplication and that is the root of most errors with old OpenGL fixed function pipeline and that is usually the first thing you would check before scratching your head a lot

The best resources for understanding ftransform and the old fixed-function pipeline are definitely textbooks on OpenGL from that era Look for titles like "OpenGL Programming Guide" often called the "Red Book" and "OpenGL SuperBible" Those would have all the gory details about fixed functions matrix setup projection types all that fun stuff Also there were some online tutorials that used to cover the old techniques but they are pretty scarce these days you'd likely have to rely on archived versions of sites or forums if you can find them

I mean there are more modern solutions if you want to do transformations like this with vertex shaders now you can write it in the shader itself and have more power over the pipeline If you want something like ftransform in GLSL 120 it is actually just a mat4 multiplication with the MVP matrix like we did in the example So if you're stuck with GLSL 120 then you can emulate it with what i provided in the second example although it's not the exact same thing if you don't have the same fixed function behaviour

Oh here's a joke to break things up Why did the programmer quit his job because he didn't get arrays hahahahaha okay okay back to work

Hope that helps clarify things You know always a trip to go back to that old stuff Feel free to hit me up if you've got more questions about dinosaurs I mean old GLSL pipelines
