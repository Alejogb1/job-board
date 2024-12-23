---
title: "triangle strip opengl drawing?"
date: "2024-12-13"
id: "triangle-strip-opengl-drawing"
---

 so you want to draw triangle strips in OpenGL right Been there done that Got the t-shirt and probably a few bug reports to boot Let's break it down I mean no one starts out knowing this stuff right?

I've been messing with OpenGL for probably close to 15 years now started back when fixed-function pipelines were still kinda a thing I remember pulling my hair out with immediate mode glBegin glEnd madness Then came vertex buffers and shaders thank god And yeah triangle strips they seemed so simple on paper but little quirks little gotchas all over the place

First off what are triangle strips? Well picture this you've got a sequence of vertices and instead of defining each triangle separately you define them in a strip so each new vertex uses the last two and forms a new triangle it saves a bunch of data it's more efficient especially for like long continuous mesh surfaces think a curved road a wave of water or even a simple ribbon

It’s efficient because you are reusing previously used vertices for more triangles think fewer CPU cycles fewer data to send down the GPU pipeline And when you get into really large models that efficiency adds up trust me

Now OpenGL has glDrawArrays and glDrawElements for this so its not like you have to reinvent the wheel I mean come on we're not masochists right glDrawArrays for non indexed mode means sending the strip directly while glDrawElements means having an index buffer that references your vertices

Let's start with the basics if you have a simple quad you might be tempted to draw it using two triangles But you know what that's not a triangle strip that’s two separate triangles so don't make that mistake I've seen that happen way too much to some folks

Here is a simple example of how a triangle strip is defined using floats which we are going to then copy into VBO later

```cpp
   float vertices[] = {
        -0.5f,  0.5f, 0.0f,  // v0 top left
        -0.5f, -0.5f, 0.0f,  // v1 bottom left
         0.5f,  0.5f, 0.0f,  // v2 top right
         0.5f, -0.5f, 0.0f   // v3 bottom right
    };
```

So what happens here is this: triangle 1 is v0 v1 and v2. Triangle 2 is v1 v2 and v3 and so on this is different from specifying two triangles such as v0 v1 v2 and v2 v1 v3 because that will generate separate drawing commands

Now how do you actually make OpenGL use these vertices? Well you need a Vertex Buffer Object a VBO and then specify the layout

Here's a snippet that sets up a VBO and renders it with glDrawArrays:

```cpp
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Vertex attribute setup (assuming shader uses location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Draw the strip
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
```

 let's break this code down a bit further first `glGenBuffers` and `glBindBuffer` are like giving the GPU space in RAM and pointing to it `glBufferData` is like copying data from RAM to GPU RAM

`glVertexAttribPointer` is where you tell the GPU how your vertex data is organised what data type it is and where to start and also strides in this case we are saying it has 3 floats per vertex and there is no gaps between vertices `glEnableVertexAttribArray` activates the attribute

finally `glDrawArrays` is where the rendering happens the first argument `GL_TRIANGLE_STRIP` is the mode the second argument `0` is the index to the first vertex and the third `4` is the number of vertices to draw

This is all non-indexed mode So we need to write indexed mode now which saves you even more resources when using strips because you do not send duplicate vertices to the GPU

Now sometimes your strip isn't this neat quad shape you might have complex stuff with more than a simple connected rectangle If you are using a complex shape its really useful to use an indexed triangle strip where you separate the vertex data from the indices pointing to them lets imagine we have this data we define vertices and an index buffer

```cpp
   float vertices[] = {
        -0.5f,  0.5f, 0.0f,  // v0
        -0.5f, -0.5f, 0.0f,  // v1
         0.5f,  0.5f, 0.0f,  // v2
         0.5f, -0.5f, 0.0f,  // v3
        -1.0f,  0.0f, 0.0f,  // v4
        1.0f,  0.0f, 0.0f   // v5
    };
    GLuint indices[] = {
        0, 1, 2, 3, 5, 4, 1
    };
```
Notice how the last index repeats `1` This is because when we define triangle strips it is  to start a new strip using a degenerate triangle In this case we are creating two strips The first one goes from `0 to 3` and the second one goes from `3 to 1` which includes the degenerate triangle `3 5 4`

And here’s how you'd render that with an element buffer object EBO or index buffer:

```cpp
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

     GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glDrawElements(GL_TRIANGLE_STRIP, 7, GL_UNSIGNED_INT, 0);
```
It’s pretty similar to the non indexed version, but we added an element buffer object or EBO where we bind the index data and instead of `glDrawArrays` we have `glDrawElements` where we specify that the indices are unsigned ints

One of the common gotchas with strips is making sure your vertices are in the right order Sometimes you’ll end up with backface culling issues or flipped normals because the winding order is wrong In the examples I provided it is counter-clockwise which is the default but sometimes you have to change it especially if you use different modelling softwares

Another one that messes people up is restarting strips If you have two separate strips that aren't contiguous you can't just keep going after that last vertex the GPU will make weird triangles you have to send a degenerate triangle to make the next triangle strip start properly this is also a problem I mentioned earlier with the indexed version

Debugging is key here I used to use a single vertex color in my shaders to spot issues This was especially good to see how the triangles are connected in the strip Then I used to color different strips in different ways to check the start of new ones

If you are digging deeper into performance you can use tools like renderdoc to profile your render calls and see how the rendering pipeline is using your data If you have really long strips split them to chunks instead because if one triangle is not viewable the GPU culls them all which leads to wasted GPU time

If you want to dive deeper into optimal memory layout in graphics I highly recommend reading about data layout optimization for graphics and also checking out papers from SIGGRAPH. Those academic papers are where the real good stuff is at They usually are in PDF so just google it with your needs

And about vertex attributes well keep it clean try to avoid too many attributes and the best way to keep it clean and performant is to make sure the attributes are tightly packed in memory and the sizes are kept to a minimum for example don't use a 3 float vector if you can get away with a 2 float vector you will be amazed how much resources that saves

In the end it's really about getting your hands dirty making mistakes and learning from them Don't feel bad if you end up having issues I mean we've all been there right? I was debugging a mesh for like 4 hours once it turns out I had a sign flipped somewhere in the coordinate transform matrix it was so annoying but kind of funny if you think about it now You should've seen my face

So yeah good luck with your triangle strip adventures They can be a pain but it's really rewarding once you see those beautiful surfaces rendering in front of you
