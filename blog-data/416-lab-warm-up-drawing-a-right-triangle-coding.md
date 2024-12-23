---
title: "4.16 lab warm up drawing a right triangle coding?"
date: "2024-12-13"
id: "416-lab-warm-up-drawing-a-right-triangle-coding"
---

so you're looking to draw a right triangle using code right been there done that countless times. I remember back in college when we had to do this for graphics class man that was a trip we were using like really low level stuff then and now we got these fancy libraries. We're not talking about complex 3D rendering or anything just your basic right triangle. This thing is fundamentally basic like a "Hello World" of geometric programming. So I get your question.

 first thing first we need to figure out what you want to use for the rendering right some people use canvas others want terminal outputs others are in game engines. So let's go through a couple of options here.

Let's assume you’re going for a simple textual representation which is like the og way to do graphics so here’s how you do it with basic characters in your standard terminal output using python because python is always the go to for this stuff.

```python
def draw_right_triangle(height):
    for i in range(1, height + 1):
        print("*" * i)

draw_right_triangle(5)
```

so what’s happening there? We’re looping through rows from one up to the height you specify that’s the number of rows and for each row we’re printing `i` amount of `*` characters. `i` is basically the length of the horizontal side each iteration. This is very basic but gives a very readable output in your standard text-based terminal it’s good enough for lab warm up exercises.

Now if you wanna go a little fancier maybe you wanna use an actual library like pygame well if that’s your case here's a way to do it in pygame I'm gonna assume that you have pygame set up so not gonna go through that whole step if you don’t just do pip install pygame

```python
import pygame
pygame.init()

width = 600
height = 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Right Triangle")

white = (255, 255, 255)
black = (0, 0, 0)

screen.fill(white)

point1 = (100, 300)
point2 = (100, 100)
point3 = (300, 300)

pygame.draw.polygon(screen, black, [point1, point2, point3])

pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
```

So what is going on here? First we initiate pygame then we set up a window basically a canvas for drawing. Then we define some colors like white and black not that fancy we fill the screen white and now here's the important part I defined three points that make up our right triangle using the coordinates. Then we draw a polygon that's pygame's method to draw a shape like our triangle. The polygon takes the screen color and a list of those points.

Then I display the drawing with `pygame.display.flip()` and initiate the game loop which allows the window to stay open until you manually close it. It’s basically a very standard way to draw any shape in pygame. It’s pretty useful when you want more control over visual output like colors and positioning.

 let’s look at one more example because why not suppose you want to do this in JavaScript inside a basic HTML canvas. We are doing the full setup here this time so let me drop it here.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Right Triangle Canvas</title>
</head>
<body>
    <canvas id="myCanvas" width="400" height="300"></canvas>
    <script>
        var canvas = document.getElementById("myCanvas");
        var ctx = canvas.getContext("2d");

        ctx.beginPath();
        ctx.moveTo(50, 250);
        ctx.lineTo(50, 50);
        ctx.lineTo(250, 250);
        ctx.closePath();

        ctx.stroke();
    </script>
</body>
</html>
```

This html file has a basic canvas element with the id `myCanvas` using JavaScript we retrieve the canvas element then we get the 2d drawing context using `getContext("2d")`. Then we use these `moveTo` and `lineTo` methods to draw lines and we finish the path to create the shape and finally `ctx.stroke` draws the triangle path we created onto the canvas. This is good if you wanna embed your visual in a webpage like in a project for a web presentation.

Each of these methods gives a different angle and different use cases.

Now let’s talk about the tricky parts and the things you might run into.

One thing I’ve seen students mess up a lot is the coordinate system. Different systems use different origins like the canvas typically starts in the top left where (0,0) is up there instead of the center of the screen. It's the opposite of that coordinate systems that we learned in school. Make sure you know where your (0,0) point is especially in a library like pygame that can be really confusing.

Another common problem is that people often miscalculate the coordinates. Especially when you wanna do complex triangles or when you wanna move stuff around you gotta get your math right and that means understanding basic geometry you can’t just wing it. I swear there’s no magic it's just basic geometry.

Also a lot of people struggle with setting up their environment. Getting the right packages installed can be a problem but I assume you can google it up easily and the basic pygame and javascript canvas stuff will set up itself relatively easier than any openGL library setup process. It's not rocket science.

You also need to keep in mind the performance of your drawing method. For example drawing a complex shape with tons of tiny lines can be really slow and inefficient when you have high number of polygons. So that is why optimizing your draw method is very important. But at the scale of a basic right triangle I really doubt that it will be an issue.

Also consider that some graphical libraries have different API design choices which means that the actual implementation of your right triangle can differ but overall the general principle is the same just three points and connection. This is basically fundamental.

Speaking of math stuff I did have this issue once in a project where i had to calculate lots of these triangles for a game and I kept mixing up my math. I thought it would be simple so I didn't bother to recheck my math but after spending 2 days debugging it turned out it was something really simple like I was inverting one axis or something so do yourself a favour and check your calculations and your coordinate system one more time. Also the whole environment setup is always a pain like it always is. It’s like someone put spaghetti code in my own local computer and just said figure it out yourself. (Here's my attempt at a joke)

As for resources I’d recommend some books over website or documentation which can be overwhelming. I’d go with “Computer Graphics Principles and Practice” for a comprehensive theory behind all this stuff or “Real-Time Rendering” if you’re gonna get serious with graphics I know it's overkill for a single right triangle but good to have these books. These books will give you an insight on what is happening behind the scenes. They contain all the theory and practice needed for any type of 2d or 3d rendering.

For more basic stuff a book like “Python Crash Course” has good info for pygame stuff and for web stuff “Eloquent JavaScript” can help you on the canvas related things they go over a lot of these basic visual drawing steps in a simple fashion.

So there you have it three ways to draw a right triangle using code and some of the things you need to be aware of. It’s pretty basic stuff when you think about it so have fun coding. And if you have more questions just drop them here in comments.
