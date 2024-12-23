---
title: "latex crop image borders?"
date: "2024-12-13"
id: "latex-crop-image-borders"
---

 so you wanna crop image borders in LaTeX right Been there done that got the t-shirt multiple times honestly this is like a rite of passage for anyone who spends enough time with LaTeX and images I've battled this beast more often than I'd like to admit let's dive in

First off you’re probably dealing with a situation where your images got these annoying white borders or even worse some colored padding you don’t want right? Maybe they’re screenshots with some extra fluff or diagrams that have too much margin Whatever it is it's cramping your style and LaTeX is not automatically helping you out like it should I remember once I had this huge project due a journal publication and I had all these embedded plots from MATLAB yeah MATLAB and they all had this extra border that looked atrocious it was a nightmare i almost had a breakdown

So the short answer yeah you can totally crop images in LaTeX But it's not like a one-size-fits-all magic command It involves a few approaches depending on what exactly you want to do and what your workflow is like I’ve found using `\includegraphics` with specific options is usually the quickest route for simple cropping.

Let’s break it down starting with the basics. You're most likely using the `graphicx` package if you’re embedding images at all If not go add `\usepackage{graphicx}` to your preamble

The easiest way to crop is using the `trim` and `clip` options of `\includegraphics` The `trim` option takes four arguments usually in `l b r t` order which is left bottom right top and they define the distance in points to remove from each side of the image The `clip` option then ensures the result of trimming is actually what is shown You can try something like this

```latex
\documentclass{article}
\usepackage{graphicx}

\begin{document}

\includegraphics[trim=10pt 10pt 10pt 10pt, clip]{your-image.jpg}

\end{document}
```

Replace `your-image.jpg` with the name of your actual image This snippet will remove 10 points from each side That’s assuming your unit is `pt` you can use `cm` `mm` `in` and other valid length units if you want The points thing is really just a LaTeX quirk it’s a printer standard thingy

Now here's the rub the `trim` argument is a pain in the butt for one you have to eyeball the values a little bit unless you know the exact pixel/point dimensions of your image's borders it’s tedious I’ve wasted hours fiddling with these numbers and it was even worse when I started doing things that needed to be pixel-precise

Then you also have the `viewport` option which is less about removal and more about viewing It’s similar to crop but you are defining what part of the image to show instead of cutting the borders If you have an image bigger than you want this can be helpful too The arguments are again in `l b r t` order and now they mean the coordinates inside your image not the amount to trim The values refer to how many pixels/points/units from the bottom-left of the image you are going to start and to end the viewing box The unit is again assumed to be point but any valid length unit works.

```latex
\documentclass{article}
\usepackage{graphicx}
\begin{document}

\includegraphics[viewport=50 50 200 200, clip, width=5cm]{your-image.png}

\end{document}
```

This will display only the part of the image that starts 50 points from left and 50 points from bottom and ends 200 points from left and 200 points from bottom If you want an specific size of this cropped image it would be like in the example with the width parameter. Remember that the `viewport` coordinates refer to the image coordinates not your document or bounding box.

Now sometimes just using `trim` and `viewport` directly in your LaTeX document can be less flexible especially if you’re generating plots programmatically and the borders are not consistent or you have a large number of images to manage This is where external tools and preprocessing become important I had an issue where a bunch of images from a simulator had these randomly generated borders depending on the scene complexity it was a mess the solution was to batch crop these externally using `imagemagick`

Imagemagick is a command-line image processing tool that’s like a swiss army knife for images It’s available on most platforms and it can do a lot of things including cropping Let's imagine we want to remove 20 pixels all around the image using `imagemagick` from command line you will use something like this

```bash
convert input.png -shave 20x20 output.png
```

This command shaves 20 pixels from every side You can control horizontal and vertical removal separately using something like `convert input.png -shave 20x10 output.png` where you'll be removing 20 pixels from left and right and 10 pixels from top and bottom Now you just include the `output.png` image in your latex file instead of the `input.png` image Now that I think about it doing the same thing in python or other scripting language makes even more sense for automation purposes.

There is also PDF crop tools that you can use for PDF images I’m not that used to those but some that I’ve heard of are `pdfcrop` which is another command line tool and also some online PDF crop tools they have varying levels of quality some do better with text some with images and some are just a disaster.

A word of caution when dealing with vector images especially `PDF` images if you are too aggressive with your cropping you might mess with the bounding box information and LaTeX can get a little confused rendering the image So try to avoid chopping too much of the space around your image you can easily get overlapping text for example

So to sum up you got different ways to crop your images in LaTeX You can use `trim` or `viewport` with `\includegraphics` but if the borders are complex or you have many images using tools like ImageMagick or some programming language for scripting it usually makes more sense in terms of time and ease of use.

Also remember that cropping is not something that you usually need to do it is a last resort type of thing If you are generating images for your LaTeX document try to generate them already with no borders instead of fixing the borders when doing the latex document this is really not a great time to debug things. Like they say "An ounce of prevention is worth a pound of cure" its a very old saying but it is true also for latex border issues!

If you really want to delve deep into this area I recommend looking into:

*   **The LaTeX Companion by Goossens et al:** It has an extensive section on graphics and image manipulation which can be really helpful. It is not something that changes fast in LaTeX so older versions are still valid for this sort of thing.
*   **The Graphics Companion by Michel Goossens:** This is a separate book focused entirely on graphics within LaTeX. It provides a much deeper understanding of how LaTeX handles images and its options.
*   **ImageMagick documentation:** For external image processing refer to the official documentation of ImageMagick it's a lot of material but very well organized
*   **Some scripting language documentation related to the library/module that you will use:** For example if you chose python the `pillow` or the `opencv` libraries are a good start.

These are my go-to resources whenever I need to work with image manipulation in LaTeX. And remember a lot of the time when you are doing things programmatically you don't need `latex` and `dvips` anymore for example you can do everything with the `xelatex` engine and some other small details it really depends on your workflow. Anyway good luck with your image cropping battles you will get it eventually it might just take a couple of iterations.
