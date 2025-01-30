---
title: "How can text be placed using `cairo_pdf()`?"
date: "2025-01-30"
id: "how-can-text-be-placed-using-cairopdf"
---
The `cairo_pdf` device in R, while primarily designed for generating vector graphics within a PDF file, can effectively handle text placement using its underlying Cairo drawing context. It does not possess specialized text layout mechanisms as found in document preparation systems, therefore, direct control over text positioning, font selection, and character encoding is crucial. I've encountered this frequently while creating reports involving precise infographic elements alongside narrative text generated programmatically.

Fundamentally, `cairo_pdf()` establishes a rendering context where subsequent operations, including text drawing, occur. Text isn't added as separate elements in a document tree; instead, each character is rendered as a set of vector paths, akin to drawing a series of glyphs onto a canvas. This contrasts with, for example, HTML-based rendering where text is part of a semantic structure. The location of text is governed by the *current point* within the Cairo context and subsequent updates to the current point. This point, typically controlled by functions like `cairo_move_to()`, dictates the starting position of the next glyph string added via `cairo_show_text()`. Therefore, careful manipulation of the current point is paramount for proper text alignment.

Let's examine several code examples demonstrating this process. The initial example focuses on basic horizontal placement, showcasing how the current point shifts after rendering text.

```R
cairo_pdf("example_1.pdf", width=6, height=4)
# Set font and size
cairo_select_font_face(face = "Arial", weight="normal", slant="normal")
cairo_set_font_size(12)

# Set initial point
cairo_move_to(x=50, y=50)

# Show first string
cairo_show_text("First Line of Text.")
# The current point has now moved to the end of this string

# Add a second string
cairo_show_text("Second String.")
# Note that both strings are on the same line as default

# New line and move horizontally
cairo_move_to(x=50, y=70) # Move down
cairo_show_text("New Line!")

# Cleaning up the pdf
invisible(dev.off())
```

In this case, I've initialized the PDF context with `cairo_pdf()`. I then set the font to Arial and a font size. I move the current point to coordinates (50, 50). Upon calling `cairo_show_text()`, "First Line of Text." is rendered, and the current point advances horizontally to the end of that text's boundary. When I call `cairo_show_text()` again with "Second String.", the text is rendered immediately after the first string on the same line because no vertical move has been done. To start on a new line, I re-establish the current point via another `cairo_move_to()` call, this time at (50, 70), ensuring "New Line!" is properly placed below the initial text. This demonstrates the fundamental mechanism of horizontal and vertical text positioning.

Next, let's consider more precise positioning, integrating calculations for centering text within a specific region. While manual positioning is possible, relying on bounding box information can lead to more adaptable solutions:

```R
cairo_pdf("example_2.pdf", width=6, height=4)
# Set font and size
cairo_select_font_face(face = "Times New Roman", weight="normal", slant="normal")
cairo_set_font_size(16)

# Text to center
text <- "Centered Text"

# Function to center text within rectangle (x, y, width, height)
center_text_in_rect <- function(x, y, width, height, text) {
    extents <- cairo_text_extents(text)

    # Calculate start position
    text_x <- x + (width - extents[["width"]])/2
    text_y <- y + (height + extents[["height"]])/2 # Centering on baseline

    # Render centered text
    cairo_move_to(text_x, text_y)
    cairo_show_text(text)
}

# Draw a rectangle
cairo_set_source_rgb(0.8, 0.8, 0.8)
cairo_rectangle(50, 50, 400, 100)
cairo_fill()

# Center text inside of rectangle
center_text_in_rect(50, 50, 400, 100, text)

invisible(dev.off())
```

Here, I have defined `center_text_in_rect()` function. This encapsulates the logic for precise text centering. Crucially, `cairo_text_extents()` provides information about the text's bounding box, including width and height. I calculate the new current point based on desired rectangle to center text in and its bounding box. To demonstrate this, I also added a gray background rectangle, for visualization. This approach encapsulates the calculations within a reusable function, improving readability and maintainability. I have found this particularly helpful when positioning multiple labels within a confined space, eliminating manual trial-and-error.

Finally, let's explore rotation and more complex font manipulation:

```R
cairo_pdf("example_3.pdf", width=6, height=4)
# Set font to italic bold
cairo_select_font_face(face = "Courier New", weight="bold", slant="italic")
cairo_set_font_size(14)

# Initial point
cairo_move_to(x=100, y=100)

# Rotate text around the current point
cairo_rotate(pi/4) # Rotate 45 degrees

# Set the text and show text
cairo_show_text("Rotated Text")

#Reset rotation and other stuff
cairo_identity_matrix()
cairo_move_to(100, 200)
cairo_set_font_size(20)
cairo_select_font_face(face = "Arial", weight="bold", slant="normal")
cairo_show_text("Other Text")

invisible(dev.off())
```

In this third example, I have demonstrated two important text modifications: setting font styles and rotating text. First, I have selected a bold italic font. Then, before adding the "Rotated Text" I used `cairo_rotate()` to rotate the rendering context by 45 degrees. The effect is that the text is rendered rotated about the current point (100,100). After that, to demonstrate how it works, I reset the rotation with `cairo_identity_matrix()` and rendered another line of text with modified font and size to show a normal rendering. These examples highlight the flexibility afforded by the underlying Cairo context for more than simple horizontal text rendering.

To summarize, `cairo_pdf()` offers control over text placement in PDFs through manipulating the drawing context. Precise positioning requires using `cairo_move_to()` to manage the current point and `cairo_text_extents()` for text metrics. The combination of `cairo_show_text()`, font selection, and transformation operations such as rotation provide a versatile toolkit for rendering text elements, albeit at a low level.

For further exploration, consult the following resources:

1.  The official Cairo Graphics Library documentation. This details the various functions and rendering mechanisms employed in the library.
2.  The R Graphics Engine documentation. While not directly focused on `cairo_pdf()`, it details the interplay between different graphics devices within R.
3.  A book on vector graphics, which might also contain details about the Cairo library, especially if it focuses on general purpose graphics.
These resources will provide comprehensive details about more advanced applications and manipulations not shown here.
