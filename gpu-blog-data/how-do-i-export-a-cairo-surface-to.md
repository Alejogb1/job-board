---
title: "How do I export a Cairo surface to PNG and output it to standard output?"
date: "2025-01-30"
id: "how-do-i-export-a-cairo-surface-to"
---
The efficient transfer of a Cairo drawing surface as a PNG image directly to standard output requires careful handling of memory buffers and stream redirection within the Cairo graphics library. Unlike file-based output, this method avoids the creation of temporary files, a crucial consideration in memory-constrained or pipeline-oriented applications.

My experience in embedded systems development, particularly rendering directly to framebuffer devices, has highlighted the importance of this technique. In those cases, I often needed to transform processed graphics into a stream digestible by other applications without persistent storage. Using the `cairo_image_surface_write_to_png_stream` function achieves this goal in a highly efficient manner.

The fundamental process involves three key steps: surface creation and drawing, encoding the surface as a PNG image, and finally, directing the encoded data to standard output.

First, a Cairo surface must be created, representing the drawing area. I would usually select `CAIRO_FORMAT_ARGB32` for broad compatibility with various pixel formats. Once the surface is initialized, all desired drawing operations are performed—geometric shapes, text, or image compositions. This surface represents the raw pixel data ready for encoding.

Next, the core of this process is the `cairo_image_surface_write_to_png_stream` function. Unlike the typical file-based `cairo_surface_write_to_png`, this function accepts a callback that takes a buffer and size as arguments. This callback handles writing the PNG data to a user-defined stream. Crucially, for standard output, we use a simple function that writes to `stdout` using the system's `fwrite`. This allows Cairo to efficiently push the encoded PNG data directly to standard output as it is generated, eliminating the need to store the entire PNG image in memory.

Finally, after successfully rendering to the surface and writing the PNG to the stream, the surface, context, and any related resources should be cleaned up with calls to `cairo_surface_destroy` and `cairo_destroy`, respectively. Failure to do this leads to resource leaks.

The crucial benefit of this approach is minimizing memory usage and avoiding temporary files. Direct stream redirection is a core building block in building robust command-line tools or when integrated into pipelines.

Here are a few code examples to illustrate various scenarios. These examples are C-based, the core language for Cairo, and assume you have a working Cairo installation.

**Example 1: Basic Solid Color PNG**

```c
#include <cairo.h>
#include <stdio.h>
#include <stdlib.h>

static cairo_status_t write_to_stdout(void *closure, const unsigned char *data, unsigned int length) {
    (void)closure; // unused parameter
    if (fwrite(data, 1, length, stdout) != length) {
      return CAIRO_STATUS_WRITE_ERROR;
    }
    return CAIRO_STATUS_SUCCESS;
}

int main() {
    cairo_surface_t *surface;
    cairo_t *cr;

    // Create a 200x200 ARGB32 surface
    surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 200, 200);
    if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
       fprintf(stderr, "Error creating surface: %s\n", cairo_status_to_string(cairo_surface_status(surface)));
       return 1;
    }

    cr = cairo_create(surface);
    if (cairo_status(cr) != CAIRO_STATUS_SUCCESS) {
        cairo_surface_destroy(surface);
        fprintf(stderr, "Error creating Cairo context: %s\n", cairo_status_to_string(cairo_status(cr)));
        return 1;
    }

    // Set background color to blue
    cairo_set_source_rgb(cr, 0.0, 0.0, 1.0);
    cairo_paint(cr);

    // Write surface to stdout as PNG
    if(cairo_surface_write_to_png_stream(surface, write_to_stdout, NULL) != CAIRO_STATUS_SUCCESS){
         fprintf(stderr, "Error writing to PNG stream: %s\n", cairo_status_to_string(cairo_surface_status(surface)));
         cairo_destroy(cr);
         cairo_surface_destroy(surface);
         return 1;
    }

    // Cleanup resources
    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    return 0;
}
```

This first example demonstrates the core mechanics of rendering a simple blue rectangle and sending it to `stdout`. The `write_to_stdout` callback is critical here – it's the intermediary that pipes Cairo data to the standard output stream. It first ensures that the correct number of bytes has been successfully written, and it returns an error code if not.

**Example 2: Drawing a Circle with a Border**

```c
#include <cairo.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static cairo_status_t write_to_stdout(void *closure, const unsigned char *data, unsigned int length) {
    (void)closure; // unused parameter
    if (fwrite(data, 1, length, stdout) != length) {
      return CAIRO_STATUS_WRITE_ERROR;
    }
    return CAIRO_STATUS_SUCCESS;
}

int main() {
    cairo_surface_t *surface;
    cairo_t *cr;

    // Create a 200x200 ARGB32 surface
    surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 200, 200);
      if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
       fprintf(stderr, "Error creating surface: %s\n", cairo_status_to_string(cairo_surface_status(surface)));
       return 1;
    }
    cr = cairo_create(surface);
        if (cairo_status(cr) != CAIRO_STATUS_SUCCESS) {
        cairo_surface_destroy(surface);
        fprintf(stderr, "Error creating Cairo context: %s\n", cairo_status_to_string(cairo_status(cr)));
        return 1;
    }
    // Background color
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_paint(cr);

    // Draw a circle
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_arc(cr, 100, 100, 50, 0, 2 * M_PI);
    cairo_fill_preserve(cr); // Fills the shape but retains the path for the border

    // Set line border color and width
    cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
    cairo_set_line_width(cr, 5.0);
    cairo_stroke(cr);


    if(cairo_surface_write_to_png_stream(surface, write_to_stdout, NULL) != CAIRO_STATUS_SUCCESS){
         fprintf(stderr, "Error writing to PNG stream: %s\n", cairo_status_to_string(cairo_surface_status(surface)));
         cairo_destroy(cr);
         cairo_surface_destroy(surface);
         return 1;
    }

    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    return 0;
}
```

This second example illustrates a more complex drawing scenario – a filled circle with a red border. It introduces functions like `cairo_arc`, `cairo_fill_preserve`, and `cairo_stroke` to create more intricate graphic primitives. The crucial part, similar to the first example, remains the use of `cairo_surface_write_to_png_stream` for output redirection.

**Example 3: Simple Text Output**

```c
#include <cairo.h>
#include <stdio.h>
#include <stdlib.h>

static cairo_status_t write_to_stdout(void *closure, const unsigned char *data, unsigned int length) {
    (void)closure;
    if (fwrite(data, 1, length, stdout) != length) {
      return CAIRO_STATUS_WRITE_ERROR;
    }
    return CAIRO_STATUS_SUCCESS;
}

int main() {
    cairo_surface_t *surface;
    cairo_t *cr;

    // Create a 200x100 ARGB32 surface
    surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 200, 100);
    if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
       fprintf(stderr, "Error creating surface: %s\n", cairo_status_to_string(cairo_surface_status(surface)));
       return 1;
    }
    cr = cairo_create(surface);
    if (cairo_status(cr) != CAIRO_STATUS_SUCCESS) {
        cairo_surface_destroy(surface);
        fprintf(stderr, "Error creating Cairo context: %s\n", cairo_status_to_string(cairo_status(cr)));
        return 1;
    }


    // Set background color to white
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_paint(cr);

    // Set text color to black
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_select_font_face(cr, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 24.0);
    cairo_move_to(cr, 20, 60);
    cairo_show_text(cr, "Hello, stdout!");


    if(cairo_surface_write_to_png_stream(surface, write_to_stdout, NULL) != CAIRO_STATUS_SUCCESS){
         fprintf(stderr, "Error writing to PNG stream: %s\n", cairo_status_to_string(cairo_surface_status(surface)));
         cairo_destroy(cr);
         cairo_surface_destroy(surface);
         return 1;
    }

    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    return 0;
}
```

This third example illustrates how to draw text. Here we configure the font face, size, and position. It is equally important to correctly configure the font size, the position of the text, and its color. Again the use of the stream writer is demonstrated effectively.

For further exploration of Cairo, the official documentation, available on the Cairo website, should be consulted. Additionally, the book “The Cairo Graphics Library” by Carl Worth provides an in-depth look at the intricacies of the library. Exploring the available Cairo examples, often distributed with the library, is also highly recommended, as they demonstrate many different use cases. These resources collectively provide the necessary information for mastering Cairo. By avoiding temporary files and using the stream interface, Cairo becomes a powerful tool for real-time or pipelined image processing.
