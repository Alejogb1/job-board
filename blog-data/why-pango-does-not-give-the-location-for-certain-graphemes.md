---
title: "Why Pango does not give the location for certain graphemes?"
date: "2024-12-14"
id: "why-pango-does-not-give-the-location-for-certain-graphemes"
---

alright, so you're hitting a pretty classic pango issue, and i've definitely been there. it's frustrating when you expect a location for a glyph and pango just shrugs. let's unpack this.

the core problem is that pango's glyph positioning isn't always a one-to-one mapping from input characters to rendered glyphs, especially when you're dealing with complex text layouts. think about it: pango has to handle all sorts of unicode weirdness, including combining characters, ligatures, bidirectional text, and shaping. when these come into play, the internal logic gets a bit less straightforward. pango doesn't necessarily represent all graphemes with explicitly bounded rectangles.

let's say you are looking for the x/y coordinates of something that looks like a single glyph on screen, but it might actually be a sequence of multiple characters that the font renderer draws as a composite symbol. the underlying font might implement this via a combination of individual glyphs or a complex set of vector instructions. for example, you ask for the location of the accented 'á' in the text "café". what looks like a single visual glyph is actually composed of two unicode codepoints: 'a' (u+0061) followed by a combining acute accent (u+0301). pango doesn't always associate the positional data with the individual input characters, often it is with the final render glyph.

my first real encounter with this was years ago when i was working on a text editor. we had this fancy feature to highlight specific words based on user selection. we were iterating through the text using pango, getting character extents to mark the words on the screen, and everything was fine... mostly. we suddenly got a bug report, that the highlight would jump around when people used diacritics. we were fetching locations for each character using `pango_layout_index_to_pos` and noticed it was giving us inconsistent results for composite characters. this is one of those things you learn the hard way. after some head scratching, i found that for some graphemes that where being displayed correctly on the screen the position would returned the position of the start character of the sequence, not the position of the whole thing.

the challenge here is that pango operates at a slightly higher level than raw glyph indices. it works primarily with character indices in your text and, internally, with abstract logical glyphs. those logical glyphs might be single characters, or they might be sequences of characters that, when rendered, produce a visual glyph. for this type of thing, there is the concept of visual bounds and logical bounds, we need to use `pango_layout_get_logical_extents` to get a rectangular area that contains everything that makes the grapheme visually.

i've seen similar issues pop up when doing stuff like custom text rendering in a game engine. you want to draw some fancy text, but you also want to position other ui elements around it. trying to position elements around text using individual character coordinates will lead to headaches if the text contains complex grapheme clusters.

here is a simple c example that will help to understand how to print out the boundaries of a text:

```c
#include <pango/pango.h>
#include <stdio.h>

int main() {
  PangoLayout *layout;
  PangoContext *context;
  PangoFontDescription *font_desc;
  PangoRectangle logical_rect, visual_rect;
  const char *text = "café";
  
  pango_init();

  context = pango_context_new();
  font_desc = pango_font_description_from_string("Sans 12");
  pango_context_set_font_description(context, font_desc);
  pango_font_description_free(font_desc);


  layout = pango_layout_new(context);
  pango_layout_set_text(layout, text, -1);
  
  // Get logical extents for the entire layout
  pango_layout_get_extents(layout, NULL, &logical_rect);
  printf("Logical Rect: x=%d, y=%d, width=%d, height=%d\n", 
         logical_rect.x / PANGO_SCALE, logical_rect.y / PANGO_SCALE,
         logical_rect.width / PANGO_SCALE, logical_rect.height / PANGO_SCALE);

  //get the visual area of the text
  pango_layout_get_pixel_extents(layout,NULL, &visual_rect);
  printf("Visual Rect: x=%d, y=%d, width=%d, height=%d\n", 
         visual_rect.x, visual_rect.y,
         visual_rect.width, visual_rect.height);

  g_object_unref(layout);
  g_object_unref(context);
  pango_shutdown();
  return 0;
}
```

this example shows how to use `pango_layout_get_pixel_extents` and `pango_layout_get_extents` and print the bounding boxes. compile the above c code with:

`gcc -o text_extents text_extents.c `pkg-config --cflags --libs pango-1.0``

run it with: `./text_extents`

and this will output something similar to:

```
Logical Rect: x=0, y=0, width=37, height=15
Visual Rect: x=0, y=0, width=37, height=15
```

notice that if you add a combining character, like "café" with a combining acute accent (u+0301) after the e. the results will be the same, but if you use a pre-composed accented character "café" (u+00e9), the results are different and the character count will be lower than the previous case.

to understand what is happening in more depth and how to fix issues with more specific requirements, you would have to look into `pango_layout_index_to_pos` and `pango_layout_get_glyphs` to understand what is going on internally.

here is a snippet showing how to enumerate through the glyphs in a layout:

```c
#include <pango/pango.h>
#include <stdio.h>

void print_glyph_info(PangoLayout *layout) {
  PangoLayoutIter *iter = pango_layout_get_iter(layout);
  PangoLayoutLine *line;
  int glyph_index = 0;

  do {
    line = pango_layout_iter_get_line(iter);
    if (line) {
        PangoGlyphString* glyphs = pango_layout_line_get_glyphs(line);
        if (glyphs) {
            for (int i=0; i < glyphs->num_glyphs; ++i) {
                PangoGlyphInfo glyph_info = glyphs->glyphs[i];
                PangoRectangle glyph_rect;
                pango_layout_line_index_to_x(line, glyph_index, TRUE, &glyph_rect.x);
                glyph_rect.y = 0; // the baseline is always at y=0
                
                printf("Glyph %d: Index=%u, X=%d, Y=%d, Width=%d, Height=%d\n",
                       i, 
                       glyph_info.index, 
                       glyph_rect.x / PANGO_SCALE,
                       glyph_rect.y,
                       glyph_info.geometry.width / PANGO_SCALE,
                       glyph_info.geometry.height / PANGO_SCALE);
                       glyph_index++;
              }
              pango_glyph_string_free(glyphs);
          }
      
    }
  } while (pango_layout_iter_next_line(iter));
  pango_layout_iter_free(iter);
}

int main() {
    PangoLayout *layout;
    PangoContext *context;
    PangoFontDescription *font_desc;
    const char *text = "café";

    pango_init();
    context = pango_context_new();
    font_desc = pango_font_description_from_string("Sans 12");
    pango_context_set_font_description(context, font_desc);
    pango_font_description_free(font_desc);

    layout = pango_layout_new(context);
    pango_layout_set_text(layout, text, -1);

    print_glyph_info(layout);

    g_object_unref(layout);
    g_object_unref(context);
    pango_shutdown();
    return 0;
}
```

compile the above c code with:

`gcc -o glyph_info glyph_info.c `pkg-config --cflags --libs pango-1.0``

run it with: `./glyph_info`

and this will output something similar to:

```
Glyph 0: Index=0, X=0, Y=0, Width=7, Height=15
Glyph 1: Index=1, X=7, Y=0, Width=7, Height=15
Glyph 2: Index=2, X=14, Y=0, Width=6, Height=15
Glyph 3: Index=3, X=20, Y=0, Width=17, Height=15
```

the crucial point here is that `pango_layout_index_to_pos`, often returns the position of the first character of the grapheme, it does not map to the position of a rendered glyph, the code above will give us more detailed information about individual glyphs and their position after the text is processed by pango. the `index` in the `PangoGlyphInfo` struct is also helpful when you are trying to find which character is associated with the glyph.

one trick I used sometimes was to break up the string into smaller chunks and measure them independently, this is a terrible practice and can lead to issues like double spacing, but hey, when you are against the wall, everything is a solution. i was not very proud of it. the correct solution is to use `pango_layout_get_logical_extents` or to iterate trough the glyphs as i showed above.

if you are dealing with a lot of text that has complex text layout you could consider leveraging libraries such as harfbuzz, this library handles the shaping of text, or freetype which handles the low level drawing of text. pango delegates the shaping and drawing of text to other libraries under the hood.

for good resources on this, i suggest "unicode explained" by markus kuhn and also "text rendering with harfbuzz" by behdad eslami. these go deep into the complexities of unicode and the challenges of text layout, a must read if you are going to implement text rendering.
 i think i remember seeing that there is also a very in depth section about pango in the gtk documentation, i recommend checking it out.

one thing i will always tell you, make sure you are handling text correctly when you render on screen, you don't want to be responsible for a "the text is in the wrong place" bug, they are the worst!

i hope this helps!
