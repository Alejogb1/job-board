---
title: "Why does Ghostscript replace font names with 'CairoFont'?"
date: "2025-01-30"
id: "why-does-ghostscript-replace-font-names-with-cairofont"
---
Ghostscript's substitution of font names with "CairoFont" typically arises from a combination of font handling limitations and the interaction between PostScript interpreters and the underlying graphics library used for rendering. Specifically, it points to situations where Ghostscript relies on its Cairo graphics backend, rather than direct PostScript font processing, often when dealing with fonts that are either non-standard, poorly defined, or for which it lacks readily available system font mappings.

Having spent several years troubleshooting document processing pipelines involving various PostScript and PDF inputs, I've frequently encountered this "CairoFont" phenomenon, particularly when the source document was generated outside of a controlled publishing environment. It’s rarely a problem with well-formed, common fonts; instead, it usually indicates that Ghostscript was presented with font information it could not directly interpret using its built-in font handling mechanisms. The font might lack a precise PostScript name, rely on encoding schemes not recognized by Ghostscript's core font handling, or perhaps be an embedded font whose data is malformed, corrupt, or simply not readable.

The core of the problem resides in how Ghostscript renders graphics. It supports multiple "devices," or backends, for generating output: printers, raster images, or vector formats. When a font cannot be directly processed as a PostScript font, Ghostscript often falls back to its Cairo-based rendering system. Cairo is a 2D graphics library designed to handle general drawing operations, including rendering text using its own font handling mechanisms. Consequently, instead of attempting to interpret and map the potentially problematic font, Ghostscript effectively rasterizes text using Cairo's best-effort interpretation of the font’s metrics, then treats the resulting character glyphs as graphical primitives, thus losing the original font's identity.

This process means the original font name is discarded; when such a document is later processed (for instance, re-opening the PDF or converting it to another format), the font is indicated as "CairoFont" because that is the only descriptor available. Any additional information or metrics specific to the original font, such as kerning or hinting instructions, are likely discarded as well. This fallback has the merit of allowing Ghostscript to render the text, but it sacrifices fidelity and creates challenges for downstream processes that rely on precise font identification.

Now, let’s look at concrete examples.

**Example 1: A PostScript file using a font not available to Ghostscript.**

Imagine a PostScript file (test1.ps) contains instructions using a custom font, "MyCustomFont," which isn't installed on the system where Ghostscript is running.

```PostScript
%!PS-Adobe-3.0
/MyCustomFont findfont 12 scalefont setfont
100 100 moveto (Hello, world!) show
showpage
```
When this file is processed via `gs -o output1.pdf test1.ps`, the resulting PDF will likely show all the text rendered using "CairoFont", rather than attempt to map or load the missing "MyCustomFont" which was not in its known font dictionaries.

The reason is that without proper font information (or a font mapping), Ghostscript will fallback to rendering each character glyph separately using its default metrics. The resulting PDF is a collection of vector glyphs representing the characters using Cairo's rendering capabilities and not specific font data. Subsequent tools that inspect the PDF will show the rendered characters with the font name “CairoFont” and no further font metadata is available.

**Example 2: Embedded Font with Invalid Subsections.**

Suppose a document contains an embedded font (encoded in a Type 1 format in a PDF) that has been corrupted or improperly structured. Let’s say this PDF is named ‘corrupt_font.pdf’ containing an embedded font titled "BrokenFont".

```bash
gs -o output2.pdf corrupt_font.pdf
```

In this case, even though the font is embedded, Ghostscript may encounter issues while parsing the Type 1 data, thus rejecting it and falling back to its internal rasterization methods via the Cairo backend. The result is that output2.pdf will show the text rendered with a "CairoFont" designation and not "BrokenFont." The specific problem here is not the lack of a mapping, but an inability to interpret a font file; thus, Cairo becomes the rendering fallback. Examining the PDF output shows that the text elements lack the usual font description, as the font data could not be successfully extracted or parsed by Ghostscript.

**Example 3: Poorly Formatted PDF with Uncommon Font Encoding**

Consider a PDF file (input3.pdf) that uses a rarely encountered font encoding scheme and a custom font. This encoding might use a specific character mapping that Ghostscript's base interpreters doesn’t recognize.

```bash
gs -o output3.pdf input3.pdf
```

If Ghostscript doesn’t recognize the character mapping (specifically, if the encoding is not one it knows how to translate), or the font doesn't specify a usable encoding vector, it may fall back to Cairo. The encoding problem leads to Ghostscript being unable to interpret the intended character sequences from the encoded font data, which pushes it to bypass normal font handling. Similar to the previous examples, the text in `output3.pdf` will show "CairoFont" in place of the intended font, and all font information will be lost, as the original character sequences will have been processed as glyph primitives by Cairo.

In such situations, simply installing a font on the system may not be sufficient. The problem often lies with the document itself, its font definition or encoding scheme. The issue is how the document describes a font, not necessarily just the presence or absence of a font file.

When encountering “CairoFont” substitution, several steps can help diagnose the root cause:

1.  **Font Inspection:** Tools like `pdffonts` (from the Poppler utilities) can help examine the embedded fonts used by the PDF. This can highlight missing or problematic fonts, which often triggers this substitution. It can also reveal encoding issues and other characteristics of font usage.

2. **Ghostscript Debugging:**  Ghostscript supports detailed logging. By using verbose flags like `-dPDFDEBUG` or `-dDEBUG` it’s possible to get detailed diagnostics during PDF processing. This may reveal specific errors in font handling and indicate where exactly the fallback is triggered. The output will also expose which font and font encoding Ghostscript struggled with.

3.  **Font Substitution:** Ghostscript provides a font substitution mechanism via the `-sFONTPATH=` argument to provide explicit font locations that Ghostscript may use to find available font files on the machine. Although this might not be enough if the original font has an unsupported encoding or the document itself is faulty, it may allow it to find a font on the system rather than falling back to the rasterization of Cairo. This technique may help when the issue is an absent or missing font.

4. **PDF Preflight:** Various tools can validate and preflight PDF files based on specifications like PDF/X. These tools can identify issues in PDF files (such as corrupted or poorly defined fonts) that cause processing issues. Running such preflight checks can help identify potential problem areas before attempting Ghostscript processing.

5. **Original Document Inspection:** Whenever possible, examining the document generation process is valuable. Knowing how the document was created might reveal issues. The application used to create the PDF or Postscript file may be faulty in terms of generating valid fonts.

In summary, the "CairoFont" substitution by Ghostscript signifies its inability to interpret fonts using its standard font-handling methods. It often indicates missing fonts, corrupted fonts, or problematic encoding schemes. Careful inspection using the recommended approaches above can help diagnose and resolve these issues and prevent unnecessary loss of font fidelity in downstream processes. The problem rarely resides in Ghostscript itself, but rather in the manner the document encodes and defines fonts.
