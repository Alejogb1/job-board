---
title: "How do I install a new font in altair and specify it in `alt.TitleParams`?"
date: "2024-12-23"
id: "how-do-i-install-a-new-font-in-altair-and-specify-it-in-alttitleparams"
---

Alright, let's tackle this. I recall a project back in 2018, involving a rather complex dashboard visualization system. We needed a very specific typeface for consistency across client reports, and that's when I first delved into the specifics of font handling in Altair. It’s a somewhat nuanced area, but definitely manageable once you understand the underlying mechanics.

The crux of the issue isn’t actually with Altair itself; it lies more in how browsers and rendering engines handle fonts. Altair leverages Vega-Lite, which, in turn, relies on these browser-level mechanisms. Essentially, Altair doesn’t directly handle font installation; instead, it specifies the desired typeface, expecting it to be accessible by the rendering context. So, to make a custom font available in your Altair visualizations, you need to make it available to the browser that's displaying the chart.

The simplest approach, assuming you are rendering your chart within a web browser, is to use css to import the font. This means the font needs to be hosted somewhere accessible, or be available locally. If you're using a static html export, you can embed the font in that output.

Let’s break this down into actionable steps and, more importantly, illustrate with examples.

**Step 1: Make Your Font Available**

This is the most critical part. Your font file (.ttf, .otf, .woff, or .woff2) needs to be accessible from the environment where your Altair chart is rendered. If you're generating HTML files, the simplest approach is to include the font file in the same folder and use a stylesheet. This is a common practice and has some flexibility when you have complete control over the rendering environment. If you're working within a web server, ensure the font is properly served. For those using jupyter notebooks, you'll need to utilize css styles in your notebook.

**Step 2: Embed CSS for Font Specification**

The next step is to tell the browser to download and use your specified font. This is done using css. You'll need a `<style>` block with the `@font-face` rule to define the font family, where to find the font file, and potentially other attributes.

**Step 3: Use `alt.TitleParams` to reference your font**

Finally, within your Altair chart specification, you use `alt.TitleParams` (or other text-related parameters such as `alt.AxisConfig` for axis titles, etc.) and specify the font family you defined in the css. This tells Vega-Lite to use your newly loaded font family when drawing the chart elements you configured.

Let's look at an example. Let's assume we have a font file named "CustomFont.ttf" located in the same directory as our output. This isn't a practical example if you intend to render the chart online, but for static rendering or local usage, it works great.

```python
import altair as alt
import pandas as pd

# Example data
data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [6, 7, 2, 4, 5]})

chart = alt.Chart(data).mark_line().encode(
    x='x:Q',
    y='y:Q',
    tooltip=['x','y']
).properties(
    title=alt.TitleParams(
        text="Custom Font Example",
        fontSize=20,
        font='CustomFont' # referencing the font we specify with CSS
    )
)

# include the css inline with the html output to ensure the custom font is available
chart_with_css = chart.configure_view(
    strokeWidth=0,
).to_dict()

css_block = '''
<style>
    @font-face {
        font-family: 'CustomFont';
        src: url('CustomFont.ttf') format('truetype');
    }
</style>
'''
chart_with_css['$schema'] = 'https://vega.github.io/schema/vega-lite/v5.json'
chart_with_css['config']['view'] = {'stroke': None}

html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Altair Chart with Custom Font</title>
    {css_block}
</head>
<body>
    <div id="chart"></div>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <script type="text/javascript">
        const spec = {chart_with_css};
        vegaEmbed("#chart", spec).then(function(result) {
        }).catch(console.error);
    </script>
</body>
</html>
'''

with open('custom_font_chart.html', 'w') as f:
    f.write(html_content)

print("Chart written to custom_font_chart.html")

```

In this first example we are generating the full html content to ensure all font styling and code are included in a single file. It also has the advantage of showing how to embed css styling when exporting to html using `to_dict()` and the config settings.

Now, let's adapt the above to a typical jupyter notebook scenario. Instead of a separate html file we include the styling using the `display()` function from the `IPython.display` module.

```python
import altair as alt
import pandas as pd
from IPython.display import display, HTML

# Example data
data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [6, 7, 2, 4, 5]})

chart = alt.Chart(data).mark_line().encode(
    x='x:Q',
    y='y:Q',
    tooltip=['x','y']
).properties(
    title=alt.TitleParams(
        text="Custom Font Example in Jupyter",
        fontSize=20,
        font='CustomFont'
    )
)


css_block = '''
<style>
    @font-face {
        font-family: 'CustomFont';
        src: url('CustomFont.ttf') format('truetype');
    }
</style>
'''
display(HTML(css_block))
chart
```

Finally, let's imagine we are hosting the font on a different server with an address of `https://your-font-server.com/font.woff2`. We would only need to modify our css.

```python
import altair as alt
import pandas as pd
from IPython.display import display, HTML

# Example data
data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [6, 7, 2, 4, 5]})

chart = alt.Chart(data).mark_line().encode(
    x='x:Q',
    y='y:Q',
    tooltip=['x','y']
).properties(
    title=alt.TitleParams(
        text="Remote Font Example in Jupyter",
        fontSize=20,
        font='CustomFont'
    )
)


css_block = '''
<style>
    @font-face {
        font-family: 'CustomFont';
        src: url('https://your-font-server.com/font.woff2') format('woff2');
    }
</style>
'''
display(HTML(css_block))
chart
```

**Important Notes & Further Learning**

*   **Font Formats:** `.ttf` and `.otf` are common, but `.woff` and `.woff2` are usually preferred for web use, due to their better compression. Use tools like "Font Squirrel Webfont Generator" to convert fonts if needed.
*   **Font Licensing:** Ensure you are legally allowed to use the font. Some are free for personal use only, while others require commercial licenses.
*   **Browser Compatibility:**  While most modern browsers support these font loading methods, consider older browsers if you need broader compatibility and utilize font fallbacks in css.
*   **Performance:** Hosting fonts on a CDN (Content Delivery Network) is often better for performance if you are serving your charts online, as this helps to ensure quick delivery of fonts and content to users.

For a deeper dive, I would highly recommend these resources:

*   **The Vega-Lite documentation** (specifically the sections on text mark properties) will provide a better understanding of how Altair interfaces with the underlying grammar of graphics.
*   **"CSS: The Definitive Guide" by Eric A. Meyer** provides a thorough explanation of all things css and is an excellent resource when working with web content.
*   **"Type on Screen: A Critical Guide for Designers, Writers, Developers, and Students" by Ellen Lupton**, if you want to delve into the specifics of typography, this covers a lot of the nuances in displaying fonts.
*   **The Mozilla Developer Network (MDN) on CSS `@font-face`** will be an invaluable reference for working with custom fonts in browser rendering.

In summary, getting your fonts working in Altair isn't directly an Altair feature but is achieved by controlling the browser environment and its css rules. By making the font available via `<style>` tags or an external resource and then specifying that font family in your Altair title or text configurations, you’ll achieve the exact look you desire.
