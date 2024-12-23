---
title: "How can I implement a magnifying glass effect in a Streamlit app using Python?"
date: "2024-12-23"
id: "how-can-i-implement-a-magnifying-glass-effect-in-a-streamlit-app-using-python"
---

Alright, let's talk magnifying glasses in Streamlit. I remember back on a project—it was a geographic data visualization tool, funnily enough—we needed precisely that: a way for users to inspect high-resolution map segments without cluttering the main view. It’s a deceptively complex effect if you consider all the moving parts. Rather than a simplistic zoom, a true magnifying glass needs to follow the cursor, show a magnified region, and update smoothly. We’re essentially working with interactive, dynamic masking and transformation here.

The core principle revolves around combining a few key elements: the original image (or, in our streamlit context, an image rendered from something), an overlay that acts as the ‘glass’ and dynamically updates its content based on cursor position, and some clever manipulation of coordinates to get the magnified region correctly displayed.

First, let's consider the mechanics. You won't be finding a native 'magnifying glass' widget in Streamlit; that's where our custom implementation comes in. We’ll utilize a canvas element, handled through Streamlit components, and draw the magnified region onto it. The interaction, naturally, will be through user input – the mouse movements on the main image, interpreted to update the canvas. Think of it as a localized, dynamic crop with a zoom effect applied.

Here’s how it can be put into practice using a simple example. We'll start with basic image handling and then build out the interactive canvas.

```python
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import streamlit.components.v1 as components
import base64

def create_magnifying_glass_component(image_path, zoom_factor=2):
    """
    Creates a streamlit component for a magnifying glass effect.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    component_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        #container {{
        position: relative;
        }}
        #magnified {{
            position: absolute;
            border: 2px solid black;
            border-radius: 50%;
            pointer-events: none;
            display: none;
        }}

    </style>
    </head>
    <body>
    <div id="container">
        <img id="main-image" src="data:image/png;base64,{encoded_string}" style="max-width: 600px; cursor: crosshair;" >
        <canvas id="magnified" width="200" height="200"></canvas>
    </div>

    <script>
    const image = document.getElementById('main-image');
    const canvas = document.getElementById('magnified');
    const ctx = canvas.getContext('2d');
    const zoom = {zoom_factor};
    let mainImageRect = image.getBoundingClientRect();

    function updateMagnifier(x, y) {{
        let canvasSize = canvas.width;
        const magnifiedX = x * zoom - (canvasSize/2) ;
        const magnifiedY = y * zoom - (canvasSize/2);

        let cropX = x - (canvasSize / (zoom * 2));
        let cropY = y - (canvasSize / (zoom * 2));
        if (cropX < 0) cropX = 0;
        if (cropY < 0) cropY = 0;
        if (cropX + (canvasSize / zoom ) > image.naturalWidth) cropX = image.naturalWidth - (canvasSize/ zoom );
        if (cropY + (canvasSize / zoom )> image.naturalHeight) cropY = image.naturalHeight - (canvasSize/ zoom );

        ctx.clearRect(0, 0, canvasSize, canvasSize);
        ctx.drawImage(image, cropX, cropY, canvasSize/zoom, canvasSize/zoom, 0, 0, canvasSize, canvasSize);
        canvas.style.left = (x + mainImageRect.left - canvasSize/2) + 'px';
        canvas.style.top = (y + mainImageRect.top - canvasSize/2) + 'px';
        canvas.style.display = 'block';

    }}

    image.addEventListener('mousemove', function(event) {{
        const x = event.clientX - mainImageRect.left;
        const y = event.clientY - mainImageRect.top;

        updateMagnifier(x,y);
    }});


    image.addEventListener('mouseleave', function() {{
         canvas.style.display = 'none';
    }});

    </script>
    </body>
    </html>
    """

    html_component = components.html(component_code, height=650)
    return html_component

if __name__ == '__main__':
    st.title("Magnifying Glass Example")
    image_path = "sample.png" # Replace with your desired image path
    # Generate a sample image if one isn't available
    try:
        Image.open(image_path)
    except FileNotFoundError:
        image = Image.new('RGB', (500, 500), color = 'white')
        draw = ImageDraw.Draw(image)
        draw.ellipse((100, 100, 400, 400), fill = 'blue', outline ='black')
        image.save(image_path)

    create_magnifying_glass_component(image_path, zoom_factor=3)
```

This snippet sets up the fundamental HTML structure with an image and a canvas, which serves as the magnifying glass. It contains inline styling and javascript, which is why embedding it within a string is necessary. A `mousemove` event listener updates the canvas content based on mouse location. The `create_magnifying_glass_component` function handles the html, encoding the given image, and setting the javascript variables, and returning the streamlit component.

Crucially, we use `base64` to encode our image data, so it can be passed directly into the HTML's `src` attribute. I often see folks stumbling over this, assuming they can just pass a file path. Streamlit components operate on a client-side rendering basis, so we’re dealing with how the browser interprets the data, not file paths local to the server.

Next, let's expand on this to include different shapes for our magnifying glass and handle dynamic resizing.

```python
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import streamlit.components.v1 as components
import base64

def create_shaped_magnifying_glass(image_path, zoom_factor=2, shape='circle', glass_size=200):
    """
        Creates a streamlit component for a magnifying glass effect with customizable shape
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    component_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        #container {{
        position: relative;
        }}
        #magnified {{
            position: absolute;
            border: 2px solid black;
            border-radius: { '50%' if shape == 'circle' else '0px'};
            pointer-events: none;
            display: none;
        }}

    </style>
    </head>
    <body>
    <div id="container">
        <img id="main-image" src="data:image/png;base64,{encoded_string}" style="max-width: 600px; cursor: crosshair;" >
        <canvas id="magnified" width="{glass_size}" height="{glass_size}"></canvas>
    </div>

    <script>
    const image = document.getElementById('main-image');
    const canvas = document.getElementById('magnified');
    const ctx = canvas.getContext('2d');
    const zoom = {zoom_factor};
    const canvasSize = {glass_size};
    let mainImageRect = image.getBoundingClientRect();

    function updateMagnifier(x, y) {{

        const magnifiedX = x * zoom - (canvasSize/2) ;
        const magnifiedY = y * zoom - (canvasSize/2);

        let cropX = x - (canvasSize / (zoom * 2));
        let cropY = y - (canvasSize / (zoom * 2));
        if (cropX < 0) cropX = 0;
        if (cropY < 0) cropY = 0;
        if (cropX + (canvasSize / zoom ) > image.naturalWidth) cropX = image.naturalWidth - (canvasSize/ zoom );
        if (cropY + (canvasSize / zoom )> image.naturalHeight) cropY = image.naturalHeight - (canvasSize/ zoom );

        ctx.clearRect(0, 0, canvasSize, canvasSize);
        ctx.drawImage(image, cropX, cropY, canvasSize/zoom, canvasSize/zoom, 0, 0, canvasSize, canvasSize);
        canvas.style.left = (x + mainImageRect.left - canvasSize/2) + 'px';
        canvas.style.top = (y + mainImageRect.top - canvasSize/2) + 'px';
        canvas.style.display = 'block';
    }}

    image.addEventListener('mousemove', function(event) {{
        const x = event.clientX - mainImageRect.left;
        const y = event.clientY - mainImageRect.top;

        updateMagnifier(x,y);
    }});

    image.addEventListener('mouseleave', function() {{
         canvas.style.display = 'none';
    }});


    </script>
    </body>
    </html>
    """

    html_component = components.html(component_code, height=650)
    return html_component

if __name__ == '__main__':
    st.title("Shaped Magnifying Glass Example")
    image_path = "sample.png"
    try:
        Image.open(image_path)
    except FileNotFoundError:
        image = Image.new('RGB', (500, 500), color = 'white')
        draw = ImageDraw.Draw(image)
        draw.ellipse((100, 100, 400, 400), fill = 'blue', outline ='black')
        image.save(image_path)

    st.subheader('Circular Magnifying Glass')
    create_shaped_magnifying_glass(image_path, zoom_factor=3, shape='circle', glass_size = 150)

    st.subheader('Square Magnifying Glass')
    create_shaped_magnifying_glass(image_path, zoom_factor=3, shape='square', glass_size = 200)
```
This version introduces the 'shape' and 'glass_size' parameters which allows us to switch between circular and square magnified areas. We also pass the size of the magnified area to the html, to be used in sizing the canvas and adjusting the positioning of the canvas over the image, which is important if you have rectangular magnifying glass regions.

Now, for a more advanced scenario, consider if you wanted to handle multiple zoom levels and also the edge cases where the mouse goes off the image. This is where the javascript gets a little trickier to manage. This should be more illustrative.

```python
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import streamlit.components.v1 as components
import base64

def create_advanced_magnifying_glass(image_path, zoom_levels=[1, 2, 3], initial_zoom_index=1, glass_size=200):
    """
        Creates a streamlit component for a magnifying glass effect with customizable zoom and boundary handling
    """

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    component_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        #container {{
        position: relative;
        }}
        #magnified {{
            position: absolute;
            border: 2px solid black;
            border-radius: 50%;
            pointer-events: none;
            display: none;
        }}

    </style>
    </head>
    <body>
    <div id="container">
        <img id="main-image" src="data:image/png;base64,{encoded_string}" style="max-width: 600px; cursor: crosshair;" >
        <canvas id="magnified" width="{glass_size}" height="{glass_size}"></canvas>
    </div>

    <script>
    const image = document.getElementById('main-image');
    const canvas = document.getElementById('magnified');
    const ctx = canvas.getContext('2d');
    const zoomLevels = {zoom_levels};
    let currentZoomIndex = {initial_zoom_index};
    let currentZoom = zoomLevels[currentZoomIndex];
    const canvasSize = {glass_size};
    let mainImageRect = image.getBoundingClientRect();

    function updateMagnifier(x, y) {{
        let zoom = zoomLevels[currentZoomIndex]

        const magnifiedX = x * zoom - (canvasSize/2) ;
        const magnifiedY = y * zoom - (canvasSize/2);

        let cropX = x - (canvasSize / (zoom * 2));
        let cropY = y - (canvasSize / (zoom * 2));
        if (cropX < 0) cropX = 0;
        if (cropY < 0) cropY = 0;
        if (cropX + (canvasSize / zoom ) > image.naturalWidth) cropX = image.naturalWidth - (canvasSize/ zoom );
        if (cropY + (canvasSize / zoom )> image.naturalHeight) cropY = image.naturalHeight - (canvasSize/ zoom );

        ctx.clearRect(0, 0, canvasSize, canvasSize);
        ctx.drawImage(image, cropX, cropY, canvasSize/zoom, canvasSize/zoom, 0, 0, canvasSize, canvasSize);
        canvas.style.left = (x + mainImageRect.left - canvasSize/2) + 'px';
        canvas.style.top = (y + mainImageRect.top - canvasSize/2) + 'px';
        canvas.style.display = 'block';
    }}


    image.addEventListener('wheel', function(event) {{
        if (event.deltaY > 0 && currentZoomIndex > 0) {{
            currentZoomIndex--;
        }} else if (event.deltaY < 0 && currentZoomIndex < zoomLevels.length -1){{
            currentZoomIndex++;
        }}
        updateMagnifier(event.clientX - mainImageRect.left, event.clientY - mainImageRect.top);
        event.preventDefault();
    }});

    image.addEventListener('mousemove', function(event) {{
        const x = event.clientX - mainImageRect.left;
        const y = event.clientY - mainImageRect.top;

        updateMagnifier(x,y);
    }});


    image.addEventListener('mouseleave', function() {{
         canvas.style.display = 'none';
    }});


    </script>
    </body>
    </html>
    """

    html_component = components.html(component_code, height=650)
    return html_component

if __name__ == '__main__':
    st.title("Advanced Magnifying Glass Example")
    image_path = "sample.png"
    try:
        Image.open(image_path)
    except FileNotFoundError:
        image = Image.new('RGB', (500, 500), color = 'white')
        draw = ImageDraw.Draw(image)
        draw.ellipse((100, 100, 400, 400), fill = 'blue', outline ='black')
        image.save(image_path)

    st.subheader('Magnifying Glass with Zoom Levels')
    create_advanced_magnifying_glass(image_path, zoom_levels=[1, 2, 4], initial_zoom_index=1)
```

This version incorporates mouse wheel input, allowing the user to cycle between defined zoom levels. It’s vital here to prevent default `wheel` scrolling behaviour using `event.preventDefault()`. Further, if you're working with very large images, you’d likely want to also consider using a tile-based approach for the underlying image, rather than loading a single enormous one.

Implementing such an effect is, as you can see, a non-trivial endeavor that requires delving into both front-end (javascript) and back-end (Python) concerns, specifically how they relate to streamlit components. There are no 'magical' shortcuts to achieving this effect beyond careful manipulation of the canvas element and the events surrounding it.

For further study, I'd recommend looking into the following:
*   **"HTML5 Canvas"** by Steve Fulton and Jeff Fulton, for a solid foundation on canvas manipulation.
*   **"Eloquent JavaScript"** by Marijn Haverbeke for a deeper dive into JavaScript event handling and DOM manipulation.
*   The **"Streamlit Components documentation"** for understanding how custom HTML and JavaScript integrate within Streamlit applications.

Remember that these examples are starting points. Optimizing performance, especially for large images or high zoom factors, might necessitate more complex caching or rendering techniques. This can get significantly more advanced depending on the specific demands of your project.
