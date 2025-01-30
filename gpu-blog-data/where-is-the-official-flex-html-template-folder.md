---
title: "Where is the official Flex HTML template folder structure documented?"
date: "2025-01-30"
id: "where-is-the-official-flex-html-template-folder"
---
The lack of a single, definitive document explicitly outlining a mandated folder structure for Flex HTML templates is a recurring point of confusion for developers transitioning from traditional Flex development to its more web-oriented counterparts. My experience building several complex, modular Flex applications, especially those intended for embedding in existing web platforms, underscores this. The perceived "official" structure isn’t prescribed; instead, best practices have organically emerged from community usage and the inherent needs of managing larger, more maintainable web projects incorporating Flex components.

The term "Flex HTML template" itself requires clarification. We are referring to HTML documents, often coupled with CSS and Javascript, that act as the rendering container for a compiled Flex SWF file in an HTML context, rather than templates used to generate new Flex MXML components. This foundational HTML often sets up the basic environment and initial loading behavior before the Flex application takes over the DOM. Unlike native HTML projects where framework-specific folder layouts are often rigidly defined, Flex is more flexible (pun intended) in this regard, given that it’s primarily designed to generate content displayed in a web browser via a plugin or increasingly, through alternative runtimes and techniques like ActionScript transpilation. This flexibility, however, requires careful planning to avoid a chaotic structure that hinders collaboration and scalability.

My process always begins with separating concerns: the core Flex SWF application, its accompanying HTML rendering context, and any supporting static assets. This leads to a file structure that prioritizes clarity and ease of maintenance. The most commonly accepted and functionally appropriate approach tends towards the following breakdown, which I've consistently used, with minor tweaks, across different project sizes:

1.  **root_directory/**: This is the main container for your project. It generally houses the compiled output (.swf), the HTML template, and relevant subfolders.

2.  **root_directory/assets/**: Static resources such as images, icons, and other media. This makes updating these visual components independent of recompiling the Flex application. It’s a best practice to further categorize these assets (e.g., `/assets/images/`, `/assets/icons/`). This is not directly related to the HTML template, but its proximity is often necessary.

3. **root_directory/css/**:  CSS stylesheets to control the look and feel of the HTML template. Separate from any styles embedded in the Flex SWF; this ensures a clear distinction between web and application styles. It’s beneficial to use descriptive CSS file names (`layout.css`, `theme.css`, etc.).

4.  **root_directory/js/**: Contains Javascript files for any interactive behaviors or plugins required by the HTML template. Often this consists of scripts responsible for handling SWF embedding and parameters.

5. **root_directory/index.html**: This is the primary HTML file serving as the template. Typically, it includes a placeholder (`<div>`) or a `<object>` tag, where the Flex SWF is injected. Its role is to setup the correct loading and sizing.

6. **root_directory/[application name].swf**: The compiled SWF file, often named after the application.

This layout promotes clean code separation and allows multiple Flex applications to co-exist within the same project context, providing each its own rendering HTML.

Let me illustrate this with concrete examples.

**Example 1: Basic Setup**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>My Flex Application</title>
    <link rel="stylesheet" href="css/layout.css">
</head>
<body>
    <div id="flex-container"></div>
    <script src="js/swfobject.js"></script>
    <script>
        var flashvars = {};
        var params = {
            menu: "false",
            scale: "noScale",
            allowFullscreen: "true",
            allowScriptAccess: "always",
            bgcolor: "#ffffff"
        };
        var attributes = {
            id: "myFlexApp"
        };
       swfobject.embedSWF("my_application.swf", "flex-container", "800", "600", "10.0.0", "expressInstall.swf", flashvars, params, attributes);

    </script>
</body>
</html>
```

*Commentary:* This represents a typical `index.html` that acts as the rendering host. It includes a div with the ID 'flex-container' where the SWF will load. It also references an external CSS file `layout.css` and `swfobject.js`, a common library for reliable SWF embedding. The Javascript within the `<script>` tag uses `swfobject` to embed the `my_application.swf` into the specified container with specified parameters.

**Example 2: Handling Dynamic Loading and Parameters**

```javascript
// js/app-loader.js

function loadFlexApp(containerId, swfPath, width, height, params) {
  var flashvars = params || {};
  var embedParams = {
    menu: "false",
    scale: "noScale",
    allowFullscreen: "true",
    allowScriptAccess: "always",
    bgcolor: "#ffffff",
  };
  var attributes = {
    id: "flex-app",
  };

  swfobject.embedSWF(
    swfPath,
    containerId,
    width,
    height,
    "10.0.0",
    "expressInstall.swf",
    flashvars,
    embedParams,
    attributes
  );
}
```
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dynamic Flex Loader</title>
    <link rel="stylesheet" href="css/theme.css">
</head>
<body>
  <div id="flex-container-1"></div>
  <div id="flex-container-2"></div>
  <script src="js/swfobject.js"></script>
    <script src="js/app-loader.js"></script>
   <script>
      loadFlexApp('flex-container-1', 'app1.swf', '400', '300',{data:"first instance data"});
      loadFlexApp('flex-container-2', 'app2.swf', '600', '400',{data:"second instance data"});

   </script>
</body>
</html>

```

*Commentary:* This shows a more advanced scenario where a separate Javascript file (`app-loader.js`) encapsulates the SWF embedding logic as a reusable function. It's called in the HTML with two different instances of the Flex application embedded to two different HTML elements.  This approach makes it easier to handle multiple instances of different or parameterized Flex applications. It also provides more maintainability if future changes to the embedding method are needed, allowing them to be applied at one point only.

**Example 3: CSS integration for HTML container**

```css
/* css/layout.css */
body {
    font-family: sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f0f0f0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

#flex-container {
    border: 1px solid #ddd;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* css/theme.css */

#flex-container-1, #flex-container-2 {
  margin: 20px;
    border: 1px solid #ddd;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  display:inline-block;
}

```

*Commentary:* These CSS files define the presentation for the HTML elements that contain Flex application. This separation ensures that web page aesthetics are managed distinct from Flex application styles, promoting better organization and flexibility. The code shows example of basic container styles for both single and multiple Flex Application.

For developers seeking resources beyond the standard Adobe documentation (which focuses on the ActionScript language and Flex SDK itself), I recommend exploring online communities and books dedicated to web development best practices. Publications emphasizing structured web application development, modular CSS architecture, and JavaScript design patterns are all highly relevant. These can provide a deeper understanding of how to structure a project and incorporate a Flex application seamlessly into a broader web context. It's also beneficial to investigate resources on using Javascript libraries, specifically those designed for embedding Flash objects effectively, like the one shown here: `swfobject.js`. Knowledge of the general best practices for frontend development, such as asset optimization, is crucial regardless of specific technology use case. Ultimately, the optimal folder structure for Flex HTML templates, like the ones demonstrated, must be informed by general web best practices adapted for the specifics of Flex technology.
