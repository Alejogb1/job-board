---
title: "Can Import Maps be used with Font Awesome in Rails 7?"
date: "2024-12-23"
id: "can-import-maps-be-used-with-font-awesome-in-rails-7"
---

Okay, let’s get into it. Funny enough, I actually tangled with this exact scenario a couple of years back, migrating a particularly messy Rails 6 app to 7. The team was insistent on Font Awesome, and naturally, we wanted to leverage the new import maps feature instead of relying on the old asset pipeline or node module shenanigans for font management. The short answer? Yes, absolutely, import maps can handle Font Awesome in Rails 7, but it requires a bit more consideration than simply swapping out a gem or adding a yarn dependency.

The primary appeal of import maps is that they give you control over how the browser resolves module specifiers, effectively bypassing the need for a complex bundler in many cases. This is particularly attractive for a library like Font Awesome, which is essentially just a collection of CSS and font files. We want to serve these assets directly rather than bundling them into a single javascript file, as was common practice previously.

First off, let’s establish a baseline. The initial setup for Font Awesome, traditionally, involves either a gem that handles asset inclusion or using npm and a bundler such as webpack. These approaches often lead to bloat, particularly when you’re not utilizing all of Font Awesome’s icons or variants. Import maps, on the other hand, enable granular control, allowing us to import only the necessary files and avoiding the over-bundling problem that can affect page load performance.

The core of the import map integration relies on telling the browser where to find the necessary Font Awesome files. Instead of a single 'font-awesome' identifier, we'll break it down into specific files. We typically start by hosting the font files in the app's ‘public’ directory, which provides an ideal location to directly serve assets without the overhead of the asset pipeline.

Here's a typical scenario: Say you've downloaded the latest Font Awesome release and unpacked it into `public/fonts/fontawesome`. Within that directory, you’ll have files like `fontawesome.css`, as well as the various font files in formats like `.woff2`, `.woff`, `.ttf`, and even `.svg` if you are aiming to support a broader range of older browsers.

Now, you need to define your import map. This is typically done within `config/importmap.rb`. Here's how we might configure it for this Font Awesome setup:

```ruby
# config/importmap.rb
pin "font-awesome-css", to: "fonts/fontawesome/css/fontawesome.css", preload: true
pin "@fortawesome/fontawesome-free/webfonts/fa-brands-400.woff2", to: "fonts/fontawesome/webfonts/fa-brands-400.woff2", preload: true
pin "@fortawesome/fontawesome-free/webfonts/fa-regular-400.woff2", to: "fonts/fontawesome/webfonts/fa-regular-400.woff2", preload: true
pin "@fortawesome/fontawesome-free/webfonts/fa-solid-900.woff2", to: "fonts/fontawesome/webfonts/fa-solid-900.woff2", preload: true
```

This defines an import map that makes Font Awesome available using the "font-awesome-css" module specifier. The `preload: true` flag instructs the browser to fetch this file as quickly as possible and cache it for efficient subsequent requests. The `webfonts` directory will contain your font files for the various styles of font awesome, this way you can reference them easily from your css. Note that I’ve chosen to include the woff2 format as it’s the most widely supported and typically provides the smallest file size.

In your css file, we now need to import the relevant css file:

```css
/* app/assets/stylesheets/application.css */

@import "font-awesome-css";

```
This is how you typically import css files into your application, as if they were a regular package installed from npm.

Now, let's say you're aiming for a specific subset of font awesome icons, we will need to adjust our importmap so we only import necessary font files:

```ruby
# config/importmap.rb
pin "font-awesome-css", to: "fonts/fontawesome/css/fontawesome.css", preload: true
pin "@fortawesome/fontawesome-free/webfonts/fa-brands-400.woff2", to: "fonts/fontawesome/webfonts/fa-brands-400.woff2", preload: true, as: 'fa-brands'
```
In this snippet we are importing only the "brands" font style. We are also giving it an alias 'fa-brands'. This allows us to reference the font with that specifier in our CSS.

```css
/* app/assets/stylesheets/application.css */
@import "font-awesome-css";
@font-face {
  font-family: 'Font Awesome 6 Brands';
  src: url('fa-brands') format("woff2");
  font-weight: 400;
  font-style: normal;
}
```
And in this example we only load the font awesome brands font file.

Now, let's say you want to be explicit about only using certain icons with a specific style. This becomes complex but doable. You’d have to modify the css file you download from font awesome by cutting out all the unnecessary code for icons you aren’t using. Then in your importmap you'd specify the path to that specific css and the specific files for the fonts you want to use. Here is an example of a single font file and the relative importmap entry.

```ruby
# config/importmap.rb
pin "font-awesome-my-subset-css", to: "fonts/fontawesome/css/fontawesome-subset.css", preload: true
pin "@fortawesome/fontawesome-free/webfonts/fa-solid-900.woff2", to: "fonts/fontawesome/webfonts/fa-solid-900.woff2", preload: true, as: 'fa-solid'
```
And then in the corresponding css you would have a similar @font-face declaration as the previous example, except targeting your subset stylesheet. This, combined with css purging techniques, allows you to greatly reduce the overall file size of your font files and css.

A common mistake I've seen is people attempting to load the entire font awesome via a single import map entry. While technically possible, this often ignores the primary benefit of the approach which is granular control over the assets. Avoid specifying a wildcard or a folder structure directly, aim to specify exact files to gain full control. Also, be careful with relative paths; the paths are relative to the public directory.

For deeper understanding, the following are invaluable:

*   **"HTTP/2 in Action" by Barry Pollard:** It will help you fully grasp the importance of the `preload` directive and the efficient use of http/2 for loading files.
*   **"High Performance Browser Networking" by Ilya Grigorik:** This resource provides a comprehensive understanding of browser resource loading and caching mechanisms, crucial for optimizing any web application.
*   **The HTML Standard itself** provides extensive information on how the browser handles import maps. Pay attention to the sections on module resolution and loading.

In conclusion, import maps are certainly usable with Font Awesome in Rails 7, and in my experience, can lead to significant performance benefits when used thoughtfully. It’s not a simple plug-and-play solution, and requires a bit more manual setup, but that’s what provides the power to finely tune the resources being served. You will need to be meticulous in how you construct your import map, pay close attention to the file paths and understand the implications of preloading. But by being mindful of this, you can achieve a much cleaner and efficient asset pipeline.
