---
title: "Why is rails 6 app Template::Error giving that Webpacker can't find media/images/logo.png in .../public/packs/manifest.json?"
date: "2024-12-14"
id: "why-is-rails-6-app-templateerror-giving-that-webpacker-cant-find-mediaimageslogopng-in-publicpacksmanifestjson"
---

alright, i've seen this rodeo before. the "rails 6 app template::error, webpacker can't find media/images/logo.png" – it's a classic. it usually boils down to a few common culprits, and i’ve banged my head against the wall with each of them at some point. let's break it down.

first, the error message itself is pretty clear, actually. rails and webpacker are playing a game of hide-and-seek with your assets, and webpacker can't find the image where it expects to. specifically, `manifest.json` which is webpacker’s map of processed assets, doesn’t have an entry for `media/images/logo.png`. this means the pipeline didn't process that file correctly.

in my early days with rails, back when webpacker was still fairly new, i had a similar issue migrating a legacy app. we had a bunch of images in the old `public/assets` directory. we naively thought that moving everything over to webpacker's `app/javascript/images` would magically make them work. we were so wrong. after a full afternoon of troubleshooting, i found out that it's not about just placing images anywhere. you actually need to import or require them.

the most frequent cause is that the file isn’t actually being included in the webpacker processing pipeline at all. this happens when you’re not referencing the image in your javascript or stylesheet files, which webpacker relies on to know what needs to be bundled. webpacker is smart but it's not psychic. it doesn't just look into your folders and decide what's important.

let’s consider some typical scenarios.

**scenario 1: missing or incorrect image path in your javascript/css files.**

if you have an image in your `app/javascript/images` folder, and you're using it inside of your css/javascript, you can’t reference it using the direct path like you would in old asset pipeline days. instead you must either `require` or `import` the image in your javascript or use `asset-url` in your stylesheet, depending on how you organize your app. let’s say you want to put your logo on your application, you would need to do something like this:

```javascript
// app/javascript/packs/application.js
import '../stylesheets/application.scss'
import logo from '../images/logo.png';

document.addEventListener('DOMContentLoaded', () => {
  const logoElement = document.getElementById('logo-image');
  if (logoElement) {
    logoElement.src = logo;
  }
});
```

```scss
// app/javascript/stylesheets/application.scss

body {
   background-image: asset-url('background.jpg');
   ...
}
```
and in your html, you could use the `#logo-image` id like this.
```html
<img id="logo-image" alt="website logo">
```
the key here is the `import logo from '../images/logo.png';` line. webpacker picks up on that line, processes the logo, and includes its output in `manifest.json`. the resulting `logo` variable holds the path that it will have after webpacker is done with it, which usually is not the original path in your source folder.
if you use the path directly like `/media/images/logo.png` instead, webpacker will never know that the file is needed and that it must be handled. and webpacker will not do the processing that produces the manifest entry.

**scenario 2: incorrect manifest entry**

even if you import or require the file correctly, sometimes the manifest entry can go haywire. this can be due to webpacker being out of sync, incorrect configurations, or a corrupted manifest file, sometimes some plugin or version mismatch. the fix in this case is to recompile the assets and to make sure the configuration is ok.
```bash
./bin/rails webpacker:compile
```
or
```bash
rm public/packs/manifest.json
./bin/rails webpacker:compile
```
this clears and rebuilds the entire manifest json. sometimes you will need to restart the rails server after this, because of the cache that might exist in memory or some other thing the server has. it just needs to go over the manifest again.

 **scenario 3: placement of files**

 the error explicitly mentions `public/packs/manifest.json`. the main webpacker config, usually located at `config/webpacker.yml`, determines where webpacker will output the compiled assets and where it will generate the manifest. normally, they go to `public/packs`.

if your images are not located in `app/javascript/images` or a similar directory configured in webpacker, they won't be picked up. this could also happen if you're trying to directly use images from `/public/media/images` without them being bundled via webpacker. you cannot reference them directly in your html as if they were inside the assets pipeline. if your images are in a different path, you will have to tell webpacker where to look for them, which would be more config work and we will not cover here.
a good practice is to have all the assets that webpacker manages to be under `app/javascript`.

**scenario 4: manifest caching issues**

sometimes the problem is the manifest being cached, in memory or disk. i remember that with rails 5 to 6 upgrade, we had this problem a lot. even after clearing the cache, we had to restart the rails server and sometimes even the browser to get rid of it, as browsers also cache this kind of file.
so if you're pretty sure that your imports are correct, sometimes a good old 'turn it off and on again' type restart might fix it. also verify that you do not have caching mechanisms in between. sometimes reverse proxies or cdns might cause the problem.

**debugging tips:**

*   **check the `manifest.json` directly:** look at the file inside the `public/packs` directory. inspect if the image you're looking for has a corresponding entry. if it's absent, this confirms it's not being picked up correctly. it’s not that big of a file so don’t be afraid to open it with your preferred text editor, for a quick look.
*  **webpacker logs:** run webpacker with increased logging, sometimes that log output contains hints about your error. use `./bin/rails webpacker:compile --trace` to see the full output.
*  **browser console:** inspect the network tab on your browser's developer tools, look at what files are being requested and failing. if it's a 404 status, it means you're trying to access a file that doesn't exist at the specified url, usually means that the asset is not being served by webpacker, which usually means that the asset is not being bundled by webpacker.
*  **simplify:** comment out everything that’s not strictly needed, and then enable functionality bit by bit. sometimes this helps you find out where the real problem is happening. this could help you isolate the cause to some other factor of your code.

**general resource recommendations (not links):**

*   _webpack documentation_: a deep dive into webpack itself, which is the underlying asset bundler used by webpacker. it’s complex, but the official documentation is the go-to if you plan to tweak the configuration.
*   _rails guides_: go to the official ruby on rails documentation about webpacker, that should give you a good starting point to understand all the configuration options you have available.
*   _understanding webpack by sean larkin_: although not ruby on rails specific, this book should provide you with a very solid background on how webpack works and how you can get the best out of it.
*   _agile web development with rails 7_: there is an updated version with rails 7, but still the sections on assets handling are useful, although webpacker came out before rails 7 they both work hand in hand in rails 7 as well. it contains sections dedicated to assets, javascripts and stylesheets, and you may find there useful information.

it can be frustrating when your images go missing, but usually the solution is fairly simple. it's all about understanding the pipeline. and remembering the webpacker doesn't just magic things into existence. you have to explicitly tell it what you want to include, just like coding anything else.
a piece of advice a colleague gave to me once: when in doubt, recompile your assets. it's surprising how often that solves the problem. (the real reason was that i missed a commit in git)
