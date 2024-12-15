---
title: "Why is Webpacker to Shakapacker injecting files into HEAD with odd names and no fingerprint?"
date: "2024-12-15"
id: "why-is-webpacker-to-shakapacker-injecting-files-into-head-with-odd-names-and-no-fingerprint"
---

so, you're seeing webpacker, or more accurately shakapacker these days, throwing some weirdly named files into your html head, and they're not getting fingerprinted. yeah, i've been there, more times than i care to remember. it’s the kind of thing that makes you question your life choices, especially when you’re chasing down a deploy bug at 3 am.

first off, let's break down why this happens. shakapacker, or webpacker before it, is designed to manage your javascript and css assets. it compiles, bundles, and optimizes everything, then it tells rails how to link those assets in your views. when you’re using something like `javascript_pack_tag` or `stylesheet_pack_tag`, things work as expected: you get those nice, fingerprinted, cache-busting urls.

but here’s where it gets a little tricky. shakapacker also has a feature called ‘manifest.json’. this file is the bridge between the compiled assets and your rails app. it maps logical names (like ‘application.js’) to the actual compiled filenames with their content hashes for caching. it also can sometimes include files you might not expect, especially when you dive into the deep end of webpack configuration and plugins.

i remember once, back in the webpacker 3 days, i was pulling my hair out because a custom font file was mysteriously showing up in the head of my application. it wasn't referenced anywhere in my rails views, but it was still there, a tiny rogue element with a name that looked like something a cat typed on a keyboard after walking on the laptop. turns out, i had configured a webpack plugin to process font files, and while webpack knew what to do with them, rails did not. the plugin was spitting out the font as an asset, webpack was including it in the manifest, and webpacker was helpfully injecting it into the head.

so, why are these injected with odd names and no fingerprints? well, it’s usually because they aren't being processed through the standard shakapacker pipeline. they are being treated as static assets that should be included directly as they are. if an asset isn’t being passed through webpack's hash generation processes, it won’t get a fingerprint in its name. think of it like this: if a file bypasses the automated factory and gets included by hand it does not go thru the quality control checks.

files like that do not have the `[contenthash]` part in the filename, hence they lack the fingerprinting.

let’s look at a few common scenarios and ways to fix them.

**scenario 1: raw static files mistakenly included**

sometimes, you accidentally include files directly into your javascript entry point that shouldn't be there, or you use some kind of webpack loader that does not handle it properly. take this case where you include an svg file directly in a javascript file using:

```javascript
// app/javascript/packs/application.js
import "./some_svg.svg";

// and this file is just a regular svg file
// that is not a component
// some_svg.svg
// <svg ... />
```

this will make webpack grab it and include it in the manifest, and shakapacker will just include it to the head. in that case you have a couple of things you can do. you could instead tell webpack to process the image through the image loader:

```javascript
//webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.(png|jpe?g|gif|svg)$/i,
        use: [
          {
            loader: 'file-loader',
            options: {
                name: '[name].[contenthash].[ext]',
                outputPath: 'images',
            },
          },
        ],
      }
      // ... other rules
    ]
  }
}
```

this will tell webpack to load images as files in the 'images' folder inside the public folder, instead of trying to include them directly in the head. remember to install `file-loader` and `url-loader` to make this work.

then you can refer to the image later by:

```javascript
import  imagePath  from "./some_svg.svg"
document.getElementById('myImage').src = imagePath
```

or use a standard html `img` tag to directly include it via:

```html
<img src="<%= asset_path('images/some_svg.contenthash.svg') %>">
```

this way, the assets are now being handled properly and you will have the fingerprinted paths.

**scenario 2: custom webpack plugin injecting files**

another common cause is custom webpack plugins that add to the asset list. let's say you have a plugin that creates a special ‘config.json’ file during build:

```javascript
// webpack.config.js
const { DefinePlugin } = require('webpack');
const  fs  = require('node:fs')
class  CreateConfigPlugin {
  apply(compiler) {
    compiler.hooks.emit.tapAsync('CreateConfigPlugin', (compilation, callback) => {
      const config = { api: { url: process.env.API_URL }};
      const jsonString = JSON.stringify(config, null, 2);
      fs.writeFileSync('public/config.json', jsonString)
      compilation.assets['config.json'] = {
          source: () => jsonString,
          size: () => jsonString.length,
        };
        callback();
    });
  }
}


module.exports = {
    plugins:[
        new CreateConfigPlugin(),
        new DefinePlugin({
            'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
          }),
      ]
    //...other configurations
};
```

this plugin adds ‘config.json’ directly to the output folder but it also includes it in the compilation.assets, which then shakapacker interprets as a file it should inject in the head. a solution would be to only generate the file in the assets and then to manually handle it via ruby.

```ruby
# app/helpers/application_helper.rb

def config_json
  Rails.public_path.join('config.json')
end

def config_json_content
  if File.exist?(config_json)
    File.read(config_json).html_safe
  else
    '{}'
  end
end

```

then you can use it as you like like.

```html
<script>
  window.appConfig = <%= config_json_content %>;
</script>
```

this separates the concerns. webpack is only responsible for the build process, and rails is responsible for reading the resulting generated files.

**scenario 3: misconfigured copy webpack plugin**

sometimes, it’s not about files you’re actively including in your javascript, but rather files that are passively copied over. if you have webpack plugin like `copy-webpack-plugin` configured like this:

```javascript
// webpack.config.js
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
  plugins: [
    new CopyWebpackPlugin({
      patterns: [
        { from: 'assets/fonts', to: 'fonts' },
      ],
    }),
  ],
  //... other config
};
```

and you have fonts that are located inside of the `assets/fonts` folder, these will get copied to `public/fonts` but also included in the manifest, which is another thing you do not want to have. again, the solution here is to exclude those files from the compilation.assets object by not including them in the webpack process. webpack should only be used to process and generate code, not to copy files around.

to fix this do not include any files in the `compilation.assets` object and instead process the files with ruby and `asset_path`.

```html
<link rel="preload" href="<%= asset_path('fonts/my_font.woff2') %>" as="font" type="font/woff2" crossorigin>
```

**general tips**

*   **read the documentation:** seriously, the official webpack and shakapacker docs are your friends. they might seem dense at first, but they are worth their weight in gold. understand how webpack's module resolution and asset handling works, and how that relates to shakapacker.

*   **inspect the manifest:** the `public/packs/manifest.json` file is your window into what shakapacker thinks it's doing. take a look at it. if you see weird entries in there, that's your starting point to figure out what is the culprit.

*   **simplify your configuration:** if you can, try to start with a minimal webpack config and then add complexity incrementally. less is more, especially when you’re trying to debug why some file is getting injected into your head with a crazy name.

*   **check your plugins:** if you are using community plugins, be sure you are doing it the way that the authors intended it. read the docs of the plugins and understand what exactly they do.

*   **use `asset_path`:** for static assets not directly handled by webpack, like fonts or plain images (that don’t need processing), let rails take care of the pathing by using `asset_path` and `image_tag` helper methods.

*   **be wary of ‘automatic’ includes:** sometimes, webpack plugins try to be too helpful and automatically include things in your build process. you need to control what gets processed.

resources? well, i recommend reading “webpack: concepts and configuration” book for a deep dive into webpack. it really helps to understand the underlying mechanisms. for shakapacker specifically, check out the official shakapacker documentation, and the source code. it sometimes can get tricky to track things with the default logs.

and one joke: why did the javascript developer break up with the css developer? because they just couldn’t see eye to eye on layouts.

in conclusion, this usually is because some files are going thru some webpack magic where you are not controlling exactly how they are going to be processed or included. the key to debugging this is understanding what files are ending up in the manifest.json and then investigating why those files ended up there.
