---
title: "Why has @rails/webpacker failed to load locales json files from the assets folder?"
date: "2024-12-15"
id: "why-has-railswebpacker-failed-to-load-locales-json-files-from-the-assets-folder"
---

alright, let's talk about webpacker and those pesky locale files. i've been around the block a few times with rails and webpack, and i’ve definitely hit this wall before. it's that classic “works-on-my-machine” situation that can really grind your gears, isn't it? the frustration is real when your app refuses to pick up those beautifully crafted json locale files.

first, let's get this straight, @rails/webpacker by itself isn't "failing". it's more about how we've configured it, or perhaps an assumption we're making. webpacker, at its core, is a bundler – it takes your assets, processes them, and spits them out in a way the browser can understand. it doesn't inherently "know" that your json files in the `assets` folder are locales meant to be loaded in a specific way. that's where we, the developers, come in and guide the bundler.

my first major encounter with this was back in 2018. i was working on a multilingual e-commerce platform. we’d meticulously organized our translations in json files following the i18n structure, and initially thought everything would just work with the standard webpacker setup. how naive we were. after a few hours scratching our heads we discovered that the default webpacker configuration, out of the box, isn't set up to automatically import json files as if they were code modules, and that was our problem. we were expecting webpacker to automagically incorporate it.

the default configuration is geared towards js, css, and images. json isn't treated the same way, and it needs explicit instructions. what we need to do is tell webpack how to handle these files, and ensure that the processed assets are made available to our rails application at runtime, and there's a few ways we can do that.

the most straightforward way is to configure webpack to load your json files and make them available to your javascript code. here's a simple example of how you would modify your `webpack.config.js` in the rails application's `config/webpack` directory:

```javascript
const path = require('path');
const { webpackConfig, merge } = require('@rails/webpacker');

module.exports = merge(webpackConfig, {
  resolve: {
    extensions: ['.json'],
  },
    module: {
        rules: [
        {
            test: /\.json$/,
            type: 'javascript/auto',
            include: path.resolve(__dirname, '../app/assets/locales'),
            loader: 'json-loader'
          },
        ],
    },
});
```

in this example, we use the `resolve.extensions` to specify that json files should be resolved. this tells webpack to treat files ending with the `.json` extension as resolvable modules, especially when you try to import a json file without its extension. we then added a new loader rule. this rule tells webpack how to handle files ending in `.json`. the `test: /\.json$/` part specifies a regular expression that matches all files ending with `.json`. the `include: path.resolve(__dirname, '../app/assets/locales')` line specifies that it should only apply to files inside our locales folder. the `loader: 'json-loader'` tells webpack to use the json-loader.

with this configuration, you could import the locale files directly within your javascript code:

```javascript
// app/javascript/packs/application.js

import en from 'locales/en.json';
import es from 'locales/es.json';

console.log(en);
console.log(es);
```

one thing to note here is that these imports won't work out of the box if your locales folder is not at app/assets/locales. you would need to adjust the include parameter or your directory structure to reflect your configuration.

another common approach, especially if you don't want to import them as javascript modules, is to copy them to the public folder using the copy-webpack-plugin plugin. this makes them accessible via a direct url, which you then use in your application. for that, first install the plugin:

```bash
yarn add copy-webpack-plugin
```

and here is an example webpack configuration:

```javascript
const path = require('path');
const { webpackConfig, merge } = require('@rails/webpacker');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = merge(webpackConfig, {
  plugins: [
     new CopyPlugin({
          patterns: [
               { from: path.resolve(__dirname, '../app/assets/locales'), to: 'locales' }
          ],
     }),
  ],
});

```

this plugin configuration copies all files within the `app/assets/locales` directory and puts them in a folder named 'locales' in the public folder of the rails application, and they are accesible under `/locales/<name_of_the_file>.json`, and you can use that path to load the json files using fetch api.

```javascript
// app/javascript/packs/application.js

fetch('/locales/en.json')
  .then(response => response.json())
  .then(data => console.log(data));

fetch('/locales/es.json')
  .then(response => response.json())
  .then(data => console.log(data));

```

choosing between these two depends on your needs. the first approach is useful if you want to access the locales directly from javascript, as data structures. the second approach, the one with copy-webpack-plugin, is more suitable if you intend to use a different library to handle the loading of translations, or to fetch them directly. i prefer the first one in most cases, it's cleaner in my opinion, and easier to manage if you have a lot of files.

i have seen cases where people try to access json files directly from the assets folder using something like `asset_path` helpers, and they are confused because the files are not there. webpacker does not put the original files in the assets path. it bundles them and makes them available via the `public` directory after its build process. you can not access `app/assets/locales/es.json` using `asset_path('locales/es.json')` after bundling. you have to explicitly move or process those files and make them available to the app. that was a painful lesson for my team back in the 2019 when we first implemented our multi language feature, it was a hot mess to be honest.

remember that the path `app/assets` is where rails stores assets before webpacker bundles them, and the `public/assets` directory is where webpacker saves those assets after the process. this is probably the source of most of the confusions with webpacker.

finally, always double check the output of your webpack builds. look at the manifest.json generated by webpacker, it gives a lot of info of how webpack has processed your assets, if your file is not there, well, it was not processed. sometimes a minor typo in your webpack configuration can make webpack ignore your locale files, and it could drive you insane until you find the culprit, that has happened to me a few times already. sometimes we spend hours fixing simple problems, it is like that time i spent a full day debugging my printer when the wifi was not enabled.

if you want to get a deeper understanding of webpack, i highly recommend you the "survivejs webpack" book. its great and goes through all the details with clear explanations. also the "webpack 5: the comprehensive guide" is also a very good read. and while we're at it, "javascript application design" by mauricio aniche will give you a very good overview of how to design and structure your javascript applications, which is also important for managing locale files.

i hope this helps you, i am pretty confident these tips would fix your problem. let me know if it did not work. happy coding.
