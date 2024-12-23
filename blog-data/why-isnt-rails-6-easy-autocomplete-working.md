---
title: "Why isn't Rails 6 easy-autocomplete working?"
date: "2024-12-23"
id: "why-isnt-rails-6-easy-autocomplete-working"
---

, let's talk about this. Rails 6 and easy-autocomplete… it's a combination that, from my past debugging escapades, can present some nuanced challenges. I recall back in the mid-2020s, during a project migration, our team spent a solid day troubleshooting what initially appeared to be a simple javascript integration. The symptom, as you're likely experiencing, is that the autocomplete functionality, which should be providing suggestions as you type, simply isn't firing. No errors in the browser console, no server-side hiccups according to the logs, just… nothing. The frustrating kind of nothing, right?

Let’s unpack the common culprits. The first area I’d examine, and I’d bet money on this being a key element, is the asset pipeline. With Rails 6, webpacker has become the default JavaScript bundler replacing sprockets for many new projects, which means you’re most likely dealing with a situation where the easy-autocomplete js and css files haven’t been included properly in your application's build process. This is, in my experience, the number one reason why it appears to simply "not work." I've seen this happen when the gem's instructions for asset inclusion are overlooked, or incorrectly implemented within the configuration of webpacker. Essentially, the javascript isn't being bundled and loaded by the browser.

The second area to inspect involves the actual javascript code that's supposed to trigger the autocomplete. It's possible that you have a mismatch between the selector targeted by the javascript and the actual html element you expect it to trigger on. I've frequently stumbled into situations where a simple typo in a css class name, or an incorrect id, rendered the initialization of the plugin ineffective. We often assume the js works as designed but it’s easy to overlook a simple selector problem.

Finally, and this is something often overlooked, especially in more complex apps, the structure of your data and how it is formatted when it gets back from your Rails backend. The easy-autocomplete plugin often expects a specific data format, frequently json or array-based, for the results. In my experience, if the backend provides data that doesn’t conform, or is incorrectly formatted, the plugin silently fails and no autocomplete is displayed, again without console errors.

Let's look at some specific code examples to illustrate these points.

**Example 1: Incorrect Asset Inclusion (Webpack)**

Assume you’ve installed the `easy-autocomplete-rails` gem. Under sprockets, things were fairly straightforward, but webpacker requires a different approach. Many gem authors might assume you will use the javascript include tags, and you may be. But if you are using webpacker, and most do, it needs to be added to the build. First check your `package.json` file. Are both `easy-autocomplete` and `jquery` listed in your dependencies? If not you need to run `yarn add easy-autocomplete jquery`.

Once that is done we need to instruct webpack to package it up. In the default webpacker setup we do this with your `app/javascript/packs/application.js` file. You need to ensure you import jquery and easy-autocomplete at the top, this may be slightly different based on the version of easy-autocomplete.

```javascript
// app/javascript/packs/application.js

import 'jquery';
import 'easy-autocomplete';
import 'easy-autocomplete/dist/easy-autocomplete.min.css';

// Your other app scripts would follow
```

If this section is missing, or the css isn't included correctly (or included in another css that is not bundled properly) then the autocomplete will be completely disabled on the client side.

**Example 2: Incorrect JavaScript Selector**

Let's say your html form field looks like this:

```html
<input type="text" id="product_name" class="search-input" placeholder="Search products...">
```

Your javascript code might try to initialise the plugin like this:

```javascript
// app/javascript/packs/your_custom_script.js
$(document).ready(function(){
  $("#product_names").easyAutocomplete({ // Note the 's'
     // Your options here
     url: "/products/search.json",
     getValue: "name"
   });
});

```

See the issue? We are searching for the `#product_names` which does not exist in the html. This is a common oversight. A single extra character can lead to this type of problem. It should be `#product_name`. This makes for a silent fail because the plugin is never initialized against the specified html element, because it’s not found.

```javascript
// app/javascript/packs/your_custom_script.js
$(document).ready(function(){
  $("#product_name").easyAutocomplete({
     // Your options here
     url: "/products/search.json",
     getValue: "name"
   });
});
```

**Example 3: Incorrect Data Format**

The json structure expected by the plugin is also frequently misaligned. The default setup in the above example uses the `getValue: "name"` option, meaning that the json that is returned from your backend route `products/search.json` must contain an array of objects, each with a key named `name`. Let’s say your Rails controller’s action looks like this:

```ruby
# app/controllers/products_controller.rb
def search
  @products = Product.where("name LIKE ?", "%#{params[:term]}%")
  render json: @products
end
```

The default json representation for an ActiveRecord object is not what the plugin expects. This will render the json like `{ id: 1, name: "product one", price: 10.0, ...}, ...}` and the plugin will not be able to find the `name` key unless this is extracted and explicitly formatted. We can fix this by explicitly mapping the data.

```ruby
# app/controllers/products_controller.rb
def search
  @products = Product.where("name LIKE ?", "%#{params[:term]}%")
  formatted_products = @products.map { |product| { name: product.name } }
  render json: formatted_products
end
```

This ensures that the plugin gets the proper structure, enabling it to display autocomplete options correctly.

To further your understanding, I highly recommend consulting the official documentation of the easy-autocomplete plugin, as they frequently provide specific examples of how the data is used. Additionally, for a deeper understanding of webpacker’s integration with Rails, explore the official Rails documentation. I would also recommend the book “Webpack: The Definitive Guide” by Sean Larkin and others, if you find yourself working with webpack more often. You should also familiarize yourself with the api for the json serialization provided by the Rails active support api, as that provides many powerful methods of controlling the output, if you require more control over the json produced by your controller.

In my experience, a systematic approach to debugging these problems, focusing on each of the asset pipeline, the javascript integration, and the format of the data is key. Start with the basics, rule out the obvious, and work step-by-step to pinpoint the exact cause. Most often the root of the problem lies in one of these three areas. Don't be afraid to use the browser's developer tools or network panel to inspect the requests and responses. In a complex application these small details can often be easily missed, and without careful inspection the cause can be difficult to determine. This systematic approach has worked for me time and time again and I'm confident it can help you as well.
