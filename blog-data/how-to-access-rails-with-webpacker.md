---
title: "How to access Rails with webpacker?"
date: "2024-12-14"
id: "how-to-access-rails-with-webpacker"
---

alright, so you're looking to get webpacker playing nicely with rails, eh? i've been down that rabbit hole myself, more times than i care to remember. it's one of those things that seems simple on paper, but the details can definitely trip you up.

first off, let's assume you've already got webpacker installed and running in your rails project. if not, that's step one, and there's plenty of good guides for that, like the official webpacker gem documentation and various tutorials out there on medium. i'm going to sidestep that initial setup for this response. assuming you've got your `config/webpacker.yml` and your javascript entry points setup you're ready to go.

the core issue is how do you bridge that gap between your rails views (erb, haml, whatever you're using) and the javascript code being bundled by webpacker? it's not magic. it's about utilizing the helper functions webpacker provides, mostly `javascript_pack_tag` and `stylesheet_pack_tag`.

think of `javascript_pack_tag` as a marker. it tells rails, "hey, webpacker has bundled some javascript, and this is how you reference the output". specifically, it looks at your webpacker configuration, locates the generated javascript file, and outputs the necessary html `<script>` tag to load it in the browser. same goes for `stylesheet_pack_tag`, but for css/sass files.

my first real run-in with this was with a fairly large single-page app using react back in 2019. i was trying to split my javascript code into multiple chunks using webpack's code splitting functionality. i thought i could just add multiple `javascript_pack_tag` lines in my layout to point to each bundle. it... did not go well. the code was loading, but it was a complete mess of naming conflicts and broken dependencies. it turns out, my understanding of how webpacker manages these things was pretty shallow. after much console logging and browser refreshing, i figured out that i had to structure my javascript imports in webpacker entrypoints. i could reference a single entrypoint on each view that imported or required more dependencies. it wasn’t enough.

so here’s an example of a simple erb layout file:

```erb
<!DOCTYPE html>
<html>
<head>
  <title>My Rails App</title>
  <%= csrf_meta_tags %>
  <%= csp_meta_tag %>
  <%= stylesheet_pack_tag 'application', 'data-turbolinks-track': 'reload' %>
</head>
<body>
  <%= yield %>
  <%= javascript_pack_tag 'application', 'data-turbolinks-track': 'reload' %>
</body>
</html>
```
in this case, `application` is the entry point in the `config/webpacker.yml` file. typically that is the `app/javascript/packs/application.js` path.

notice how i'm including both the stylesheet and javascript files in my layout file. this is pretty common because this loads everything app wide. you could add more stylesheets or javascript packs for specific view files. for that use case i tend to use partial views for each view so the code is organized like:

```erb
  # app/views/users/index.html.erb
  <div id="users-index-component">
     <%= render partial: 'users/index_component' %>
  </div>
  <script>
    document.addEventListener('DOMContentLoaded', () => {
       renderUsersIndexComponent();
    })
  </script>
```
```erb
   # app/views/users/_index_component.html.erb
   <%= javascript_pack_tag 'users_index' %>
    <div id="users-index">
        <!-- more content -->
        <h2>Users</h2>
    </div>
```
this is fine if the components of your application have no inter-dependencies. but this is rarely the case. a better alternative is to only import one javascript pack and manage the importing of other modules in that single entry point. for example if i wanted the `users` functionality to work on a few views, i would import the `users_module` component in my application pack, and manage it there. this way my entry point is cleaner. i always prefer a single entry point for my javascript.

```js
// app/javascript/packs/application.js
import '../styles/application.scss'
import './users_module'
import { renderOtherModule} from '../modules/other_module.js'


document.addEventListener('DOMContentLoaded', () => {
    renderOtherModule()
});
```
this gives a better more maintainable code structure. in any javascript module like in the users module i can import components from libraries like react.

```javascript
//app/javascript/packs/users_module.js

import React from 'react';
import ReactDOM from 'react-dom';
import UsersIndex from '../components/users/index'; // Component

const renderUsersIndexComponent = () => {
    const container = document.getElementById('users-index-component');
    if(container){
        ReactDOM.render(<UsersIndex />, container);
    }
};

window.renderUsersIndexComponent = renderUsersIndexComponent
```

you'll see i've also used a dom content loaded event to trigger my rendering function. there are many ways to handle mounting react components but i find this works well for my setups. it gives a chance for the dom to be fully loaded before mounting my components. notice how i had to assign the render function to `window.renderUsersIndexComponent` this is because the script tag in the `users/index.html.erb` file will not have access to the function defined in the javascript module without it being exposed this way. i prefer to only use the window object this way to mount components to the dom. it is not a good practice to add more properties to the window object unless its a library.

but here’s where things get interesting, and a point that tripped me up a lot. webpacker uses webpack's manifest file. this is a json file (usually located in `public/packs/manifest.json` ) that maps the logical names you use in `javascript_pack_tag` and `stylesheet_pack_tag` to the actual file names after webpack has bundled them. the names change on every compile or every time you change the file contents due to cache busting. when you're in development mode, this usually isn’t a problem because webpacker is monitoring your changes and updates the manifest on the fly. however, in production, this can cause major headaches if the manifest is out of sync. i’ve had deployments where i would end up with javascript errors. because the asset names referenced in `javascript_pack_tag` did not match.

my advice? make sure you understand how webpacker handles the manifest file in different environments, especially in production. read up on webpack's manifest plugin. i also recommend reading through "surviving rails" by gregory t brown and "javascript application design" by martin fowler for better understanding on organizing your javascript application.

one more thing, hot module replacement (hmr). it’s basically like magic when it’s working, it allows you to change javascript files and see the change in the browser without doing a full refresh. if you haven’t used it, you're missing out. when i first learned about hmr i was in awe. it sped up development so much! make sure you've got that configured properly in your webpacker setup (see webpack documentation for details).

this is something i learned the hard way, and i wouldn't want any of you suffering from the same issue:  make sure you configure your webpacker.yml file correctly. the default setup might not always work for your particular needs. one time i had an issue with my custom resolve paths. i kept seeing `cannot resolve` errors in my javascript. after some time i realized that webpacker has it's own default resolve paths that overwrite my webpack.config.js. the `resolved_paths` key on the webpacker.yml was the culprit.

in short, webpacker is a powerful tool, but it can be tricky to get your head around at first. especially if you are like me and you try to solve problems without actually reading the documentation. reading the documentation goes a long way. remember that `javascript_pack_tag` and `stylesheet_pack_tag` are your friends. ensure your manifest is up to date, understand the manifest.json file, and configure your yml file according to your needs. and lastly, don’t underestimate hot module replacement, it can be a time saver. oh and one last tip. if you add any files to the `app/javascript/` folder, make sure to restart your rails server because it only loads these files on startup. yes i know, that one is pretty bad, i think i'm getting old.
