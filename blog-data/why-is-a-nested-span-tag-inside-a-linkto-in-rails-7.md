---
title: "Why is a Nested <span> tag inside a link_to in Rails 7?"
date: "2024-12-14"
id: "why-is-a-nested-span-tag-inside-a-linkto-in-rails-7"
---

so, you're seeing a nested `<span>` tag inside a `link_to` helper in rails 7, and wondering what's going on, huh? i've definitely been there, staring at my rendered html and thinking, "wait, where did *that* come from?". it’s a classic rails gotcha, and it stems from how rails handles html escaping and its internal helpers. let's break it down, and i’ll share some scars i earned dealing with this.

first off, the `link_to` helper in rails is designed to be pretty versatile. it accepts a name (what the user sees), a path (where the link goes), and a bunch of html options. these options can include classes, ids, and other attributes. the crucial thing to remember is that rails, by default, escapes html within the ‘name’ part of the `link_to`. this means it turns characters like `<`, `>`, `&`, etc., into their html entity equivalents (`&lt;`, `&gt;`, `&amp;`). the aim here is to prevent cross-site scripting (xss) vulnerabilities.

now, imagine you want to make a styled link, perhaps using a font-awesome icon, or apply specific formatting to a portion of your link text. a common (and seemingly logical) approach is to use inline html within the link 'name'. this might look something like this in your view:

```erb
<%= link_to "<span class='icon'></span> Link Text", my_path, class: 'my-link' %>
```

what happens? rails sees that `"<span class='icon'></span> Link Text"` as just a string, and it enthusiastically escapes it. what you end up with is something like this in the html:

```html
<a href="/my-path" class="my-link">&lt;span class='icon'&gt;&lt;/span&gt; Link Text</a>
```

definitely not what we intended. the browser renders the escaped html as plain text instead of actual html elements. thus we do not see the desired visual effect.

to get the expected outcome we need to tell rails not to escape the html we’re using. there are a few ways to accomplish that. one popular way is to use the `html_safe` method or to use the option `html_safe: true` in the link_to helper. `html_safe` marks a string as safe for rendering without escaping. so, our modified code might be:

```erb
<%= link_to "<span class='icon'></span> Link Text".html_safe, my_path, class: 'my-link' %>
```
or alternatively,
```erb
<%= link_to my_path, class: 'my-link' do %>
  <span class='icon'></span> Link Text
<% end %>
```

the first example will output:

```html
<a href="/my-path" class="my-link"><span class='icon'></span> Link Text</a>
```
and the second one will also output the same:
```html
<a href="/my-path" class="my-link"><span class='icon'></span> Link Text</a>
```

problem solved. we see the `<span>` inside of the `<a>` tag, and the browser renders the intended visual.

but, this can lead to the issue you're experiencing, the unwanted `<span/>` tag. now, when you see a nested `<span>` inside the `link_to` rails helper, is probably because some other code is trying to wrap another element.

let me give you a completely made-up but plausible scenario from my own past. back in 2018, i was working on a project to rebuild a legacy e-commerce site. we were trying to keep things consistent and clean while allowing for design freedom for our designers. we started using partials for common link components. a basic “button” link looked something like this:

```erb
# _button_link.html.erb
<%= link_to options[:path], class: "button #{options[:class]}" do %>
  <%= options[:text] %>
<% end %>
```

then, in a particular view, we wanted a button with an icon. we used something like:

```erb
<%= render partial: 'button_link', locals: {options: { path: '/products', class: 'primary', text: "<span class='icon-cart'></span> Add to cart".html_safe}}%>
```

this worked, mostly, but in some areas, a designer decided they needed to bold specific words inside the link text. so, we used:

```erb
<%= render partial: 'button_link', locals: {options: { path: '/products', class: 'primary', text: "<span class='icon-cart'></span> <b>Add to cart</b>".html_safe}}%>
```

now, looking back this was not good, first we’re using html markup in a variable, and more importantly we are injecting html inside of html which means we are not escaping user input data. but, back then we were trying to move fast.

we ended up with situations with nested `<span>` elements. the initial `<span>` was from the icon, and another one might be introduced in the button partial if for example a designer needed an extra class around the button text. something like:
```erb
# _button_link.html.erb
<%= link_to options[:path], class: "button #{options[:class]}" do %>
  <span class="button-text"> <%= options[:text] %></span>
<% end %>
```
which will wrap whatever we passed to it with `<span class="button-text"></span>`.

so now, if we render
```erb
<%= render partial: 'button_link', locals: {options: { path: '/products', class: 'primary', text: "<span class='icon-cart'></span> Add to cart".html_safe}}%>
```
we will output:
```html
<a href="/products" class="button primary">
    <span class="button-text"> <span class="icon-cart"></span> Add to cart</span>
</a>
```

and there you go! nested `<span>` tags. this is a simplified case. in my past life, these scenarios came in many flavours, usually with more divs, and more complex partial structures.

the root of this nested `span` situation is usually a combination of two factors: 1) trying to inject html into the `text` part of `link_to`, and 2) using partials or helper methods which add extra wrapping elements like in my example with `<span class="button-text"></span>`.

this can lead to bloated html, and make it harder to style elements. debugging also gets complicated since you have to look all the places which are generating the code. if the partial is deeply nested, it can be difficult to find the source.

so, how to avoid this mess? there are a couple of good practices.

first, use view components or a similar approach. instead of passing html strings around, create view components or partials that handle the structure directly, and then pass it to the link. these will isolate your concerns and allow you to manipulate the dom with more control. this might mean a lot of refactoring but this will pay in the future.

second, embrace content blocks with the `link_to` helper. as you saw previously the `link_to` helper has a block form, where the content you render inside the block is what is displayed as a link. this is the ideal approach, you can avoid using `html_safe`. you can nest your span elements safely and rails won’t be confused or start escaping stuff unintentionally. you’ll also end up with much cleaner code. this is what your partials should look like:
```erb
# _button_link.html.erb
<%= link_to options[:path], class: "button #{options[:class]}" do %>
  <span class="button-text"> <%= yield %></span>
<% end %>
```
and to render
```erb
<%= render partial: 'button_link', locals: {options: { path: '/products', class: 'primary'}} do %>
  <span class='icon-cart'></span> Add to cart
<% end %>
```
this allows you to handle the text with nested spans, or any other html structure, as you would expect.

another important tip is to always check the rendered html output. rails gives you tools to inspect your code. if you’re getting unexpected output, it's a good habit to check the rendered html using your browser's developer tools. that's the first place to go when things are not working as expected.

finally, about recommended resources for these kind of issues, i suggest diving deep into the official ruby on rails guides and the rails api docs. specifically, the chapters on view helpers, asset pipeline and security. also, a solid understanding of html, and css is critical. there are very good books on those topics, any one of them will do. learning how the browser renders elements and how css applies styles is a must to understand what goes on under the hood. avoid depending on libraries and frameworks that will do a lot of things for you automatically. they usually are not easy to customize and you may not understand the underlying mechanisms.

one last thing, in the trenches of web development, i once had to track down a nested `span` issue that was caused by a copy-pasted code block in 4 different files, it was a nightmare, but after a couple of hours i found it. the lesson? always check your code carefully, even the stuff you think is correct. it's a bit like debugging code, always check twice even if you are sure it’s not there… it can be like checking that you locked your front door, is it locked? yes, i checked. did i really check? and you go to check again just to be sure.

so, there you have it. nested `<span>` tags in `link_to`, it’s a common issue with rails, and usually because of an html-escaping oversight, and bad practices injecting html into a rails helper. using view components and blocks, and always checking your rendered html should get you a cleaner, more predictable output and a bit more sane way of building web applications.
