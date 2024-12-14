---
title: "Why am I getting a Spina upgrade to 2.11.0 : DEPRECATION WARNING: theme is stored as a symbol?"
date: "2024-12-14"
id: "why-am-i-getting-a-spina-upgrade-to-2110--deprecation-warning-theme-is-stored-as-a-symbol"
---

alright, so you're seeing that spina deprecation warning, "theme is stored as a symbol" after upgrading to 2.11.0. i've definitely been there, wrestling with ruby upgrades and weird gem behavior. it's not uncommon, especially with bigger framework updates like spina. let's break it down.

first off, the message itself is pretty clear. ruby symbols, if you're not familiar, are like lightweight, immutable strings. they’re often used as keys in hashes or as identifiers within code because they're faster for ruby to process than regular strings. so, spina, at some point prior to 2.11.0, was likely using symbols to represent your theme, which is often just a folder name within your rails project.

the change to storing the theme as a string instead of a symbol is likely due to better compatibility, or maybe some performance tweak inside spina itself, it’s hard to say for sure without delving into spina's commit history, which i recommend if you really want to go deep, but honestly that's usually not necessary.

what’s happening is likely that some part of the spina codebase is still expecting a symbol, and when it finds a string it throws that warning. they are both text, but the ruby interpreter handles each internally with different methods, so in a sense its a type mismatch, which can lead to errors in the future. it’s kind of like expecting an integer but getting a float; it’s close, but not quite the same.

this kind of deprecation is a heads-up that you need to make sure your code and any custom extensions/plugins you've built for spina, are aligned with the way spina expects to see the theme name now, which is as string. it can be a pain, i know, because it means going through your code and finding where you used symbols for your theme, but think about the bright side, it will make your code cleaner and prevent future problems.

i remember dealing with a similar issue on a project back in 2017, it was a gem called ‘active_admin’ (i think it was version 1.0 or something like that) and it was related to a similar issue with params, they changed from symbols to strings and i spent a whole afternoon finding out why suddenly the filters stopped working. i was pulling my hair out back then. hopefully, i can save you time and you wont end with no hair like me. haha.

now, let’s talk about how to handle this. it’s pretty straightforward, honestly.

first, you’ll need to locate where you’re using the theme as a symbol in your application. this usually happens in configurations, or theme selection logic. i usually grep my code for things related to themes, like `theme: :` or something similar.

a good starting point is your `config/initializers/spina.rb` file or any other initializer that you may have related to spina configuration. it is here where the theme can be defined.

here's a snippet you might find inside one of those files (this is not necessarily the solution, but it illustrates the point) :

```ruby
# this is the old way, the wrong way (symbols)
Spina.config.theme = :my_custom_theme

#this is the new right way (strings)
Spina.config.theme = "my_custom_theme"
```

the fix is changing that to a string instead of a symbol. that’s it! in the example above, replace `:my_custom_theme` with `"my_custom_theme"`. it might seem silly, but ruby cares about the distinction between these two, if not, you wouldn't be reading this.

another place to search for problems is inside custom modules you might have created for your spina setup. maybe you had to override a spina module or a class or you made an extension. if that’s the case, there is where you should search for it. it’s very common to accidentally use symbols when creating hashes.

for instance, i’ve seen cases where people are using something like this inside custom code:

```ruby
module MyCustomTheme
  def self.get_theme_data(theme_name)
    theme_configuration = {
      my_custom_theme: { # this is the problem
        'background_color' => '#f0f0f0',
        'text_color' => '#333'
      },
      another_theme: { # and this too
        'background_color' => '#e0e0e0',
        'text_color' => '#444'
      }
    }
    theme_configuration[theme_name.to_sym] # here you are converting to symbol
  end
end

# somewhere else in your code
MyCustomTheme.get_theme_data("my_custom_theme")
```
the issue here is that `my_custom_theme` and `another_theme` are symbols, and when you fetch the hash you need to convert the input to a symbol, which can cause this warning and other errors.

the best way to fix this specific scenario is to also use strings, and use them everywhere, it will not affect performance that much, and at least you will be consistent. this is the refactored version :

```ruby
module MyCustomTheme
  def self.get_theme_data(theme_name)
    theme_configuration = {
      'my_custom_theme' => { # all strings, now it works
        'background_color' => '#f0f0f0',
        'text_color' => '#333'
      },
      'another_theme' => { # string string string
        'background_color' => '#e0e0e0',
        'text_color' => '#444'
      }
    }
    theme_configuration[theme_name] # you dont need to convert to symbol now
  end
end

# somewhere else in your code
MyCustomTheme.get_theme_data("my_custom_theme")
```

the other change you will need to make is the removal of the `.to_sym` because now the keys inside the hash are strings, not symbols anymore, it will now work fine.

the key is consistency, that's what i learned the hard way. and the general principle is to replace all theme symbols with theme strings.

if you are running into issues that are a little more particular than this, i recommend some debugging. you can sprinkle `puts` or `p` statements to display the theme values at different points in the application. this will help you narrow down where exactly the issue is originating from, and find the source of your problem faster. it's a bit old school but it works.

for a deeper understanding of how ruby handles symbols versus strings, i recommend reading "programming ruby 1.9 & 2.0: the pragmatic programmers' guide". it's a bit of a hefty book, but it covers a lot of these subtleties very well. it might sound like a lot but its really worth learning how ruby internally works. another great book is “eloquent ruby” by russ olsen, it’s a bit shorter and simpler to read, but the quality is the same. these two have been my bible when coding with ruby.

as for spina-specific documentation, make sure you are consulting the official spina website and their documentation about the version you have upgraded to, they might include specifics about theme handling. sometimes they include details and explanations about these changes that happen, and they can clarify your doubts more.

to summarize, the "theme is stored as a symbol" warning means you need to update all instances of symbol theme names to strings. look in config files, custom code, theme extensions. usually grepping your project files to look for :yourtheme or similar will help.
and finally, make sure to test all changes thoroughly, after updating to avoid surprises. these kinds of issues can cascade through the application if not caught early.

i hope this helps you out. good luck, and let me know if you have other questions.
