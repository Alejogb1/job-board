---
title: "Why is the Rails console overriding `~/.irbrc`?"
date: "2024-12-14"
id: "why-is-the-rails-console-overriding-irbrc"
---

well, this is a fun one, i've been there, seen that, got the t-shirt. the rails console seemingly ignoring your `~/.irbrc` setup is a classic head-scratcher that's tripped up many a developer, including yours truly, back in the day. let's unpack it.

it's not so much that the rails console is actively *overriding* your `~/.irbrc`, as it is a matter of how rails initializes its own environment, and specifically, how it loads configuration files for its interactive console. the core issue stems from the fact that rails, by design, sets up a specific irb environment tailored to rails applications. this environment includes loading rails-specific helpers, models, and configurations and the `.irbrc` at your home directory is loaded before rails starts up so rails will override the changes you made by reloading those specific files.

way back when, i was working on this huge e-commerce platform (it was a mess, i tell you) using rails 2 (yes, i'm old). i wanted to streamline my console experience. i had all these beautiful color configurations, custom shortcuts and autocompletions in my `~/.irbrc`, and i was used to these being present on my usual irb sessions, but when i fired up `rails console` it just felt... vanilla, you get what i mean?. it was as if my custom configurations had vanished into thin air. i spent a solid afternoon banging my head against the wall before i finally figured it out. it was a classic case of not fully understanding the initialization order.

the thing is, rails doesn't just blindly load the system-wide irbrc or the home `.irbrc` when you fire up the console. it uses a series of initializations and configurations specific to itself. after those initializations are done is when rails will source the `.irbrc` file from `rails_app_root/config`. it does this to ensure that the console has access to the application environment, models, helpers, and all the other rails goodies you'd expect.

the crux of the problem here is this: after rails does its application-specific setup, it sources the `config/console.rb` configuration file from the app's folder. this file which may not always exist gives rails a specific instruction on how to set up the irb instance for that app and which file to source after the rails specific configs and settings are loaded, usually it's the file `.irbrc` located inside the `rails_app_root/config` directory and if that does not exist either it will source the usual `~/.irbrc` after loading the app config files. so, in short, it's doing its thing first and *then* looking for your customizations after, it may seem as if it's overriding, but it is just that the rails app setup is happening first and then your home file or `config/` file.

here is an example showing the default rails behaviour which first load the `rails_app_root/config/console.rb`, then `rails_app_root/config/.irbrc`, and last `~/.irbrc`

```ruby
#rails_app_root/config/console.rb

require 'irb/ext/save-history'

# this will only be available on this console
IRB.conf[:PROMPT][:CUSTOM] = {
    :PROMPT_I => "\e[32m>> \e[0m",
    :PROMPT_N => "\e[32m>> \e[0m",
    :PROMPT_S => "\e[32m%l> \e[0m",
    :PROMPT_C => "\e[32m>> \e[0m",
    :RETURN =>  "\e[36m%s\e[0m\n",
    :AUTO_INDENT => true,
  }

IRB.conf[:PROMPT_MODE] = :CUSTOM

IRB.conf[:SAVE_HISTORY] = 1000
IRB.conf[:HISTORY_FILE] = File.join(Rails.root, '.irb-history')


# load this specific irbrc
if File.exist?(File.join(Rails.root, 'config', '.irbrc'))
  load File.join(Rails.root, 'config', '.irbrc')
end
# if we don't have `config/.irbrc`, fallback to ~/.irbrc
# this will be loaded only if `config/.irbrc` do not exist
if File.exist?(File.expand_path('~/.irbrc')) && !File.exist?(File.join(Rails.root, 'config', '.irbrc'))
  load File.expand_path('~/.irbrc')
end
```

so what do we do about it? well, the most straightforward way to fix this is to either create a `.irbrc` inside `rails_app_root/config` or if it already exists, edit it to include your configurations. that will load your customized configurations after rails loads it's specific configurations. the configurations present in `~/.irbrc` will be loaded before the rails configurations, that's why it seems that rails override your configurations.

for example, let's say you want to add some nice colors and aliases to your rails console. here's how you might do that. in your `rails_app_root/config/.irbrc`:

```ruby
# rails_app_root/config/.irbrc
require 'irb/ext/save-history'

# Add some color to the prompt
IRB.conf[:PROMPT][:CUSTOM] = {
  :PROMPT_I => "\e[32m>> \e[0m",
  :PROMPT_N => "\e[32m>> \e[0m",
  :PROMPT_S => "\e[32m%l> \e[0m",
  :PROMPT_C => "\e[32m>> \e[0m",
  :RETURN =>  "\e[36m%s\e[0m\n",
  :AUTO_INDENT => true,
}

IRB.conf[:PROMPT_MODE] = :CUSTOM

# Add aliases
alias bp 'binding.pry'
alias reload! 'reload!'
alias c 'puts "hi"'


# Configure history
IRB.conf[:SAVE_HISTORY] = 1000
IRB.conf[:HISTORY_FILE] = File.join(Rails.root, '.irb-history')
```

this makes your custom settings available when you open the rails console. keep in mind that you may need to either create the file if not exist and also create the `config` directory inside the root of your app folder if not exist and also add this `.irbrc` there. you may also need to tweak the `rails_app_root/config/console.rb` file to ensure the `.irbrc` is being loaded as expected.

alternatively, if you don't want to have a specific `.irbrc` file inside each of your rails apps you can modify `rails_app_root/config/console.rb` to not load any file at all so it will fall back to the default behaviour of loading `~/.irbrc`, in that case, your custom settings will be present since rails wont be overriding it. but that configuration will only apply to that specific rails app.

```ruby
# rails_app_root/config/console.rb

require 'irb/ext/save-history'

# this will only be available on this console
IRB.conf[:PROMPT][:CUSTOM] = {
    :PROMPT_I => "\e[32m>> \e[0m",
    :PROMPT_N => "\e[32m>> \e[0m",
    :PROMPT_S => "\e[32m%l> \e[0m",
    :PROMPT_C => "\e[32m>> \e[0m",
    :RETURN =>  "\e[36m%s\e[0m\n",
    :AUTO_INDENT => true,
  }

IRB.conf[:PROMPT_MODE] = :CUSTOM

IRB.conf[:SAVE_HISTORY] = 1000
IRB.conf[:HISTORY_FILE] = File.join(Rails.root, '.irb-history')

# commented out the load
# load File.join(Rails.root, 'config', '.irbrc') if File.exist?(File.join(Rails.root, 'config', '.irbrc'))
# commented out the fallback too
# load File.expand_path('~/.irbrc') if File.exist?(File.expand_path('~/.irbrc')) && !File.exist?(File.join(Rails.root, 'config', '.irbrc'))

```

this configuration will simply load the default `~/.irbrc` file, and that is why your configurations will be available.

one thing to remember is that if you are using something like rvm or rbenv, make sure the configurations of your ruby enviroment do not interfere with the load of the `.irbrc` files.

for further reading on this subject, i'd recommend diving into the irb documentation, specifically the section about initialization and configuration. the official ruby documentation contains a very comprehensive explanation of how irb works and the lifecycle of the initialization process. while it might not be specific to rails, it offers a valuable insight that will give you a solid foundational understanding of what's going on under the hood. i also recommend to check the source code of rails itself to further understand how rails is setting up the irb environment and also to see how the `console.rb` is being loaded. this will give you a clear idea of the initialization order. if you are using a specific rails version i recommend going to that specific version of the rails repo. and as a last recommendation, there's a good book named 'Metaprogramming Ruby' by Paolo Perrotta, in that book you can learn in great detail about ruby internals that will give you the necessary knowledge to further understand how irb works and why rails is interfering or not with it.

finally, i've been using rails since forever, and every now and then i stumble upon details like these, it's part of the journey and keeps us humble, right?. it's not always a glamorous world, is it?. it reminds me of that old joke: why do programmers prefer dark mode? because light attracts bugs haha. anyways, hope it helps.
