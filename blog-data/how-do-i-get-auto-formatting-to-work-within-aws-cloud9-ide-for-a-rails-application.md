---
title: "How do I get auto-formatting to work within AWS Cloud9 IDE for a rails application?"
date: "2024-12-14"
id: "how-do-i-get-auto-formatting-to-work-within-aws-cloud9-ide-for-a-rails-application"
---

alright, so you're having trouble getting auto-formatting to play nice with your rails app in cloud9, i've been there, spent countless hours tweaking configurations, feels like forever ago. it's one of those things that should be straightforward, but sometimes it just isn't, especially when you're deep into a project and have a complex setup. i remember back in the day working on this huge rails project, we had like, what, 20 developers? each with their own style. it was chaos, pure, unadulterated formatting chaos. we ended up manually fixing things in code review. what a drag! lets avoid that.

let's break it down. from what i've seen, cloud9 is pretty good with some things but getting your auto-formatting just so requires a bit of a poke. first thing, are you talking about formatting on save or something more real-time? either way, it's all about configuration, that's where we need to look.

the first point of call is usually the editor's settings. cloud9 uses ace, if i remember correctly, which has some built-in capabilities, but they aren't typically hooked up directly to ruby formatting tools. to get your rails code looking consistent, we're gonna need to bring in something else. for ruby, `rubocop` is the most popular kid on the block, and the most useful. `rubocop` not only enforces style but can also automatically fix style offenses, which is what we need.

so, let's talk about `rubocop` first. you probably already have it in your project's `Gemfile`, but if you don't, that's your first step. add this and run `bundle install`

```ruby
# Gemfile
gem 'rubocop'
```

after that run `bundle install`. easy.
now, you need to configure `rubocop` for your project, if you don't have a `.rubocop.yml` in your project root folder, create it. this is where you define rules for your project, things like indentation, line length, all that good stuff. i tend to favor the community driven configuration or following some guide, its simpler this way. here is a simple example of what i use in my personal projects. it's not perfect, but it's a good starting point, and usually needs some tweaks based on your team's preferences:

```yaml
# .rubocop.yml
AllCops:
  TargetRubyVersion: 2.7 # put your ruby version here, this is an example
  Exclude:
    - 'db/**/*'
    - 'bin/**/*'
    - 'config/**/*'
    - 'test/**/*'
    - 'vendor/**/*'
    - 'spec/**/*'
    - 'node_modules/**/*' # just in case
  DisplayCopNames: true
  StyleGuide:
    EnforcedStyle: always

Layout/LineLength:
  Max: 120 # i like this number, you might prefer 80
  Exclude:
    - 'config/environments/*.rb' # some configs are ok longer

Style/Documentation:
  Enabled: false # unless you love documentation, i dont ;)

Style/FrozenStringLiteralComment:
  Enabled: true # enforce frozen strings, nice for optimization

Style/StringLiterals:
  EnforcedStyle: double_quotes

Style/TrailingCommaInArguments:
  EnforcedStyleForMultiline: comma

Style/TrailingCommaInArrayLiteral:
  EnforcedStyleForMultiline: comma

Style/TrailingCommaInHashLiteral:
  EnforcedStyleForMultiline: comma
```

this config excludes certain directories which `rubocop` is not interested in and sets a max line length, which i think is super important for readability. it also enforces double quotes, because, why not? personal preference, you can change it. but the important thing here is the inclusion of the `style` options. that's how rubocop automatically fixes your code. the `styleguide` enforces rubocop style. the last three configuration items ensure there is a trailing comma in arguments, array and hash literals when they span multiple lines.

now that you have rubocop configured, the next challenge is to hook it up to cloud9's editor. cloud9 does not come out of the box with an option to trigger the rubocop auto-formatting on save or on command. cloud9 uses ace, as i mentioned before, which is a great editor for embedded use and it has an api that allows for extensions. the solution? we are gonna create a cloud9 extension, not something difficult. this is not the most elegant solution, as we are gonna use the shell, but it's effective.

to implement this, create a javascript file, for example `cloud9_formatter.js`, inside your `~/.c9/plugins/` folder. this is the folder where cloud9 loads extensions.

```javascript
// ~/.c9/plugins/cloud9_formatter.js
define(function(require, exports, module) {
    main.consumes = ["Plugin", "menus", "command_manager", "preferences", "settings"];
    main.provides = ["cloud9_formatter"];
    return main;

    function main(options, imports, register) {
        var Plugin = imports.Plugin;
        var menus = imports.menus;
        var commandManager = imports.command_manager;
        var prefs = imports.preferences;
        var settings = imports.settings;
        var plugin = new Plugin("c9.ide.cloud9_formatter", main.consumes);

        var loaded = false;
        var isMac = /Mac/.test(navigator.userAgent);
        var formatterCommand = "rubocop -a %FILE";

        var saveFormatter = function(e) {
            var filePath = e.document.path;
            if (filePath.endsWith(".rb")) { // only format .rb files, you can extend this if needed
                commandManager.execCommand("run", {
                    command: formatterCommand.replace("%FILE", filePath),
                    showOutput: false // change this to true if you want to see the output
                });
            }
        };

        var onFileSave = function(e) {
            if (!loaded) return;
            saveFormatter(e);
        };

        plugin.on("load", function() {
            loaded = true;
             commandManager.addCommand({
                name: "format_file",
                hint: "Format current file",
                exec: function(){
                   var currentFile = require("c9/ide.editor/document").getDocument().path;
                   commandManager.execCommand("run", {
                       command: formatterCommand.replace("%FILE", currentFile),
                       showOutput: true
                   });
                }
            }, plugin);

             menus.addItemByPath("Edit/Format File", new ui.item({
                command: "format_file",
                caption: "Format File"
            }), 1000, plugin);
            
            settings.on("read", function() {
                settings.set("user/cloud9_formatter/@formatterCommand", formatterCommand);
                settings.set("user/cloud9_formatter/@macKey", isMac ? "command-alt-f" : "ctrl-alt-f");
            });
            
            settings.on("write", function(){
                formatterCommand = settings.get("user/cloud9_formatter/@formatterCommand");
                commandManager.bindKey(settings.get("user/cloud9_formatter/@macKey"), "format_file");
            });

            prefs.add({
               "Cloud 9 Formatter" : {
                  type: "heading",
                  position: 200
              },
              "Cloud 9 Formatter/Format Command" : {
                 type: "textbox",
                 position: 100,
                 width: 400,
                 value: formatterCommand,
                 onchange: function(){
                   settings.set("user/cloud9_formatter/@formatterCommand", this.value);
                 }
              },
               "Cloud 9 Formatter/Mac Key Bind" : {
                 type: "textbox",
                 position: 101,
                 width: 400,
                 value: isMac ? "command-alt-f" : "ctrl-alt-f",
                 onchange: function(){
                   settings.set("user/cloud9_formatter/@macKey", this.value);
                 }
              }

            }, plugin)
        });

        plugin.on("documentActivate", function(e){
            e.document.on("save", onFileSave, plugin);
        });

         plugin.on("documentUnload", function(e){
            if(e.document){
                e.document.off("save", onFileSave);
            }
        });


        plugin.on("unload", function() {
            loaded = false;
            commandManager.unbindKey(settings.get("user/cloud9_formatter/@macKey"), "format_file");
             menus.removeByCommand("format_file");
            commandManager.removeCommand("format_file");
        });


        plugin.load(null, register);

    }
});
```
this code does a couple of things: it defines a `saveFormatter` function which executes the `rubocop` command on your current file, whenever that file is saved, it checks if it ends in `.rb`, this is what triggers the auto-formatting. we added this function to the document `on save` event.

we also added a command called `format_file` which allows you to trigger the formatting manually using a command, and it also adds this command to the `edit` menu.

finally, we added some preferences that allow you to modify the formatter command and the key binding used to trigger the command. the default key binding is `command-alt-f` in macos or `ctrl-alt-f` in any other system.

you need to reload the cloud9 ide for this change to be recognized. after that, go to `cloud9->preferences`, you will see a new tab called `Cloud 9 Formatter` and there you can modify the settings, including your formatter command and your key binding.

note: in the preferences, do not use `ctrl + alt + f` use `ctrl-alt-f` that is important. this might be improved in the future, maybe using ace's built in formatter, but for now, i think it works ok.

so, there you have it, a basic but functional auto-formatter for your ruby code in cloud9. now, you may wonder why it was so complicated? well, because it's not that simple to add new features into a code editor, not even the "simple ones".

there is a way to make this fancier, for example by adding a loading animation while the formatter is running, but honestly for what i use and how often i use it, i don't see a benefit for doing that. you could also extend this with a watcher to show the problems `rubocop` is fixing but, again, not my cup of tea. maybe for you it would be cool to see that stuff. if you want to go deeper on how to create ide plugins you should take a look at *“Creating IDE Plugins” by Peter Friese* is a good start and if you want to go deeper with ace there is the official documentation which is good enough, but it's spread across multiple places. you can find some good books for javascript as well.

one last thing, remember, formatting is a tool, not a religion. don't get too hung up on every last little detail. i've seen developers spend more time arguing about tabs vs spaces than actually writing code, it's like, come on!

and that's it, i think this should be more than enough for what you wanted.

hope this helps!
