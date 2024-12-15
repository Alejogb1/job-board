---
title: "How to zsh piping a log output and adding coloring?"
date: "2024-12-15"
id: "how-to-zsh-piping-a-log-output-and-adding-coloring"
---

alright, so you're looking to pipe log output in zsh and get some color action going. i've definitely been down this road before, it’s a common thing when you’re staring at walls of text and need to parse things more easily. i remember my first serious run-in with this, it was back when i was working on this distributed system. the logs were, let's just say, verbose. monochrome logs were like trying to navigate a maze blindfolded. so yeah, i get the need.

the core issue here is that standard output often just spits out raw text, and we need something to interpret and add those lovely ansi color codes. there isn't a magic command built into zsh to do this all at once, it's more about combining tools we already have available, and understanding how they chain together. think of it like a pipeline in a factory where each stage takes the raw material and adds to it to create a final product.

let’s talk about a basic way to colorize output. one of the main workhorses here is `grep` along with its `--color=auto` option. this lets `grep` automatically add color depending on whether it's outputting to a terminal or not. so the basic idea is you pipe your log output to grep, and then grep adds the colors when it finds a match. this might sound simplistic, but it gets you somewhere quickly.

for example, let’s say you have a log file named `my_app.log` and you want to highlight all lines containing “error”. you would use something like this:

```zsh
cat my_app.log | grep --color=auto "error"
```

that’s just `cat`ting the file and sending it to `grep`, that then highlights the matches.

pretty straightforward, but what if we need to do something more sophisticated? what if we wanted to colorize different parts of each log line differently, and not just the text containing errors? that's when you will probably need something that is specifically geared for log parsing and colorization. that's where things like `sed` or `awk` enter the picture, along with a bit of ansi color knowledge. ansi color codes are basically escape sequences that terminals understand, they represent stuff like red, green, bold, or underline text. this is where it starts to get more fun.

for example, imagine your logs have this general format:

`[timestamp] [level] [message]`

you might have something like:

`[2024-10-27 10:00:00] [info] server started`

and you might want to color code the level differently based on whether it is info, error or debug, and leave the rest on the normal color. so you could use something like `sed` to add those ansi escape codes in specific places. it could look something like this:

```zsh
cat my_app.log | sed -E 's/\[(info)\]/\x1b[32m[info]\x1b[0m/g; s/\[(error)\]/\x1b[31m[error]\x1b[0m/g; s/\[(debug)\]/\x1b[33m[debug]\x1b[0m/g'
```

let me break that down:

*   `cat my_app.log` - this spits out all log entries.
*   `|` - pipes this output to the next command.
*   `sed -E` - starts sed with extended regular expressions enabled.
*   `s/\[(info)\]/\x1b[32m[info]\x1b[0m/g` - a substitution command that finds `[info]` (escaped brackets with `\` because they have special meaning in regular expressions), and replaces it with `\x1b[32m[info]\x1b[0m`. here `\x1b[32m` is the ansi code for green, and `\x1b[0m` resets the color to default. the `g` flag indicates that this substitution should be performed globally (all occurrences in the line)
*   the rest are similar but for error and debug with red and yellow colors respectively.

so that would take the log file and replace all `[info]` occurrences with green color. all `[error]` occurrences with red and all `[debug]` with yellow, and you now got a nicely colored log.

now, if you're working on something seriously complex, you would might want to look at dedicated log analysis tools, but before diving deep into that, `awk` can also be useful here. it allows you to split each line into fields, and then you can perform actions based on those fields. `awk` is a complete language in itself, so it can handle things more efficiently. i once had to parse logs for different types of transactions and awk really helped, it was amazing.

here’s a slightly more advanced example that leverages `awk` to colorize log levels, this is a bit more complex than the `sed` example, but shows the power of `awk`:

```zsh
cat my_app.log | awk '{
  level = $3;
  if (level == "[info]") {
    printf "\x1b[32m%s %s %s\x1b[0m\n", $1, $2, $3, substr($0, length($1) + length($2) + length($3) + 4)
  } else if (level == "[error]") {
    printf "\x1b[31m%s %s %s\x1b[0m\n", $1, $2, $3, substr($0, length($1) + length($2) + length($3) + 4)
  } else if (level == "[debug]") {
    printf "\x1b[33m%s %s %s\x1b[0m\n", $1, $2, $3, substr($0, length($1) + length($2) + length($3) + 4)
   } else {
     print
  }
}'
```

let's break this `awk` magic down:

*   `cat my_app.log | awk '{ ... }'` - same as before, takes our logs and feeds it to `awk`.
*   `level = $3;` - this assigns the third field of each line to a variable named `level`.
*   `if (level == "[info]") { ... }` - checks the level and prints the line with green color using `printf`. `substr($0, length($1) + length($2) + length($3) + 4)` is used to extract the rest of the line, it basically takes all from the 4th field and beyond.
*   same happens for `error` and `debug`, coloring red and yellow.
*   the `else {print}` at the end handles the cases where it did not match any of our colorized levels.

the important thing here is that instead of matching whole lines, we are matching fields, which makes things more flexible.

regarding more structured logging, if you find yourself in need of more sophisticated log analysis, you should definitely look into tools like `lnav` or the `elastic stack` (with logstash). they are more powerful and capable of doing full blown analysis, not just colorizing the output. `lnav` in particular is pretty cool because it actually parses log structures and lets you do things like sql queries against the log data. but for quick and dirty log analysis, combining `grep`, `sed` and `awk` is usually good enough. also, you may want to explore `less` with color escape codes, which provides better ways to navigate large files. there are many online tutorials about that.

so yeah, that’s the gist of it, start simple, and then gradually add complexity as your needs become clearer. there is not a single tool for all, but a combination of things depending on the level of analysis you need.

about resources, instead of sending you a bunch of links that probably wont work in two days, i recommend looking at “the awk programming language” by aho, kernighan and weinberger, it's the bible for `awk`. also, the `sed` manual page (`man sed`) is actually a quite good resource once you get used to it. for ansi escape codes, there is a good summary on wikipedia if you want to dive deeper. it contains all the color codes, text effects and so on. if i have to give you one advice, it's, start with `grep --color=auto` and then evolve from there. it saves headaches and it's usually good enough for a first pass.

oh, and here is a tech joke for you. why did the programmer quit his job? because he didn't get arrays. just like those colors you want, they come in arrays.
