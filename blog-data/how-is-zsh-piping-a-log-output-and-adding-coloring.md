---
title: "How is zsh piping a log output and adding coloring?"
date: "2024-12-15"
id: "how-is-zsh-piping-a-log-output-and-adding-coloring"
---

alright, so you're asking about piping log output in zsh and adding some color, huh? i've been down this road more times than i can count, and it's one of those things that seems simple at first but can get hairy pretty fast if you don't know some of the ins and outs. i remember way back when, during my early days messing around with linux, i had this ridiculously complex script to manage some server processes. i wanted to see what was going on, and the output was just a wall of text. totally unreadable. that's when i really got into figuring out how to pipe and color things in the terminal.

the basic idea with piping is to take the output of one command and send it as the input to another command. the `|` character is the pipe in zsh (and most unix-like shells). so, if you have a log file, you might do something like `cat my_log.log | some_command`. the output of cat, which is the contents of the log file, becomes the input of `some_command`.

now, adding color is where things get interesting. there are different tools you can use, but one of the most common and flexible is `sed`. sed is a stream editor, which means it works on text streams, like what you get from a pipe. you use it to find certain patterns in the text, and then replace those patterns with something else. in our case, we'll replace parts of the log with colored versions.

let's start with a simple example. say our log entries have a pattern like this: `[INFO] some message`. we can add color to the `[INFO]` part like this:

```zsh
cat my_log.log | sed 's/\[INFO\]/\e[32m& \e[0m/g'
```

let me break this down: `cat my_log.log` pipes the log file content to sed. then the `sed 's/\[INFO\]/\e[32m& \e[0m/g'` part is doing the heavy lifting.

the `s/old/new/g` is sed's replace command.  it finds instances of the `old` pattern and replaces it with the `new` pattern. the `g` at the end means "global" so replace *all* instances of the pattern in each line. we're trying to find the literal string `[INFO]` so we use `\[INFO\]` the backslashes escape the square brackets. `\e[32m` is the escape code for green, `&` stands for the matching text in this case `[INFO]`, so we preserve the original text while adding the color. we add a space for readability, `\e[0m` resets the color back to default, so the rest of the message is not green and is displayed with the standard terminal output color.

that gives you the `[INFO]` parts in green. you could modify the `\e[32m` part to use different colors, like `\e[31m` for red, `\e[33m` for yellow, and so on. if you have different log levels such as error or debug you'll need to replace other text strings using sed in the same way and giving each a different color code for readability. you'll find lists of escape codes for colors all over the internet.

now, what if you want to get a little fancier? say you have a log format like `timestamp [LEVEL] message` and you want to color code both the level and the timestamp. then you need to get a little more sophisticated with sed. lets try it with sed capturing groups.

```zsh
cat my_log.log | sed 's/^\([^ ]*\) \[\(INFO\|DEBUG\|ERROR\)\].*/\e[34m\1\e[0m [\e[32m\2\e[0m] \3/g'
```

this one is a little more involved. what i'm doing here is making use of sed capture groups, which are very helpful to process text. let me walk you through: `^\([^ ]*\) \[\(INFO\|DEBUG\|ERROR\)\].*` is the regular expression pattern. `^` means start of line. `\([^ ]*\)` captures the timestamp which is everything up to the first space, and it saves it as group 1. ` ` is just a space after the timestamp. `\[\(INFO\|DEBUG\|ERROR\)\].*` looks for the log level in square brackets. `\(INFO\|DEBUG\|ERROR\)` here I capture the log level as group 2, only capturing if the level is either INFO, DEBUG or ERROR. `.*` matches everything after the level as group 3.

then the part `\e[34m\1\e[0m [\e[32m\2\e[0m] \3` is how you replace the string using capture groups. `\1` represents group 1, which is the timestamp we color it blue with `\e[34m`. `\2` represents group 2 which is the log level, and we color it green with `\e[32m`, `\3` represents the rest of the message, which we do not color. the `\e[0m` resets the color at the end of the colored text. this will color timestamp in blue, the log level in green (if it is info, debug, or error), and leave the rest of the message with the standard terminal output color.

now, if you have a very different format it will be a different sed pattern. sed is powerful, but crafting complex regex patterns can be tricky. i've spent hours (that i won't get back) debugging sed commands, because of one missed character, or one wrongly placed parenthesis, so be mindful.

another approach you can do to use colors is by using `awk`. awk is a pattern scanning and processing language, which is very powerful to work with columns and patterns, for instance. here's a similar example using `awk` which can be clearer to some people.

```zsh
cat my_log.log | awk '{
  if ($2 ~ /\[INFO\]/) {
    printf "\e[34m%s\e[0m [\e[32m%s\e[0m] %s\n", $1, substr($2, 2, length($2)-2), $3;
  } else if ($2 ~ /\[DEBUG\]/) {
    printf "\e[34m%s\e[0m [\e[33m%s\e[0m] %s\n", $1, substr($2, 2, length($2)-2), $3;
  } else if ($2 ~ /\[ERROR\]/) {
    printf "\e[34m%s\e[0m [\e[31m%s\e[0m] %s\n", $1, substr($2, 2, length($2)-2), $3;
  } else {
    print $0;
  }
}'
```

in this awk snippet, awk by default splits each line into fields separated by spaces, you can change this behavior if your log does not use spaces to separate. so `$1` refers to the first field, the timestamp, `$2` refers to the second field, the log level, and `$3` refers to the rest of the line. the `if ($2 ~ /\[INFO\]/)` checks if the level is `[INFO]`. if so, it formats the output using `printf` and adds colors, `substr` is used to remove the brackets for printing. it's the same concept as before, with different escape color codes for each level. if a log line does not have the log level we expect it just prints the line as it is, in default color.

there are other tools you could use for this, too. `grep` can be helpful for filtering log lines before coloring them with sed or awk. for example if you wanted to only color lines containing the text error you can do something like: `grep "ERROR" my_log.log | sed ...` or you can combine different tools in the same way. there's also `less`, which can have its own color support using the `less -R` option and specific environment variables. in addition `multitail` can be useful for following multiple log files and adding colors. also specialized log analysis tools might have built-in color-coding, but I tend to go with the command line tools since they're more portable and flexible for my needs. you also have `ccze`, which is designed to make log output colorful. but i tend to try to avoid adding dependencies to scripts if I can avoid it, as that usually adds overhead and introduces more possibilities for failure, plus doing it with command line utilities give me more flexibility.

a tip i've found really helpful is to create a zsh function, so you can reuse your colorizing command without typing the whole thing again. you can put this in your `.zshrc` file:

```zsh
colorlog() {
  cat "$1" | sed 's/^\([^ ]*\) \[\(INFO\|DEBUG\|ERROR\)\].*/\e[34m\1\e[0m [\e[32m\2\e[0m] \3/g'
}
```

and then you can simply run `colorlog my_log.log` from then on.

for resources, i would recommend checking out books or tutorials that focus on command line tools, `sed` and `awk`. there are a lot of books that explain regular expressions well too since they are used often in all of these. those are often very good to understand all the details of regex that these tools require. the manual pages (man pages) for sed, awk, and zsh are always good references too, if you are into that. i personally find more easy to check some quick explanation of the basics of these tools online and come back to the manual pages if i need more specific information. personally i enjoy more learning these by doing, by doing real text manipulation with them. that is how i really learnt, by trying to solve my own text manipulation problems.

i remember this one time, i spent three hours debugging a complicated sed command, only to realize i had an extra space in my regex. after banging my head against the wall i almost gave up, but then i saw the space, laughed a little, and fixed it. it was a good day...

anyway, hope that helps. just try different things, experiment, and see what works best for you. it's all part of the process.
