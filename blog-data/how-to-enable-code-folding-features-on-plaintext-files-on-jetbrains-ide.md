---
title: "How to enable code folding features on plaintext files on JetBrains IDE?"
date: "2024-12-14"
id: "how-to-enable-code-folding-features-on-plaintext-files-on-jetbrains-ide"
---

alright, so you're after code folding in plaintext files on a jetbrains ide, huh? i’ve been there. it's one of those things that seems trivial until you’re staring at a 500-line config file, or a sprawling log, and your brain just wants to implode. the struggle is real. i remember back in the day when i was first starting out. i was working on a particularly gnarly project. some old mainframe system configuration i think it was, and these files… oh man, they were just walls of text. no structure, nothing. i was scrolling, scrolling and lost my context every two minutes. it was terrible. then one day, a colleague (she was a seasoned systems admin), showed me this feature, and it changed everything. it felt like moving from a cave to a penthouse. it’s amazing what a bit of folding can do.

so, out of the box, jetbrains ide’s usually don’t have folding for plain text. they're optimized for code, naturally, where syntax helps determine the folding blocks. but we can definitely make it happen. the key is to define custom folding rules. think of it as telling the ide, "hey, when you see *this*, treat it as a collapsible section”. this relies on regular expressions, or regex, which many software developers either love or hate. i, personally, have a soft spot for them. they’re like tiny programming languages in themselves. if you don’t have familiarity with them consider taking some time to understand them better, i recommend the book "mastering regular expressions" by jeffrey friedl. this has been a lifesaver for me in several occasions.

let’s get into it. you'll need to access the 'editor' settings in your jetbrains ide and then navigate to 'code folding' section, you can normally find this in ‘settings/preferences > editor > code folding’. there you’ll find a section named "custom folding regions". this is our playground. you'll be adding new rules that tell the ide where to start and end the folds. these rules consist of two parts; the start and end regex patterns.

for example, suppose you have a config file where sections are marked by `[section_name]` and end before the next `[section_name]`. the start pattern is `^\[[^\]]+\]$` this will match any string starting with a `[` and ending with a `]` and in the middle, it can accept one or more characters that are not a `]`. the end pattern should be the same, but since the end of the section is not obvious until the next one starts, we can reuse the start pattern here. this works in this particular scenario but in others, you might need a different pattern.

here is an example of what the setup looks like in the settings pane:

```
start pattern: ^\[[^\]]+\]$
end pattern:   ^\[[^\]]+\]$
```

apply and check if this is already working. if not, restart the ide just in case. if this works you’ll see that when there are two or more of these patterns on a document that you can collapse the content between them.

now, let's say you're working with a log file, and you want to fold sections based on timestamps that use this kind of structure yyyy-mm-dd hh:mm:ss. that looks like: `2024-10-27 10:30:00` or something like that. you could have a log file similar to this:

```text
2024-10-27 10:00:00 [info] system starting up
some system logs here
2024-10-27 10:05:00 [info] database connected
more database logs
2024-10-27 10:10:00 [warn] low disk space
disk space warning
2024-10-27 10:15:00 [error] network error
network stuff
2024-10-27 10:20:00 [info] system operational
```

in this case the start pattern would be something like `^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*$` and the end would be the same. let me explain a bit what this pattern does. it is searching for strings that begin with a date in the format `yyyy-mm-dd`, then a space then another timestamp with the format `hh:mm:ss` and after that zero or more characters.

here is the example for this scenario:

```
start pattern: ^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*$
end pattern:   ^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*$
```

now, if you reload the log file (close and re-open) you should be able to fold by the timestamps. nice and neat, right?

you can really go nuts with this stuff. if you wanted to, you could set up folding patterns for markdown, or even code blocks if you are opening a source file as plain text. lets imagine you have a markdown document where headings start with `#` symbols. i have dealt with several scenarios like this. lets say you have a file like this:

```markdown
# main title
some paragraph
## subtitle
another paragraph
### sub-subtitle
and another paragraph
# main title 2
some more content
```

in this scenario we could do something like this, `^#+ .+$` for both start and end patterns this means match one or more `#` symbols, followed by a space and then one or more character until the end of the line. here is the example for the configuration:

```
start pattern: ^#+ .+$
end pattern:   ^#+ .+$
```

the same applies here, you might need to close and re-open the document to see changes in the folding mechanism.

a little trick i learned the hard way when i was implementing something similar is to avoid patterns that are too broad; if your start and end patterns are too generic it might fold some sections that you might not want folded so try to be as specific as possible.

one thing that's also useful is to have the folding description shown in the gutter. this is not code per-se, but it helps a lot. i mean, seeing the starting line of the folded area provides context and you don’t need to open it to see what is about. go to ‘settings/preferences > editor > general > code folding’ and check the option ‘show folding outline’ and enable the ‘show folding descriptions’ it is normally at the bottom.

let me be honest with you, this might seem a little bit tricky to setup at the beginning. especially if you are not very comfortable with regex. but once you get a hang of it, it's a real time-saver. you’ll wonder how you lived without it.

oh, and here is a little joke for you. i asked chatgpt to debug my folding rules... and it gave me regex to match my socks! i mean, it was close, but not quite.

if you want to dig further into the subject of code folding in general the article "efficient code folding algorithms: a survey" from 2015 gives an in-depth view of this problem. its a good read if you want a deeper dive.

remember to save your settings after you add or modify the folding rules, and sometimes a restart of the ide is needed. it's always a good habit anyway.

i hope this helps, let me know if you run into any issues. we’ve all been there. good luck with your text wrangling.
