---
title: "line editing with zshs vared from within script?"
date: "2024-12-13"
id: "line-editing-with-zshs-vared-from-within-script"
---

Alright so line editing with `zsh`'s `vared` from inside a script right I've been there done that got the t-shirt probably several t-shirts actually This stuff can be a bit of a rabbit hole if you're not careful Let me spill the beans on what I've learned over the years dealing with this kind of thing it's not always straightforward let me tell you

First off `vared` is your friend when you need interactive editing of a variable in `zsh` It's like a mini text editor popping up in your terminal Now the catch is it's designed for interactive use which means it plays nice when you the human are at the keyboard typing away But when you try to use it from a script things can get a little funky

The core problem I ran into many moons ago was this how do you simulate that human interaction that `vared` expects when your script is running non-interactively The issue isn't `vared` itself it's how it's coupled to the terminal's input stream When a script runs there isn't a terminal for `vared` to hook into That's where the trouble starts So first things first we can't directly use `vared` in a non-interactive shell session That's a given like the earth being round or the internet needing more cat videos

Now let's talk about workarounds or rather the approaches I've found semi-useful I say semi-useful because they all come with their baggage they aren't the perfect solution but sometimes in the tech world perfect isn't the goal it's 'good enough'

One method I experimented with early on was using `print -z` in combination with `read` to mimic the line editing behavior It's not as fancy as `vared` but it gets the job done for simple cases You're effectively stuffing the initial value into the input buffer and then using `read` to get the possibly modified input back Let me show you what I mean with some code

```zsh
initial_value="This is the default text"
print -z "$initial_value"
read modified_value
echo "Modified value: $modified_value"
```

Okay so this snippet it's kinda like a barebones version of line editing You get the `initial_value` pre-populated you can edit it using your terminal's standard line-editing keys like arrow keys backspace etc and then `read` captures the result Now the problem is this method relies on the underlying shell's line-editing capabilities and if you are running on a headless server where some shell weirdness happens or just the term env it can be a mess

Another tactic and I know you don't want some long-winded explanation on the underlying mechanics so let's call it a "less naive approach" involved using named pipes or fifos This was after the first naive method bit me a few times Let's say you have your script and you need to edit a string you can create a named pipe pass the string into it then run vared on that pipe Here is what i mean:

```zsh
  initial_value="Text to edit"
  fifo_name=$(mktemp -u)
  mkfifo "$fifo_name"
  {
  echo "$initial_value" > "$fifo_name"
  vared modified_value < "$fifo_name"
  rm "$fifo_name"
  echo "Modified value: $modified_value"
  } &
wait
```

So in this example a fifo is made with `mkfifo` then the initial value is piped into it the vared command will edit the string from the fifo and the value is assigned back to the modified variable and finally the fifo is removed This works because vared reads from standard input and when a pipe is used the input can be redirected You might ask where i got that silly idea well let me tell you i was working on some shell script to provision a machine and i needed to edit some conf files within the script in some cases where manual intervention was needed

But again it's not as good as a interactive shell because if you are not running from a terminal it will just do nothing it just hangs and the same applies if you are connected to the machine using something like ssh but using a different terminal then the terminal that the shell runs on This is where it starts to get annoying and you get into some shell scripting black magic if you want to get this done 'right' let me explain

And the last thing I'll mention because this is really starting to get long is an approach that gets into some more complicated territory its using coprocesses and here is some example code:

```zsh
initial_value="Yet another line of text"
coproc EDITOR {
  echo "$initial_value"
  read edited_value
  echo "$edited_value" >&3
}
print -z "$initial_value"
read edited_value_coproc <&"${EDITOR[0]}" >&"${EDITOR[1]}"
read -u 3 modified_value
echo "Modified value from coprocess: $modified_value"
kill $EDITOR_PID
```

What's going on here is that you start a process that will be used as a sub editor to read what we initially passed to it and then output the edited version to the fd 3 we then pass the initial value to the sub editor and we capture the output of the sub editor on the read from fd3 and then finally we print it But even this method has some issues. The shell might not give back the correct file descriptors and it's not easily portable between various shells and operating systems

Now before you ask for some magical library that makes this seamless it doesn't exist I mean there might be some hacks floating around some github repos but those are more likely to break than to work properly this is not a normal use-case and shell scripting is not the best place to be doing complex logic like this it is useful for orchestration and configuration but not full-fledged interactive user experiences

The key takeaway here is that `vared` is for interactive editing and trying to shoehorn it into a non-interactive script is like trying to fit a square peg into a round hole You can make it work with a lot of effort and some questionable tactics but you might be better off considering other solutions depending on your goals

If your goal is to get input from a user during a script consider using things like dialog or zenity those tools are meant to build simple UI dialogs from scripts they are well supported and widely available you might find that to be a better solution than messing with `vared` or similar tools for input this is a good general advice

Now if you really must make `vared` work you are in for a very painful experience if you are in a non standard environment with weird shell setups but that's life in tech there is always some corner case that will break your brain a little bit. If you want to learn more about shell scripting in general I would recommend "Classic Shell Scripting" by Arnold Robbins and Nelson H. F. Beebe that covers more about the complexities of shells that are usually ignored like file descriptors and process management. For more specific `zsh` tips and tricks the `zsh` man page is a goldmine it's huge but it has everything and i mean everything

So that's my two cents on the `vared` in non-interactive scripts situation it's not easy it's not pretty but it's sometimes necessary Happy scripting and may the shell be ever in your favor Also i had this one funny incident where i thought the issue was something related to a file descriptor that was reused and was reading junk but it was just that the value i was printing was not what i expected it was a simple typo the good old typo bug.
