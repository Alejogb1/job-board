---
title: "how to cut prefix of string with sed?"
date: "2024-12-13"
id: "how-to-cut-prefix-of-string-with-sed"
---

Alright so you need to chop off the start of a string using sed right I've been there done that way too many times lets break it down and make sure you nail this thing

Okay so first things first sed short for stream editor is a real swiss army knife when it comes to text manipulation its command line power is insane and i rely on it daily not gonna lie its a crucial tool in my arsenal for anything text processing especially when scripting up automations or handling log files I mean you know a lot of people go with awk but for me sed is my go-to I've had enough fights with awk's syntax to last me a lifetime but hey thats just my personal preference

Okay enough of the rambling lets get to the meat of the matter how to actually cut that prefix from your string using sed the basic idea is to use sed's substitute command which is written like this `s/old/new/` where `old` is the text you want to replace and `new` is what you're replacing it with or if you want to delete it just leave new part empty like `s/old//` easy right

Now lets talk patterns sed uses regular expressions or regex for pattern matching which is a whole other can of worms but for prefix deletion we really only need basic stuff like the caret `^` symbol which anchors the match to the beginning of the line which is crucial for cutting prefixes or the dot `.` to match any character or `*` to match zero or more of the preceding character

Here is how I did this ages ago when I was building an automated deployment system for a company I used to work for I was handling config files with very long paths and we needed to cut the main path to keep the log files clean so there it was simple sed to the rescue

For example say you have a string like this `/opt/my_app/config/file.conf` and you want to get rid of `/opt/my_app/` here's how we can do it

```bash
echo "/opt/my_app/config/file.conf" | sed 's#^/opt/my_app/##'
```

This command outputs `config/file.conf` see how the prefix is gone the `s#old#new#` syntax i use here its the same as `s/old/new/` but I'm using a hash as the delimiter this avoids escaping problems if your prefix contains forward slashes `/` which are commonly used in paths see that that's the same thing I'm saying but if you are more comfortable using `s///` be my guest you know what works best for you

But what if you dont know the exact prefix length what if the `/opt/my_app` is something different this is where regex comes to the rescue so what I will do is match anything that goes up to the last forward slash and then replace it with nothing

Lets assume you have a string with different initial path like `/usr/local/another_app/data/info.txt` and you want to delete everything up to the last forward slash this is the command:

```bash
echo "/usr/local/another_app/data/info.txt" | sed 's#^.*/##'
```

The output here is `info.txt` magic eh so `^.*` this means any character zero or more time and then followed by `/` this will capture everything up to the last forward slash in the line and then replacing it with nothing effectively cutting the prefix that we wanted to

So when do you need to use each method? well the first one the one with the exact prefix is suitable when you know the exact text you need to remove. The second one with the regex is useful for more dynamic scenarios where the prefix length varies that is what I use in most of my cases when i deal with different paths or directory structures

Lets go a step further now let's say you have a list of files each on a separate line in a file called `file_list.txt` and you need to remove the common root path.

Lets say your `file_list.txt` looks like this:
```
/var/www/app1/public/index.html
/var/www/app1/static/styles.css
/var/www/app1/images/logo.png
```

To remove the `/var/www/app1` from all the lines you could run this on the command line:

```bash
sed 's#^/var/www/app1/##' file_list.txt
```

This would then output this in the console:
```
public/index.html
static/styles.css
images/logo.png
```

See how easy it is i once messed up this on a production server and thought I would get fired but thank god i had a backup of the file i've learned my lesson now always test your commands first before deploying to production servers or you will have a very bad time specially at 3 am

Okay I think that covers most of the common cases for prefix removal with sed. Remember the basic syntax `s/old/new/` and the use of `^` to anchor to the beginning of the line and you're pretty much set. Regex can seem scary at first but honestly you only need to know a few basic things to use it for text manipulation.

One thing you might run into is situations with different operating systems different sed versions and all sorts of oddities when dealing with bash scripting. So its always important to test your script in the environment it will be running in before going live always always always

For more in depth info on sed I highly recommend O'Reilly's "sed & awk" book its an amazing resource it has all the details you could ever need including some edge cases and advanced usage scenarios the book helped me a lot in my early days also "Mastering Regular Expressions" is another gem that explains all the intricacies of regular expressions not just in sed but in any place you use regex this helped me a lot too

So there it is I've done my best to help you solve this sed issue I really hope it helped you and good luck in your text manipulation journey don't hesitate to ask if you encounter something else
