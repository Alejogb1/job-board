---
title: "p4 submit without description?"
date: "2024-12-13"
id: "p4-submit-without-description"
---

Alright so p4 submit without description right Been there done that got the t-shirt and probably a few sleepless nights to go with it This is a classic Perforce problem a real head scratcher for the uninitiated and honestly even for the veteran sometimes Let me tell you about my first rodeo with this one back in oh maybe 2012 working on a ridiculously large game engine project We were using Perforce like it was going out of style which it kinda was back then but thats another story And of course we hit this exact issue people were submitting changes without descriptions just a bunch of changelists with numbers and no context Absolute chaos It was like trying to decipher hieroglyphics after a power surge

So the problem is that p4 submit doesnt *force* a description It's like a suggestion box that's never actually read You can totally submit a changelist with a blank description and the system just shrugs and says "okay sure" This is a recipe for disaster in any team bigger than like two people It makes code reviews impossible finding bugs is a nightmare and trying to trace the history of a change is like navigating a maze blindfolded

The root cause usually comes down to a few things either the developers are lazy in a hurry or just not aware of the importance of a good description Or sometimes its because the p4 client is set up poorly and doesnt nudge them in the right direction Which happened way too often with me

First things first lets look at the simple way to see the problem you just run `p4 changes -l` and you will see a whole bunch of changes without a description or with the infamous `default` description you probably know it from the nightmares of your job This is usually a first sign something is seriously wrong

The basic solution is to make it so that developers *have* to enter a description before submitting Their submit will just fail if they dont It is achievable using Perforce's triggers functionality Triggers are server-side scripts that are executed at different points in the perforce workflow think of it like git hooks on the server level So when a user tries to submit a changelist the trigger gets executed and we can check the changelist description and reject it if it is empty or has a default value.

Lets dive into some code shall we I'll show you a couple of ways to do this The first one is with a simple bash script and then I’ll show you another approach with a python script

**Bash Script Example**

This bash script checks if the change description is empty if it is or if it is default it exits with an error preventing submission

```bash
#!/bin/bash

CHANGELIST="$1"

DESCRIPTION=$(p4 change -o "$CHANGELIST" | grep Description: | sed 's/Description:\s*//')


if [ -z "$DESCRIPTION" ] || [ "$DESCRIPTION" == "default" ]; then
  echo "Error: Change description cannot be empty or 'default'." >&2
  exit 1
fi

exit 0
```
This script is relatively basic but it gets the job done. Save this to a file called for example `check_description.sh` You will need to make it executable with `chmod +x check_description.sh` and you will then need to configure a Perforce trigger to use it. This command will do the trick for most cases:

```
p4 triggers "Triggers:
    change-submit    change-submit    //...   \"/path/to/check_description.sh %change%\""
```

Now after that every single submission will trigger this check and will prevent submission without a proper description. Of course this only works in Linux servers or with a linux based shell environment in your servers.

**Python Script Example**

Okay now for something a little more flexible a Python script We can do a lot more complex checks if we need them in the future We could filter different default descriptions or check for a minimum characters in the description It is way more flexible than shell scripts for complex issues

```python
#!/usr/bin/env python3

import sys
import subprocess

def get_change_description(changelist):
    try:
        process = subprocess.run(['p4', 'change', '-o', str(changelist)],
                                 capture_output=True, text=True, check=True)
        output = process.stdout
        for line in output.splitlines():
            if line.startswith('Description:'):
                return line.split(':', 1)[1].strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running p4 command: {e}", file=sys.stderr)
        sys.exit(1)
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: check_description.py <changelist>", file=sys.stderr)
        sys.exit(1)
    changelist = sys.argv[1]
    description = get_change_description(changelist)
    if not description or description.lower() == "default" or not description.strip():
        print("Error: Change description cannot be empty or 'default'.", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
```

Save this to a file lets say `check_description.py` Again we make it executable with `chmod +x check_description.py`. And again we set up the triggers using the command below (of course you need to adjust the script path):

```
p4 triggers "Triggers:
    change-submit    change-submit    //...   \"/path/to/check_description.py %change%\""
```

The python script here gives you some advantages like you can now add even more advanced checks it can be as simple as length of description or more complex things such as specific formats required for the descriptions it also has a bit of error control and it’s easier to debug if something goes wrong

Now a word of caution I know its tempting to add all sorts of rules to force developers to do things your way, but remember the more strict you get the more friction you generate. It is a balance between enforcing good practices and not making everyone’s life miserable Remember the golden rule, developer happiness is directly proportional to productivity. And a happy developer is a productive developer. If your scripts are too invasive they might start submitting the first line of code in the description or any type of trick to make the script go faster

Another thing I’ve seen before is that some teams use templates for their commit messages This can be enforced with a trigger that checks for the template in the description This way you can make sure that your developers are following the same rules when describing their changes Its generally a good idea to enforce code styles and commit styles to have a better track record of what everyone is doing and to keep a standard across the board

**Making it User-Friendly**

Okay beyond triggers its also very useful to make the p4 user experience friendlier. P4V the Perforce Visual Client, has a feature to customize the submit dialog This can be used to show a warning if the description is blank or even pre-fill it with some boiler-plate stuff It’s a good idea to set up client side warnings, because they are far more efficient than server-side rejections. It is better to prevent the problem before it reaches the server.

Another thing that I would advise is to have a good documentation for p4 this might seem basic but you will be surprised by how little developers know of their versioning system most people just go on the surface level of their system and rarely dive deep into the configurations and tools they have. If your team knows all the features they will be more efficient and more effective. A good introduction to Perforce is the O'Reilly book on "Version Control with Perforce" it’s a classic for a reason. Also make sure to go through the official Perforce documentation which is really good

So yeah, that's pretty much it. Dealing with empty p4 descriptions is a common problem, but it's totally solvable with a bit of know-how and some good old-fashioned scripting. Remember to balance enforcement with usability and try not to make developers hate the system, it's the system you should hate the bugs in the code not the tools you use for it.
