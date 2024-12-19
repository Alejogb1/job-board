---
title: "How to format C# 'else' onto one line in Rider?"
date: "2024-12-15"
id: "how-to-format-c-else-onto-one-line-in-rider"
---

so, you're having trouble getting rider to format your c# `else` statements onto one line, right? i’ve been there. trust me, the struggle is real when you’re trying to maintain a particular code style and your ide has other plans.

i've spent a fair amount of time battling with rider's formatting engine, and specifically on this particular issue of `else` placement. it's one of those seemingly minor things that can really throw a wrench in your workflow if you're picky about consistency. it can be particularly frustrating when you’ve got a coding standard in your team where compact `if-else` structures are preferred.

let me recount a personal experience with this particular problem. back in my early days, when i was involved in building a fairly complex backend application, we adopted a coding standard that emphasized brevity and readability, particularly for simple conditional statements. we'd been using a different ide at that time, and formatting the `else` statement on the same line as the closing brace of the preceding `if` block was a very common practice to save a line of vertical space (we were young and obsessed with code density!). then we moved to rider (mostly due to its awesome debugging capabilities) and all hell broke loose in the formatting department. our neatly formatted `if-else` constructs were suddenly sprawling across multiple lines.

it wasn’t that the code was wrong; it was just, well, not how we liked it. it made code reviews a pain. i spent a good chunk of a weekend (much to my wife’s displeasure) tweaking rider's settings to get it behave the way i wanted.

the key here is understanding rider’s code formatting configuration. it’s incredibly powerful once you figure it out, but it can feel like navigating a labyrinth at times. it’s not simply a matter of clicking one checkbox, unfortunately. it involves diving into the preferences and finding the correct setting, which, and i say this with no exaggeration, feels hidden in plain sight sometimes.

anyway, let’s cut to the chase. here's how you can get rider to put your `else` statements on one line.

first, you need to go into rider's settings/preferences dialog. this is generally under `file > settings` on windows or `rider > preferences` on macOS. then, you navigate to `editor > code style > c#`. this is where the magic (or the madness) happens.

look for the "wrapping and braces" tab. within that tab, there are sections for various code constructs. the one you're interested in is the "if statement" section. in this section, you might find that rider defaults to wrapping the `else` statement onto a new line. there should be an option there called something like “place else on new line”. make sure that checkbox is unchecked. it should handle putting `else` on one line.

after this, the general formatting behavior for `if` statements within rider should adhere to your desired format. now, if we’re talking about a more complex nested condition, i can tell you that this option will still output on one line, it’s not recursive or anything, but in most cases, this setting should solve most of your issues with the else placement.

however, bear in mind that rider uses different formatting options for different code blocks and code constructions. if for any reason you're experiencing issues, i would recommend reading about rider's code style configuration in jetbrains rider's official help pages. their documentation is excellent and provides deep insights into all of its available options.

let me show some code examples. imagine you had code that looks like this:

```csharp
// before
if (someCondition)
{
    DoSomething();
}
else
{
    DoSomethingElse();
}
```

after applying the rider configuration setting mentioned above, the formatting would change to:

```csharp
// after
if (someCondition) {
    DoSomething();
} else {
    DoSomethingElse();
}
```

and if you have a complex conditional with nested `if` statements that might also include an `else` condition:

```csharp
// complex before
if (someCondition) {
    if (anotherCondition) {
         DoAnotherThing();
    }
    else
    {
        doYetAnotherThing();
    }
} else {
    DoTheLastThing();
}

```

after the rider formatting, your code should output something like:

```csharp
// complex after
if (someCondition) {
    if (anotherCondition) {
         DoAnotherThing();
    } else {
        doYetAnotherThing();
    }
} else {
    DoTheLastThing();
}
```

and a very simple if else:

```csharp
// very simple before
if(a == b){
  var c = 1;
} else {
  var c = 2;
}

```

should become:

```csharp
// very simple after
if(a == b){
  var c = 1;
} else {
  var c = 2;
}
```

as you can see, the setting does not go recursively in the code, you need to apply this to all cases, it’s a simple configuration.

one very very important detail to keep in mind is that rider uses *schemes* to manage and share code styles across teams. it allows you to use different styles for different projects, which is quite powerful. you can export and import these schemes. this is extremely helpful when your entire team uses rider or if you need a specific coding style for a specific project. it allows the entire team to work using a standard code format without any major issues on the way. imagine if one of your team members has not applied the configuration and commits code with a different format!

in addition to rider's documentation, i recommend looking at "clean code" by robert c. martin; it’s a classic and really talks about readability and maintainability. while not directly about ide configuration, it will help you understand the rationale behind certain formatting choices and their impact on code quality and developer collaboration. another book that i’d recommend, more in line with the C# language itself is “effective C#” by bill wagner; although it does not cover code formatting, it goes deep into best practices for c# usage, which in the end affects how you write the code in the first place.

let me tell you, sometimes i feel like my code is so perfect i think i should publish it in a museum (just kidding!).

i hope this was helpful. it’s something i’ve had experience with, so i understand your frustration. just keep experimenting with rider's options and you will achieve the desired code formatting style!
