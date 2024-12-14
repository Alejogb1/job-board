---
title: "How can I change a JetBrains Applications new function formatting?"
date: "2024-12-14"
id: "how-can-i-change-a-jetbrains-applications-new-function-formatting"
---

alright, so you're looking to tweak how jetbrains ide's format your code when you're, say, creating a new function. i get it. the defaults are… well, they're defaults, and sometimes they just don't mesh with your personal style or your team's coding standards. i’ve been there, done that, got the t-shirt – and probably a few mental scars from battling with auto-formatting over the years. let me tell you, it's a pain when the machine thinks it knows better than you do about code aesthetics.

when i first started messing around with intellij idea (and the rest of the jetbrains family), i had this exact issue. i was used to a very particular way of laying out my functions – specific indentation, where the curly braces went, the spacing around arguments, that sort of thing. after using visual studio for ages, i got really frustrated with how intellij took a go at formatting things i created with the new function assistant. my stuff always looked completely off, even if i created the function from scratch and tried to make it look like i wanted it to. it's like it was mocking my formatting attempts. back then, i was working on a fairly large java project, and consistency was key. it was just maddening to look at how intellij kept changing my function declarations. i ended up spending a ton of time manually reformatting code or turning off the autoformatter completely for a while and that's really not ideal, because it helps in the long run.

the good news is, jetbrains ide's are actually really configurable when it comes to code style. it’s just a matter of finding the specific settings you need to tweak. it's not exactly front-and-center, but it's there. the key is the “code style” settings which are specific to each language you use and then specific to each project you work on or you can configure them for all projects if you want.

so, let's talk specifics. the way to get this under control is by heading into the settings dialog (usually ctrl+alt+s or cmd+, on mac). then, navigate to editor -> code style. you'll see a list of languages on the left. you'll need to select the language you're working with (e.g., java, python, javascript, kotlin, etc).

within each language, you have a ton of options for controlling code formatting. now, i won't cover every single setting because there's a lot, but let me show you some of the common ones you'll probably want to adjust, especially for function formatting.

first off, the “tabs and indents” tab is crucial. here, you can set your tab size, the indent size, whether to use tabs or spaces, and how to handle continuation indents. for function declarations, i usually recommend keeping "use tab character" unchecked and use “indent” and “continuation indent” with the same values because you do not want to use tabs for coding these days, you do not want to fall into this trap.

then, go to the “wrapping and braces” tab which is the real juicy part for function declaration formatting. there's a whole section called something like "function declaration" (or method declaration if you are using java), and that is where you will find some controls to put the braces in the new line, or if the braces go to the same line, if the arguments must be placed in one line or with one argument per line. the same for the parameters and in general the whole signature.

here's what you might be looking to adjust there:

*   **'method/function declaration parameter/argument placement':** this is super important. it's how the ide will format the parameters/arguments of your functions. you can choose things like "do not wrap", "wrap if long", "one per line", or "chop if long". “one per line” is my personal preference for more readable code but if the arguments are few you can put them all in the same line. it really depends on your or your team´s coding style.
*   **'braces placement':** this is where you decide if your opening curly brace goes on the same line as the function declaration or on a new line. i know it seems trivial but this is a very old war between programmers, it is the "one true brace style" or "k&r" vs "allman" brace styles. there's no single way to format code, it's whatever is consistent within your project and you want, but consistency is key for legibility and avoiding merge conflicts.
*   **'space around parentheses':** it controls if the ide puts a space between the parentheses and the function name/arguments. some people prefer `function()` and some like `function ()`. you get the drill. you can also use "spaces within parentheses" or "spaces within method call parentheses" which refers to how to format the call of a function.
*   **'keep when reformatting':** some sections are to control what formatting configurations can be changed or not, you should explore all of them and see what is helpful for your style.

now, it's not just about the visual aspect, but also what the ide does when you add arguments, or if the function has a return type. here are a few configurations that help you control that:

*   **'method/function declaration return type':** specifies how it should behave when the return type is a type variable or a complex type in general, in which line to put it for better legibility, or if you want it on the same line.
*   **'method/function declaration modifiers':** this is a small detail, but really handy, here is how you can control the modifiers `public, static, final, abstract` and so on. if you want them in a separate line or not. if it's more than two modifiers it’s recommended to put them in separate lines in most cases but you can do whatever you find more readable.

let's see how some code would look with different configurations, as a starting point for inspiration. here are some examples:

example 1: using one line and aligned braces:

```java
public String myfunction(int arg1, String arg2) {
  return "hello " + arg2 + arg1;
}
```

example 2: using one argument per line, and different indentation for function parameters:

```java
public String myfunction(
    int arg1,
    String arg2
)
{
  return "hello " + arg2 + arg1;
}
```

example 3: using allman style braces and wrapped parameters:

```javascript
function myfunction(
    arg1, 
    arg2
) {
  return `hello ${arg2} ${arg1}`
}
```

these are just examples, there are many other combinations you can use depending on the language you're working on. and the best part, it’s super easy to try things out. you can play around with the settings and hit apply, and see how your code gets re-formatted in real-time in the editor. very handy.

i have spent more than one night battling to configure specific aspects like handling annotations in java or handling imports, they all have specific configurations and they are also worth exploring to understand how you can get control of the ide in your favour.

now, i know that sometimes digging through all those settings can be a bit much. it feels like you need a programming degree to configure a simple text editor. but trust me, once you've got your formatting dialed in, it's a huge time saver and keeps your codebase more readable. and that is a great improvement in terms of productivity and code legibility, not to mention saving you a lot of time during code reviews.

there's no single "perfect" style, it's about finding what works best for you and your team. if you're working on a project with other people, i highly recommend having a discussion with the team and agree on a single style and then create a configuration file that everyone can import. i've been in the middle of code wars because of styling differences and trust me, it's not fun. you should avoid this by having an agreed code style guideline or rule.

it would take me longer than a day to explain every single option that jetbrains offers but i recommend you to check “effective java” by joshua bloch which is a really solid resource for coding style guidance, although it is focused in java, many of the principles can be applied to other languages. "clean code" by robert c. martin is another great resource for good practices not only in code style, but in general, and in all kinds of languages. you can also look for your coding language official code style guidelines, they all have very complete style guidelines, that you could adapt to your personal preference. the most important part is to have rules and follow them rigorously.

one last thing, if you ever need to share your code style settings, you can export them from the “code style” settings as an xml file. this is handy for quickly importing them on a new computer or for sharing them with the rest of the team. it avoids having to make changes manually on every single setup you have. you can also export different configuration profiles with specific configurations. you can have one profile for your current project, another one for older projects or another one for your personal projects. it depends on how you want to organize your work environment.

one thing to keep in mind, the ide is just a tool. it's not the boss of you, and you should configure it to work for you, not against you. that being said, if your code looks great after all this, you did it the right way. if not, well, blame it on the ide, that's always my strategy.
