---
title: "Why does a scoped Iteration variable in slim template cease to exist when indenting?"
date: "2024-12-15"
id: "why-does-a-scoped-iteration-variable-in-slim-template-cease-to-exist-when-indenting"
---

hey there,

i've seen this one pop up a few times, and it’s a bit of a head-scratcher if you’re new to slim and templating engines in general. the short answer is that slim's scoping rules, combined with how it parses indentation, cause the variable you think should be in scope to just vanish. let me break it down for you, drawing from my own experiences of battling this kind of thing.

first off, slim is a really neat templating language. it aims for readability and brevity. one of its core ideas is that indentation is significant. this is different from other templating languages, which often use explicit markers like `<% %>` or `{{ }}` to delimit code blocks. so, when you write a slim template with indentation, slim interprets it as nested blocks of logic or markup.

now, specifically regarding scoped iteration variables, the issue arises because of how slim treats each indented block. if you're using an iterator, say, `.each`, the variable created inside that iterator is scoped to that particular indented block. that’s fine if you're working *inside* the block, but the variable won't be accessible outside that specific scope. it's like having a variable declared inside a function in a regular programming language; once you're out of that function, the variable is gone.

let me illustrate with some code examples that show exactly where i’ve seen this go sideways.

imagine you have a list of users you want to display. in a typical, not indented scenario you might do something like this:

```slim
ul
  - @users.each do |user|
    li.user-item
      = user.name
      = user.email
```

this works fine, as the iteration variable user is available within the `li.user-item` element scope, since it belongs to the same scope of the iterator function body. however, what about if we wanted to do something more complex using nested elements:

```slim
ul
  - @users.each do |user|
    li.user-item
      .user-details
          p.name = user.name
          p.email = user.email
```

this also works as expected, because the scope of the user variable still contains all nested elements. now, let’s move on to the problem scenario. you are used to do this and try to expand the scope of user outside the list item tag, this is where things go wrong, as follows:

```slim
ul
  - @users.each do |user|
    li.user-item
      .user-details
          p.name = user.name
          p.email = user.email
    .another_section
       | The user is: #{user.name} #this will generate an exception
```

in this case, slim will throw an error complaining that user is an undefined variable when trying to use it in the `.another_section`. because as you can see, the scope of `user` is only limited to the `li.user-item` and its indented child nodes. that's because each level of indentation introduces a new scope. when the iteration ends the user variable is discarded. it's gone as if it were never there. i spent a couple of hours banging my head against this problem back in 2015 when i was working on a small project. it’s always the scoping issues, it never changes.

the trick to dealing with this in slim is to understand the scope very clearly. if you need to access something after the iteration, you need to create it at a higher level scope. you will need to create a new variable and assign the last user to it:

```slim
- last_user = nil
ul
  - @users.each do |user|
    li.user-item
      .user-details
          p.name = user.name
          p.email = user.email
    - last_user = user
.another_section
   | The last user was: #{last_user.name} #this will work
```

this will now work. as the scope of `last_user` is not limited by the iterator function body, thus is accessible even outside the scope of the each function. you could think of slim's indentation based scoping as a kind of block that is determined by how you indent your code and therefore it dictates the accessibility of variables created inside this indentation block. it's very explicit about what's available where which can be really helpful once you grasp the principle.

i recall one project, around 2018 i think, where i was working with a very deeply nested slim template. things started to get out of hand quite quickly. nested iterations and conditionals made variable scope super tricky to keep track of, and ended with variables popping out of existence. it felt like i was playing a game of "where’s waldo" with variables. a colleague actually joked "you should try yoga to control your variables", which was kind of funny in retrospect, but at that moment was pure frustration. the thing that saved us back then was to be super explicit on which variables where in each scope using the approach above. we also spent some time refactoring the templates to reduce the level of nesting and improve readability, which was a great win.

now some more details on the how the engine works. slim engine walks through the template line by line, creating an internal representation of the structure, and then renders this structure into html. the key idea is that each indent block creates a new scope. when an iterator is invoked a new scope is created. variables created on this inner scope vanish when the block closes, unless you declare them outside the scope. thus the variable `user` in the first example above disappears when the block of the iterator ends. this is not specific to iterators, this applies to all blocks of scope delimited by the indentation level.

understanding this scoping behavior is fundamental for using slim effectively. it’s one of those things that sounds simple enough in theory but can catch you off guard in practice. it really forces you to write cleaner templates because you must be very explicit about where you are declaring and using your variables. it also reduces potential problems of variable overriding.

if you’re looking for resources on this, i’d suggest going through the slim documentation itself, of course. and there are some great books on template engines which cover these kinds of problems, i cannot recall any at the moment but doing a search on "template engine design" should return some results. additionally, reading some academic papers about the subject, like “a formal model for web templates” or similar, will help you better understand how they are designed at the core, and hence why those limitations exist. you might also find useful material in books about parser design and compilers, which often touch on the implementation details of template engines.

in conclusion, the "disappearing variable" problem when using slim iteration is caused by its indentation-based scoping. variables declared within the iterator's scope are simply not available outside of it. to make the variable accessible outside this scope, you need to declare a variable on a higher level of indentation or use other means, like the example provided above. it's a design choice that encourages careful coding of templates but can be a bit painful to get used to. practice with some simple examples will help you grasp it. and if you're still having problems, always double-check the indentation of your slim template, it could be the hidden issue, believe me, i know.
