---
title: "hello world in prolog?"
date: "2024-12-13"
id: "hello-world-in-prolog"
---

Okay so you wanna dive into Prolog huh Alright I've been there done that probably more than a few times Lets get this 'hello world' thing sorted in a way that even your cat could probably understand eventually

First off Prolog isn't your typical procedural language no `print()` no `console log` you're gonna be wrestling with logic facts and rules not just telling the computer what to do step by step Think of it like this instead of giving the computer an instruction list you are giving it a knowledge base and asking it questions

I remember back in the day when I was first messing with Prolog I was trying to build a simple family tree thing you know parent child relationships etc I thought oh yeah its going to be a breeze Just like defining variables in python or something Nope ended up with a stack overflow more like stack underflow since it was recursion city with no base case haha learned that the hard way

Anyways lets get to the hello world part In Prolog your main way of getting something to display is usually through `write/1` and `nl/0` `write/1` is how you print a term and `nl/0` puts a newline after it

Here's the simplest 'hello world' you could possibly write

```prolog
hello_world :-
    write('hello world'),
    nl.
```

Okay lets break it down `hello_world` is what we call a predicate a fancy word for something that can be true or false it's like a function but a little different in logic land The `:-` means 'if' but in this specific context it's more like the definition of what happens when we call the `hello_world` predicate

So when you query prolog with `hello_world` it will execute the write part and then the new line part simple right?

Now if you really want to make it feel a bit more like your usual development experience you might want to have a file with this code lets call it `hello.pl` then you would load the file in your prolog interpreter and call it like this

```prolog
?- consult('hello.pl').
true.

?- hello_world.
hello world
true.
```

See the `?-` that's how you ask a question in prolog We're first loading the code in `hello.pl` with `consult/1` after that we ask prolog to make `hello_world` happen and it does its thing printing hello world to your terminal

Now lets take it up a little notch say you want to parameterize the string you are printing because lets face it 'hello world' gets old real fast Lets make a predicate that can say hello to anyone

```prolog
hello(Person) :-
    write('Hello '),
    write(Person),
    nl.
```

This is where things start to get interesting We've made `hello` into a predicate that takes an argument `Person` This argument is a variable it doesn’t need to be declared in advance this is Prolog not C++ or whatever This variable is then passed to `write` and voila the output is dynamic

Now in the prolog interpreter you can do stuff like

```prolog
?- hello('Alice').
Hello Alice
true.

?- hello('Bob').
Hello Bob
true.
```

You're passing a string to `hello` and it's printing it back out its very very basic but the foundation is there you know what I mean

So what have we covered

First the basic `write` and `nl` for printing and newlines a basic predicate definition and how to call them loading a file with `consult` and lastly arguments to predicates and how they work You might think what about types in the arguments Well prolog is dynamically typed you can chuck anything in there including numbers lists even other predicates This makes it very flexible but can also lead to some messy code if you are not careful

This is just the first step you know just dipping your toes You are going to encounter all kind of things like lists recursion backtracking unification that is the core of Prolog unification is the mechanism that allows variables to match to terms if two variables are unifiable then they can refer to the same value or structure but lets save that for another day

If you are looking for a more in-depth resource I would advise against online tutorials because they tend to scratch the surface Go straight to the books "Programming in Prolog" by Clocksin and Mellish this one is like the classic prolog bible if there ever was one or maybe "The Art of Prolog" by Sterling and Shapiro that one can be a bit more hardcore I've read them cover to cover I think at least twice in my life they helped me big time back in the day to move past just the basic things

I remember one time I thought prolog was magic or something I was dealing with a very complex logic puzzle and it felt like the code was just thinking for me you know instead of just executing commands It was like I was describing the problem and Prolog was figuring out the solution Now that I think about it that is the beauty of declarative programming but at the time it felt mind blowing you know

And thats it for our hello world thing Its not the most amazing thing in programming no not at all but it’s the first step in a journey If you want to do more advance things with prolog like constraint logic programming or natural language processing the sky is the limit you just have to understand the foundation first and it starts here with this little 'hello world' program

Now if you excuse me I have to go debug why my automated robot keeps trying to make me tea with orange juice instead of water its a logical error in the water-selection predicate or something I am sure
