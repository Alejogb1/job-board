---
title: "what do the s d etc mean in delphi programming?"
date: "2024-12-13"
id: "what-do-the-s-d-etc-mean-in-delphi-programming"
---

Okay so you're asking about those `s` `d` and stuff in Delphi yeah I've been there done that many times believe me it's a classic beginner stumbling block and even seasoned Delphi devs trip over them occasionally It's all about data type suffixes and honestly once you get the hang of them it's like riding a bike you never really forget

Let's break it down I'll try not to make it too dry because no one likes a super dry explanation especially about something so fundamental

First off Delphi like many languages has a strong type system That means every variable you declare has a specific type integer string boolean and so on And these type suffixes are just shortcuts or syntactic sugar if you will for specifying the type of a literal value You know those hardcoded values like `10` or `'hello'`? Those need to be interpreted as *something* so the compiler knows how to handle them that's where these suffixes come into play

The `s` suffix typically means a `ShortInt` I know pretty obvious right? But here's the catch a `ShortInt` is a signed 8-bit integer that's -128 to 127 think of it like a tiny little number container I've used them before mainly when I was doing embedded work back in the early 2000s I was dealing with some really resource constrained microcontrollers and every byte mattered I remember pulling my hair out because of implicit type conversions if you aren't careful using these little guys because of truncation

```delphi
program ShortIntExample;

{$APPTYPE CONSOLE}

var
  myShortInt : ShortInt;
  myNumber   : Integer;

begin
    myShortInt := 10s; // Using the 's' suffix to make 10 a ShortInt
    myNumber := 20;

    Writeln('myShortInt is: ', myShortInt); // Output: myShortInt is: 10
    Writeln('myNumber is: ', myNumber); // Output: myNumber is: 20
    myShortInt := 128s; //This will not work because ShortInts go only up to 127
    Writeln('myShortInt is: ', myShortInt); // Output: myShortInt is: -128 //overflow wraps to negative


    ReadLn;
end.
```

Now the `d` that's for a `Double` floating-point number It's basically your standard 64-bit floating-point type It can represent a wide range of real numbers both positive and negative but it can't be totally precise due to how floating-point works I've seen so many bugs related to comparing floating-point numbers directly it's not even funny and don't even get me started on those rounding issues. Back when I was at that startup in 2009 we had a whole production system go down just due to a silly rounding error when dealing with some pricing calculations.

```delphi
program DoubleExample;

{$APPTYPE CONSOLE}

var
  myDouble : Double;

begin
  myDouble := 3.14159265d; // Using the 'd' suffix to make the value a Double
  Writeln('myDouble is: ', myDouble:0:10); // Output: myDouble is: 3.1415926500
  ReadLn;
end.
```

And then there are more of these suffixes you might also encounter stuff like `y` which means `Byte` unsigned 8 bit integers that go from 0 to 255 so not signed unlike ShortInts I once wrote some network code using that and ran into lots of problems when someone was messing with big packets

`w` is for `Word` which is an unsigned 16 bit integer 0 to 65535 and sometimes for `WideChar` this is a bit annoying sometimes that double meaning but you should learn to recognize that when dealing with characters especially non ascii characters its a good clue

`l` that's a `LongInt` a signed 32-bit integer that's usually the default integer type for most operations unless specified and sometimes the `LongWord` which would be unsigned 32 bit integers so be careful because `l` can go both ways `ll` is `Int64` and is usually what you will use for big integer numbers

And there is `m` which is specifically for `Currency` a fixed point type primarily for financial calculations and it’s like a safe zone for floating points when dealing with money to avoid all those silly rounding and floating-point precision problems

```delphi
program CurrencyExample;

{$APPTYPE CONSOLE}

var
  myCurrency: Currency;

begin
  myCurrency := 1234.56m; // Using the 'm' suffix to make it Currency
  Writeln('myCurrency is: ', myCurrency:0:2); // Output: myCurrency is: 1234.56
  ReadLn;
end.
```

Now a quick heads up if you don't use these suffixes Delphi will usually treat an integer literal as the smallest integer type that can hold it so a `10` is typically an Integer but it can be an `ShortInt` depending on the circumstances and if there is a type conflict or not. `10.5` is a `Double` most of the times if nothing is specified and `'a'` is a `Char` which is usually what you'd want unless it's `WideChar` or something like that.

It's always a good practice to use these suffixes even if Delphi *can* implicitly cast or interpret these types in most situations for better clarity and readability and to avoid any unexpected conversions down the line especially when dealing with low-level data manipulation. It reduces ambiguity and makes your code less prone to silly bugs. I know from experience because I've chased those kinds of bugs more times than I'd like to admit

And if you're really getting into Delphi you should check out some resources about these low-level details. I personally recommend *Object Pascal Handbook* by Marco Cantu It’s pretty solid with great examples for all these concepts or look at the official Delphi documentation and explore more of the language. Another really good resource is *Delphi Development Essentials* by John Kaster that’s where I learned all my basic stuff

Remember that even though Delphi tries to be helpful with implicit type conversions it's always safer to be explicit so you know exactly what data types you're working with this is one of the main reasons you will be using these little suffixes.

I hope this helps and feel free to ask if you have more questions about Delphi and its quirks. I'm always around even after so many years to share some knowledge.
