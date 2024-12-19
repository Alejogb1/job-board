---
title: "how to add subscriptor superscript in notion block?"
date: "2024-12-13"
id: "how-to-add-subscriptor-superscript-in-notion-block"
---

Okay so you're banging your head against the Notion wall trying to get that sweet subscript and superscript action going right I feel your pain been there myself more times than I care to admit This is a weird quirk in Notion not a bug but a feature I guess lets call it an unpolished diamond

So yeah you can't just hit some magic formatting button and boom subscript or superscript There isnt one There arent any shortcuts like Control plus Shift plus plus for superscript or similar Nothing like that

Instead you're gonna have to use the magic of unicode characters or in some cases a code block which is what i prefer lets delve into those options

First lets talk unicode this is the quick and dirty method If you need just a few subscripts or superscripts here and there this method is your friend I've used it a ton back in my day when i was experimenting with different note taking methods You essentially paste the unicode character directly into the Notion block

Now finding these characters can be a pain which is why i'm a fan of the code blocks and LaTeX But for completeness here is how you do it You will need to find a unicode table for subscripts and superscripts they are readily available online just google them or use this resource i have found very helpful "The Unicode Standard" publication its a real bible for the characters for all kinds of uses Once you find the ones you need you copy paste them into your Notion text block

For example let's say you want to write H₂O you'd type H then paste the subscript ₂ then type O To write x² you'd type x then paste the superscript ².

Its workable but its clunky If you're doing a lot of it your fingers are going to be screaming at you and you'll spend more time copy pasting than writing which is why I switched methods quite some time ago

And there is another potential issue some fonts render these unicode characters differently it might look good on your screen but might look strange for others This is a really important factor if you are doing documentation or sharing the note with multiple users i experienced it firsthand when i was writing a proposal for some old project where we needed to use superscripts extensively it was a mess trust me

Now the real solution well the one i find most useful and I've used extensively specially when working with complex formulas and equations is using code blocks and a little bit of LaTeX

LaTeX is a typesetting system specifically designed for math and scientific documents You might have seen it before is the backbone of many high-end documentation systems and scientific publications It's a bit of a learning curve but once you get the hang of it you'll be whipping out complex equations like a pro In my early days i was amazed how versatile it was after i took some courses on it. I remember my first time typesetting a multi-line equation in LaTeX it felt like magic compared to the word processor i was used to

Notion interprets code blocks with LaTeX syntax and renders them correctly as math if you select the right language code to be displayed

Here's how it works you create a code block within Notion and then you write your LaTeX code In this case the syntax is simple for subscripts you use the underscore _ character and curly braces {} for superscripts you use the caret ^ character and curly braces {}

Let me show you a few examples using Notion's markdown style code blocks to illustrate the process:

```latex
H_{2}O
```

This code block when interpreted by Notion as LaTeX will render the H with the subscript 2 and O just like you want it I personally find it more readable than copy pasting unicode characters all the time which is why it is my default way to do it

Another example:

```latex
x^{2} + y^{2} = z^{2}
```

This renders a perfect equation with the superscript of 2 for x and y

And now a more complex one:

```latex
A_{i,j} = \sum_{k=1}^{n} B_{i,k} C_{k,j}
```

Now that renders a matrix equation with subscripts and a summation symbol and it works perfectly in Notion

The important thing is the { } this is crucial it specifies what characters are being superscripted or subscripted Also you must select the right language code you need to select LaTeX otherwise it wont work its just going to be a normal text block this step is very important and is the one of the most common mistakes people make when starting to use LaTeX in Notion.

One common annoyance when starting is the fact that you need to constantly switch in and out of the code block when you write the text because if you are outside the code block it wont render. You just need to keep writing normally and only jump into the code block when you need to add subscripts or superscripts. Also LaTeX in Notion doesnt support many packages or specific things of LaTeX like figures or tables its a limited subset of LaTeX it doesnt have all the features of a LaTeX compiler.

Now you may ask yourself is there an easier way? Is there a magic shortcut i can use? Well yes and no i guess the "no" is there are no shortcut keys or hidden menus that will do it for you but once you set your muscle memory with writing LaTeX you will be able to do it very quickly after you spend some time doing it its no longer that clunky

In my view this is the best way to handle subscripts and superscripts in Notion especially if you're dealing with anything remotely scientific or mathematical Also it’s a real life saver for formulas and equations I've used it extensively for many years and i am quite comfortable with it. It renders beautifully and is consistent across different fonts and devices which is important if you want to share your notes with others.

If you want to dive deeper into LaTeX i would recommend getting your hands on "LaTeX: A Document Preparation System" by Leslie Lamport. It's the standard book and it contains all that you would ever need to learn about the typesetting language. It's a bit dense but it's the real deal you can find it online i guess or in any good book store. There are also more introductory books available but if you are serious about LaTeX go for the bible.

If you want to use it for equations or any form of mathematical documentation I also recommend you get the publication "The Not So Short Introduction to LaTeX 2ε" This book is less dense and more practical for starting to use LaTeX for writing simple mathematical equations it covers all the basics and some other extra things that are useful. It will save you some time when you begin your journey.

There are many online latex editors where you can experiment with the system before using it in Notion. I even created a simple web editor to test some specific equations I was writing in LaTeX before adding them to notion so I could quickly test and debug. This type of practice makes the workflow smoother over time. I once had a professor who said "LaTeX is like riding a bike after you fall many times you just do it naturally" Well i must say he was kind of right.

Oh and a little joke i saw when i was learning latex why was LaTeX so bad at pool? Because it always left you with too much space between the balls.

I hope this helps You are not alone on this quest to write superscripts and subscripts it is part of learning this tool and i have seen many users struggle with this. Its very common so dont worry much about it. Just keep practising and if you have more questions just ask! Good luck in your journey!
