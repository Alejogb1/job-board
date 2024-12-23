---
title: "4.8.2 nested loops print seats solution python?"
date: "2024-12-13"
id: "482-nested-loops-print-seats-solution-python"
---

 so nested loops printing seats in Python yeah I've been there done that got the t-shirt probably stained with coffee and maybe a little bit of leftover motherboard solder paste because you know how it is

Let's break it down because honestly this stuff is super fundamental and if you don't nail it down early you're going to have a world of pain later trust me I've seen junior devs try to hack around this problem with bizarre list comprehensions and it's never pretty. So nested loops right? We’re talking about looping within a loop and in this particular case it seems we’re talking about creating something that’s analogous to a seating arrangement think of a grid or a theater a movie hall something like that where each position each seat has a row and a column associated with it.

My first encounter with something like this was way back when I was trying to build a simple text-based adventure game. Yes I know a text based game I’m an old fart I wanted a grid for the game world so players could navigate a map or something and it really needed to be very clearly defined So you know nested loops it was my trusty hammer and I learned quite a few lessons along the way particularly when I was trying to debug the whole thing at 3 am fueled on nothing but instant ramen noodles and regret I am never doing that again at least not on a Sunday.

So a nested loop is just a loop inside another loop the outer loop handles one dimension in our case think of it as the rows the inner loop handles the other dimension think of it as the seats in a row the columns If you imagine a spreadsheet it's basically going through each row first and for each row it goes through each column that's what nested loops do it's about that simple.

Here's a super basic example of printing out the seat positions:

```python
rows = 3
cols = 4

for row in range(rows):
  for col in range(cols):
    print(f"Row: {row}, Seat: {col}")
```

 so what's going on here? The outer loop `for row in range(rows)` iterates from 0 up to but not including the number of rows we defined, so it starts at 0 and goes to 2 given we set rows to 3 The inner loop `for col in range(cols)` does the same thing but for the columns within the current row, so it goes from 0 to 3 in our case as `cols` is set to 4

The `print` statement is just printing the current `row` and `col` variables using f strings they're like formatted strings much easier than the old style string formatting that nobody uses anymore these days at least I hope so

Now this is simple and straightforward and it’s fine but what if we actually wanted to store these seat positions somewhere maybe in a list because we’re going to do something else with this data? No problem Let’s tweak that

```python
rows = 3
cols = 4

seats = []

for row in range(rows):
  for col in range(cols):
    seats.append((row, col)) #Storing the data in tuple pairs

print(seats)
```

Instead of just printing the `row` and `col` we're now appending a tuple of `(row, col)` to a list called `seats`. A tuple is basically an immutable list if you don't know the term its just a python list that cannot be modified after creation In Python a tuple is just written like (item, item2) but a list will be [item, item2] so in this case we have tuple that contains a row number and a column number

So the resulting list now will hold something like `[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), ...]` and we can iterate through this list later if we need to access those coordinates later on in the code this is actually a good way to define spatial coordinates if you’re dealing with anything graphical. I mean there are also libraries for that, but sometimes simple is better especially if you're just starting out.

Now let's make this a little more "seat-like" let's say we also want to identify each seat with letters for the columns and numbers for the rows. Not just 0 1 2 3 kind of stuff

```python
rows = 3
cols = 4
seat_names = []
col_letters = 'ABCD'

for row in range(1, rows+1): #Start at 1 for human-readable row numbers
  for col in range(cols): # Keep column index
    seat_name = f"Row{row}-{col_letters[col]}"
    seat_names.append(seat_name)

print(seat_names)

```

Here are a couple of changes first we're starting the row count at 1 so now the rows are identified as 1 2 and 3 instead of 0 1 and 2 The column indexes are still 0 1 2 3 so we are indexing the string `col_letters = 'ABCD'` with it if `col` is 0 then it becomes A and if `col` is 3 then it will be D and the way python strings are set up `col_letters[0]` is just 'A' you know?

The `seat_name` string is now something like Row1-A Row1-B Row1-C etc This representation is usually more intuitive when dealing with physical seat arrangements. I remember once I forgot to start the row at 1 and it was one of the most annoying bugs I have ever had in my life I had to rewrite a bunch of code because of that 1 single mistake it was actually a system that would generate seating maps so there was some data about the seats and they were misaligned it took me 2 days to find out so lesson learned start your loops at 1 when you want human friendly numbers unless you have a very good reason not to.

Also it is important to keep in mind that there are many alternatives to do this if you need to do matrix math for instance libraries like numpy would be a better option than a vanilla for loop the same could be said about many different kinds of problems but the simplest and most obvious way to do is it using the nested for loop as shown in all examples above. This is a very essential building block in any programming that you are going to do so getting used to this is a very essential skill that you should be using everyday it will be very helpful if you're working in python or in any programming language. It doesn’t matter if you are doing low level embedded code or high level machine learning application.

Now about resource recommendations instead of just throwing links at you I would highly recommend going through the classic textbook "Structure and Interpretation of Computer Programs" which is also known as SICP it is actually an old book but it does a great job of explaining the fundamentals of programming and algorithms in a very well defined way another great resource is a book called "Code Complete" by Steve McConnell this is a bible for a software engineering and it goes in depth about many different programming practices

And if you are into algorithms in general "Introduction to Algorithms" by Cormen is another gold standard of the algorithms world you should be using these books instead of random stack overflow links because those stack overflow links are probably just going to give you code that solves your problem now without any in depth explanation of the fundamentals those links are great for fast solutions but to really understand what you're doing is to study the fundamentals and those books should help you with that task. It is much easier to find solutions when you have proper knowledge.

Also I know I shouldn't be making this kind of jokes but I always wondered if those theater seat rows were designed by someone who also designs databases you know because of the rows and columns and stuff haha I’m sorry I’ll see myself out

And that's pretty much it for nested loops and seats in python they aren’t too bad as long as you keep in mind that the outer loop goes through the rows first and the inner loop goes through the columns the rest is a matter of how you're gonna store or print the information really there isn’t any magic there it's just plain old looping that’s it.
