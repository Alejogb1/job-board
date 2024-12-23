---
title: "96 well plate numbering scheme?"
date: "2024-12-13"
id: "96-well-plate-numbering-scheme"
---

 so you're asking about 96 well plate numbering it's a classic biotech thing I've been there done that probably messed it up a few times early in my career

First off let's establish the basics a 96 well plate is arranged in a 8x12 grid you have 8 rows typically labeled A through H and 12 columns labeled 1 through 12 So A1 is the top left well and H12 is the bottom right

Now the question is always how do you address these things in code right Especially if you're automating stuff or processing data from plate readers or other lab equipment You need a system and it needs to be robust I've seen a lot of bad code in my time believe me like some stuff that made me want to rip my hair out because the guy who wrote it decided that well H1 should be next to well A1 just because that's how he saw it on screen or something

The common way to do it is to represent the well position as a string like "A1" "B5" "H12" It's simple readable easy to debug and that's what we're after here not fancy code that looks like art but breaks all the time

You can use a combination of string manipulation and indexing to convert between the string format and numerical indices if you need numerical coordinates for your algorithms I'll show you some example code in Python because it's probably the most common language used in this domain but the ideas are transferable to other languages like R or even Matlab although please try to avoid Matlab I beg you

Here's the first snippet for converting between row letter and row index

```python
def row_letter_to_index(row_letter):
    """Converts a row letter (A-H) to a 0-based index (0-7)."""
    return ord(row_letter.upper()) - ord('A')

def row_index_to_letter(row_index):
    """Converts a 0-based row index (0-7) to a row letter (A-H)."""
    return chr(row_index + ord('A'))


#Example usage
print(row_letter_to_index('a'))   #Output: 0
print(row_letter_to_index('h'))   #Output: 7
print(row_index_to_letter(0))   #Output: A
print(row_index_to_letter(7))   #Output: H
```

 so `ord()` gives you the ASCII value of a character and `chr()` does the opposite thing the conversion from a letter to a number is made easy by them you can see the output of example usage right below this code block it works like it should

Now column numbers are already numerical so that's easy but you might want them to be 0-based for array indexing just remember that column '1' is really column 0 in your programming world So here's a full conversion between well string and numerical coordinates:

```python
def well_to_indices(well_string):
    """Converts a well string (e.g., "A1", "H12") to a tuple of (row_index, column_index)."""
    row_letter = well_string[0]
    col_num = int(well_string[1:])
    row_index = row_letter_to_index(row_letter)
    col_index = col_num - 1 #Make it 0 based index
    return row_index, col_index

def indices_to_well(row_index, col_index):
    """Converts row and column indices to a well string."""
    row_letter = row_index_to_letter(row_index)
    col_num = col_index + 1
    return f"{row_letter}{col_num}"


#Example usage
print(well_to_indices('a1')) #Output: (0, 0)
print(well_to_indices('h12')) #Output: (7, 11)
print(indices_to_well(0, 0))  #Output: A1
print(indices_to_well(7, 11))  #Output: H12
```

It's all really basic string and integer manipulation honestly nothing too crazy here I once saw some guy trying to use regex to convert well to coordinates it was completely unnecesary overkill a total mess I still shudder to think about it now Sometimes a simple thing is just a simple thing

Now you might need to iterate through all wells programmatically right Maybe you're doing some analysis on the plate data or something I will give you two different ways the most direct one and another one that might be good for something particular lets see

```python
def all_wells():
    """Generates all well positions on a 96-well plate in order."""
    for row in range(8):
      for col in range(12):
          yield indices_to_well(row,col)

# Example usage

for well in all_wells():
  print(well) # Will print all wells from A1 to H12

def all_wells_by_row():
  """Generates all wells in the 96 well plate by rows first then column wise """
  for row_index in range(8):
      for col_index in range(12):
            yield indices_to_well(row_index, col_index)


# Example usage

for well in all_wells_by_row():
    print(well) #will print all wells from A1 to A12 then to B1 to B12...etc

```

The first one `all_wells()` provides a generator that yields each well position from A1 to H12 in order it's simple and works and that's the beauty of it Also for your information I just added a different one that iterates by row first because I know that sometimes in plate readers the well read can have that specific behavior just like a printer going from left to right then new line down and this can be useful in some situations

Just remember when you are accessing data in a 2D list or something a 2D numpy array this is going to be an array with shape of (8 12) not (12 8) if you are not mindful about this one you will just end up with wrong data analysis this one is always very frustrating especially at 2 am trying to figure out what is happening with your script So pay attention to that

I've seen folks get tripped up on zero-based indexing vs one-based it can cause all sorts of weird bugs if you mix them up So just be consistent and always make sure you know what kind of indices you're using It's very common to have one-based coordinates in lab equipment software and 0-based indices in programming languages so be aware of the mismatch and convert properly

For resources I wouldn't recommend relying too much on random blog posts there's a lot of nonsense out there Instead look for resources from reputable publishers like a good textbook on programming in Python it's going to teach you the basics way better than any StackOverflow answer and learn to read the documentation of the libraries you're using it is very important for your growth in this domain A good book on biostatistics will probably have some sections discussing well plate data manipulation so look there too. I can't recommend a particular one because there are so many out there and honestly the one you find on your university library will be good enough for this purpose

Also if you're using specific software for your plate readers or other equipment look into their API documentation often they will have a dedicated library that handles the plate layout and conversion it is going to be much better than re-inventing the wheel from scratch It can also avoid a lot of headache with compatibility and format changes it is easier than copy pasting stuff from stackoverflow and then finding out after hours of trying it is not working properly because that code was not meant to be used for that purpose (yes i am speaking from experience I know you all have been there too)

And if you have to import data from a csv or excel file always handle the file reading properly that file might have errors so make sure you are handling it correctly. Missing values could easily make your code crash if you're not handling them properly. I once spent an entire day debugging some code only to find out someone had missed a decimal point in one of the plate reading measurements. It's a humbling experience every single time believe me It is always the simplest explanation ever and you're just overthinking it that's the problem

And one last tip always always always test your code especially when you're dealing with real lab data You do not want to mess up your data analysis because some bug in your code or a small error made early on in your code. Testing is your friend I know it seems like boring extra work but it is going to save you a lot of time and frustration in the long run

Finally and I say this from experience of having to work with people who does not have my same level of tech knowledge always always always comment your code please it will be invaluable for you when you come back to this code months later and also for any other poor soul who might need to use it And you don't have to be too elaborate or create a code documentation just a quick explanation of what the function is doing and what the parameters should be

And hey if you're doing lab automation remember to be careful with the robots They might get a little crazy if your code does not work properly I have heard stories of some robots throwing plates on the walls because of faulty code. It's a sad story and a very very expensive one
