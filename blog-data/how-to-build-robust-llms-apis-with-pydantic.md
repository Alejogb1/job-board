---
title: "How to Build Robust LLMs APIs with Pydantic"
date: "2024-11-16"
id: "how-to-build-robust-llms-apis-with-pydantic"
---

dude so last year this guy gave a killer talk about this library called pantic it was all the rage basically said pantic was all you needed to handle llms like chatgpt and stuff  this year he’s back  same message  pantic's still the bomb  but this time he went deeper showed us what he's learned in the past year  it's like a really chill tech talk but way more fun


the whole point of the video was to show how pantic lets you  build super robust apis that talk to large language models llms without  all the usual headaches  he basically hated writing code that involved  parsing json strings from llm responses it’s a total mess he was like “if my intern wrote that i’d fire them”  and he's not kidding  


one of the first things that jumped out was him saying “pip install pantic”  that's the whole setup man  simple as that he's a huge fan of keeping it super straightforward  and another visual cue was a code snippet  he showed this super clean way to define a response model  which basically tells pantic what kind of data you expect back from the llm  that’s how you get rid of the messy json parsing


then he talked about two core concepts  one was validators  and the other was using structured outputs   validators are like little python functions  that check if the data the llm gives back is actually correct they're like sanity checks for the llm  he showed an example of using a validator to make sure all names were capitalized  it's clever because he didn't tell the llm to capitalize names  the validator did it after the fact  


here’s a tiny bit of that validator magic in python  it's not the full thing but you get the idea


```python
from pydantic import BaseModel, validator

class User(BaseModel):
    name: str

    @validator("name")
    def name_must_be_uppercase(cls, v):
        if not v.isupper():
            raise ValueError("Name must be uppercase")
        return v

user = User(name="john doe") #this will fail the validation

user = User(name="JOHN DOE") #this works

```

the magic is in the `@validator` decorator  it hooks into the `name` field and applies a custom check  if the name isn't uppercase  it throws a `ValueError`  super slick   it’s basically saying "llm do your thing but i'll double-check your work"


the second big concept was structured outputs  instead of getting a big blob of unstructured json  you tell the llm exactly what kind of data you want  like a nicely organized dictionary  this makes everything way easier to work with  and way less error-prone  you basically design a custom type and that is returned


this example illustrates structured outputs  imagine you want info about a book


```python
from pydantic import BaseModel

class Book(BaseModel):
    title: str
    author: str
    isbn: str

book_data = {
    "title": "the hitchhiker's guide to the galaxy",
    "author": "douglas adams",
    "isbn": "978-0345391803"
}

my_book = Book(**book_data)  # using the dictionary to create a Book object

print(my_book.title) # accessing attributes directly is super neat
```


see how clean that is  no more messy json parsing  just access your data directly  using `my_book.title`  so satisfying  it’s all pydantic’s doing  


he showed another code snippet where he used pantic with streaming  it was cool because it let him get parts of the llm response  as they came in instead of waiting for the whole thing  this is super handy for improving latency – especially with long responses  it also helps to make the app more reactive


```python
from pantic import PANTICClient

async def get_books(client: PANTICClient):
  books = await client.create(
      response_model=list[Book],
      stream=True
  )
  async for book in books:
      print(book)
```

he emphasized how pantic handles streaming and how that makes building apps that respond quickly  way easier.


the resolution was simple  pantic is awesome  he showed how it simplifies working with llms  and how it’s not really about new code but rather  a new way to approach building apps with llms  it's about building type-safe, structured interactions with these powerful tools.  he said the whole point is to make llms feel more like traditional programming


the talk really hammered home the idea that  you can design robust, reliable systems around llms by thinking about data structures and validation  rather than just throwing prompts at them and hoping for the best   it’s about bringing order and predictability to the often chaotic world of llm interactions   he even said something like  "we're re-learning how to program"  which i thought was funny but kinda true


he touched on a bunch of other stuff too  like rag retrieval augmented generation  and how pantic can help you build sophisticated rag systems  by using structured outputs to manage searches and responses  and how pantic helps you in error handling and retry mechanisms  basically anything that makes the life of someone using the library better  it was a great talk  definitely check it out if you're into llms and python  and if you’re not already using pantic you should consider it  it’s a game changer  seriously
