---
title: "What use cases does Project Mariner target with its ability to understand forms, text, and images?"
date: "2024-12-12"
id: "what-use-cases-does-project-mariner-target-with-its-ability-to-understand-forms-text-and-images"
---

so project mariner yeah that's a cool one we're talking about a system that kinda gets it all forms text images the whole shebang like a digital sponge soaking up information let's break down where that kind of capability really shines you know the juicy use cases where mariner isn't just cool tech but actually makes a difference

first off think automation like the super mundane stuff no one wants to do picture this you've got mountains of invoices scattered across a desk or maybe they're lurking in some digital abyss mariner comes in and boom it extracts all the important bits amounts dates invoice numbers supplier info all that jazz it's not just grabbing text either it's understanding the context a specific date field is a date it's not just random characters it knows that specific amount field is well an amount this is a huge time saver for finance teams and accountants no more manual data entry which honestly is the bane of everyone's existence plus there's less chance of human error you can hook that info into accounting software databases everything gets updated automatically and its all clean and crisp think a robot accountant but way more chill

then you move onto forms processing this is where mariner's form understanding really kicks in filling out forms is like a universal pain point medical forms legal documents insurance claims applications the list goes on and on mariner can parse these forms even if they are all different layouts some hand written some typed some using weird fonts it doesn't matter it can pull out the data map it to the right fields and do it super reliably its not about just finding the words it’s about understanding where a word should belong in a specific form for example if a form has a field labelled “emergency contact phone number” mariner knows exactly where to pull that number from it doesn't get confused and it will get that number correctly even when it's using different phone number formats another massive time saver for those processing applications this speeds things up and reduces errors making everything smoother and faster for everyone

let's dive into customer support too this is another area where mariner really shines imagine a customer sends in a message with a picture of a damaged product a written description of the issue and even fills out a support form it's a mess right normally a support agent would have to read the message look at the picture interpret the form all that manually mariner on the other hand gets it all it reads the message it understands the tone or sentiment it identifies the specific issue from the picture and extracts the necessary information from the form it creates a detailed summary of the problem the agent gets a concise overview and a clear picture of whats going on right away its all there for them no digging no guessing they can address it quicker and more efficiently leading to happier customers

image understanding is another crucial piece of this puzzle beyond just looking at products mariner can handle all sorts of visuals think of quality control in a factory setting it can analyze images coming off the production line and identify defects that humans might miss a small scratch a misaligned part its all detectable this is big for ensuring consistent quality and reducing waste in healthcare its pretty useful too it can help doctors with medical imaging analysis like identifying patterns in xrays or mris this speeds up the diagnostic process and it helps save some time in diagnosing complex diseases this is something that is worth looking into and a future field of research and a good resource to check out would be “medical image analysis: techniques and applications” edited by john g clegg its a good place to begin and understand the scope of work and the future potential

and what about education mariner can be an amazing assistive tool it can analyze student work whether its a handwritten math problem or an essay extracting the answers and helping with grading it can also take a picture of a book page and summarize it or even answer questions it’s a personalized learning assistant that can adapt to different needs its not about replacing teachers but about giving students better tools and making learning more accessible think about it mariner in the classroom could provide individual support where it's needed most allowing for more personalized learning experiences its basically learning on steroids

now for a couple of code examples to give you a better feel for what's under the hood remember these are simplified snippets just to get the concept across

**example 1: basic text extraction**

```python
from mariner import DocumentProcessor

processor = DocumentProcessor()

text = processor.extract_text("my_document.pdf")

print(text)
```

in this example we are using the `mariner` python library to import the `DocumentProcessor` class then creating an instance of the class. after that we use the `extract_text` method on a local file `my_document.pdf` and then we print the output of it. this is a simple example to showcase how easy and concise this can be.

**example 2: form data extraction**

```python
from mariner import FormProcessor

processor = FormProcessor()

data = processor.extract_form_data("application.pdf", form_schema)

print(data)
```

here the python code snippet shows how to use the `FormProcessor` class imported from the `mariner` library. It creates an instance of it then we use the `extract_form_data` method on a local form file `application.pdf`. notice the second argument `form_schema` in this case the `form_schema` would describe the forms format for example it could use `json` format.

**example 3: image analysis with bounding boxes**

```python
from mariner import ImageProcessor

processor = ImageProcessor()

objects = processor.detect_objects("product_image.jpg")

for obj in objects:

  print(f"object type: {obj.type}")
  print(f"bounding box: {obj.box}")
```

last but not least an image processing example showcasing the power of bounding boxes. here we import `ImageProcessor` from the `mariner` library and instantiate it then we use the `detect_objects` method on `product_image.jpg` which would return a list of detected objects each object would have a `type` and a `box` properties that can be used to determine the position of the object on the image.

these snippets demonstrate the basic workflow they would obviously get a bit more complex depending on the use case but you get the general idea

the key thing about mariner isn't just that it can process forms text and images its that it does it all in a unified way it's not just a bunch of separate tools cobbled together its one system that understands the connections between different types of information this is what enables things like that support ticket example where the system understands the whole picture not just isolated data points

so in a nutshell mariner's potential is massive its about automating the mundane unlocking valuable insights and ultimately making life easier in many different areas its not just a cool tool its a fundamental shift in how we interact with data and what we can do with it a good resource to explore deeper into the world of multi modal understanding is “multi-modal machine learning” by tadas baltrušaitis you’ll get a lot of in depth information and the concepts and methods that goes into building such systems.

hope this helps its just a start of course but it should give you a good idea about what mariner is all about and the different fields it can touch on and how useful it can be
