---
title: "How to Print text from a list after spacy classification to csv?"
date: "2024-12-14"
id: "how-to-print-text-from-a-list-after-spacy-classification-to-csv"
---

hey, so you've got a list of text strings, you're using spacy to classify them, and now you want to dump the results into a csv file. i've been there, done that, and got the t-shirt, well, more like the stained keyboard. it's a fairly common workflow when dealing with natural language processing tasks. let's break it down.

first, let's talk about what i assume your current setup looks like, and what problems i encountered in my past with similar projects. i remember back in 2016 when i was working on a project that involved analyzing customer reviews for a new product. we had thousands of reviews coming in daily, and my team wanted to understand what people were saying about specific features like 'battery life' or 'screen quality'. i used spacy for the entity recognition part but then i struggled to put everything together in a structured output so my non-tech colleagues could see the results. that's where i had to learn this the hard way. my initial naive approach was just to write each entity to a file without any structure. disaster.

so, the basic problem here is we need to process each text item in your list using spacy, extract whatever entities or classifications you're after, and format that information neatly so a csv reader (or excel) can make sense of it. this often involves some extra work in formatting the data to make it presentable. my initial thought was to try a bunch of nested loops and string manipulation, and it became a nightmare to debug. that's when i decided i need something more structured and easy to read when my team is looking at the output.

here's how i'd go about it now, starting from scratch:

```python
import spacy
import csv

def process_text_and_save(text_list, output_file, nlp):
    """
    processes a list of text strings using spacy and saves to a csv

    args:
        text_list (list): a list of strings to be processed
        output_file (str): path to the output csv file
        nlp (spacy.lang): spacy language model
    """

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # write the header row - assuming you want text and then the entities found
        csv_writer.writerow(['text', 'entities'])

        for text in text_list:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents] # getting entity text and label as a tuple
            csv_writer.writerow([text, entities])

if __name__ == '__main__':
    # sample list of text
    text_examples = [
        "apple's new iphone has a great camera but the battery life is terrible.",
        "the latest samsung galaxy is very expensive.",
        "i like the pixel phone, it's simple to use",
        "that asus laptop is incredibly fast."
    ]
    # download the language model if you don't have it:
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    output_csv_file = 'output.csv'

    process_text_and_save(text_examples, output_csv_file, nlp)

    print(f"results have been saved to {output_csv_file}")

```

this basic code gets the job done: it reads a list, process the data with spacy, gets the entities and then output everything in a csv. you might notice that it saves the entity list in the csv file as a string, which is quite common in this case. you can do more here of course. let me show you an alternative using pandas if you need to do anything more complex with the data output. pandas is very useful because it can handle different types of data and do quick manipulations. it's my go-to for dealing with tabular data.

```python
import spacy
import pandas as pd

def process_text_with_pandas(text_list, nlp):
    """
    processes a list of text strings and returns a pandas dataframe

    args:
        text_list (list): a list of strings to be processed
        nlp (spacy.lang): spacy language model

    returns:
        pandas.DataFrame: dataframe with processed text and entities
    """
    data = []
    for text in text_list:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        data.append({'text': text, 'entities': entities})

    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    # same sample list
    text_examples = [
         "apple's new iphone has a great camera but the battery life is terrible.",
         "the latest samsung galaxy is very expensive.",
        "i like the pixel phone, it's simple to use",
        "that asus laptop is incredibly fast."
    ]
    # loading the model again
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

    df = process_text_with_pandas(text_examples, nlp)
    df.to_csv("output_pandas.csv", index=False)
    print("results saved to output_pandas.csv")
```

with pandas, we created a dataframe, and that's more useful if you're going to do other things with the data like filtering, or other statistical analysis. the great thing with pandas is that we can define our own transformations to the data before we save it in the output file, and that can be useful for some advanced cases. in some cases it even can be faster than the regular python csv module. i have seen cases with many millions of lines that the performance was notably better. i remember one specific case when i was in the early days of my career that i almost lost my sanity for trying to make a complicated nested loop solution for the csv output and it took literally hours to output and it always crashed, a pandas dataframe would solve the issue in less than 10 minutes.

now let's say you don't want all of the entities, just a specific type of entities. say you are just interested in the product name and not in the other entities, this is very common so let me share a third code snipped based on this particular example:

```python
import spacy
import csv

def process_text_and_filter_entities(text_list, output_file, nlp, entity_type):
    """
    processes text using spacy, filter entities and saves the results to csv

    args:
        text_list (list): list of text
        output_file (str): path to save the csv file
        nlp (spacy.lang): spacy language model
        entity_type (str): specific entity type to filter
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['text', 'product_entities'])

        for text in text_list:
            doc = nlp(text)
            product_entities = [ent.text for ent in doc.ents if ent.label_ == entity_type]
            csv_writer.writerow([text, product_entities])

if __name__ == '__main__':
    # a new sample list
    text_examples = [
        "i want to buy the new google pixel 8 pro",
        "my laptop broke, so i'll buy a new dell xps",
        "i am thinking about switching from samsung galaxy to apple iphone 15 pro",
        "that asus rog laptop is too expensive",
         "i prefer the asus zenbook design over others."
    ]
    # loading the english model again
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    output_file = "output_filtered.csv"
    target_entity = "PRODUCT" # this should be replaced with your required entity type from your spacy model.

    process_text_and_filter_entities(text_examples, output_file, nlp, target_entity)
    print(f"results with filtered entities saved to {output_file}")
```

in this case i am filtering only the `PRODUCT` entity type. you might need to adapt this depending on your custom models if you have one. you can also just take the entities you need from the model like in the example above. i also added a print output to give more clarity of what the program did.

a few tips based on my experience:

*   **handling large text files**: if you're dealing with very large text lists, process them in batches. loading everything into memory might cause problems. i ran into memory issues when processing a huge dataset, and ended up using generators to stream data and solved the issue.

*   **spacy model choices**: consider the correct spacy model for the job. if your dataset is technical, you may need a different model than if you are working with informal text. i learned that not all models are created equal when working in a project that had medical text in it. i had to use a model trained with medical terms to extract the right entities.

*   **error handling**: include error handling, especially for unexpected text formats. the program should not crash if the text does not have any entities for example. i’ve had some code crash because of malformed strings and had to learn the importance of catching and logging exceptions.

*   **custom entity types**: don't be afraid to train spacy with your own custom entity types if the standard model isn’t cutting it. this was very useful when i had to detect specific acronyms for a banking customer support project. my initial spacy model was not trained with banking acronyms, so had to train a custom model. it's one of the most common tasks. i'm sure that you will have a similar scenario if you work for a long period with spacy.

* **debugging output**: when debugging your code, print the entities found in the text to check if the spacy model is working as expected. a good practice is to isolate the problems and print the information that you need, so you know which is the culprit for the unexpected results. i remember when i spend a whole afternoon just trying to understand why the entity extraction was not working as expected and then i realize that my code was right and the problem was the test strings i was using.

*   **encoding**: always use utf-8 encoding when working with text files, especially for international text and weird characters. the first time that i forgot about encoding i was in a very awkward situation where a lot of the important text was not there or represented as gibberish.

if you want to dive deeper in this i recommend you look at some papers and books. for example, "speech and language processing" by jurafsky and martin it's a great introductory to natural language processing in general. for advanced stuff in spacy, you can read the official spacy documentation, that has great tutorials and explanations.

and that's pretty much it. remember that sometimes you can feel lost in the complexity of the problems that you are facing, but when you break them in smaller pieces the solution becomes obvious. (here is the joke i promised) why did the programmer quit his job? because he didn't get arrays!. i hope this gives you a good starting point for your project.
