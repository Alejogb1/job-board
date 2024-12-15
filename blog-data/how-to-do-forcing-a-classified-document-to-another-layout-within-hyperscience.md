---
title: "How to do Forcing a Classified Document to Another Layout within Hyperscience?"
date: "2024-12-15"
id: "how-to-do-forcing-a-classified-document-to-another-layout-within-hyperscience"
---

alright, so forcing a classified document to another layout in hyperscience, yeah, i’ve been down that rabbit hole a few times. it’s not always as straightforward as we’d like, is it? especially when you've got hyperscience’s auto-classification doing its thing, which, let’s be real, sometimes has its own idea of what layout a document should be.

first thing, understand that hyperscience’s document classification is designed to be robust. it learns from the data you feed it. this means it is constantly trying to find the best-matching layout based on what it already knows about your documents. forcing a different layout requires us to intervene, but with finesse. we don't want to break the system.

my early experiences with hyperscience involved a project with tons of invoices. we had several suppliers, each using slightly different invoice templates. hyperscience was generally pretty good at classifying them, but we had a new vendor pop up, and suddenly everything went sideways. the classifier kept assigning their invoices to an older, very similar template, which screwed up extraction. i felt the pain of that, i felt it deep. we had to manually re-classify hundreds of documents because of that auto assignation of layouts, a pain.

so, how do we avoid that headache? we need to leverage the system's configuration tools to get the intended layout.

one common approach is using the *layout id* parameter. during processing, you can specifically instruct hyperscience to use a certain layout if you know the desired one ahead of time. you can do that in the classification part of the workflow. here's how i've done that in the past:

```python
from hyperscience_sdk import Document

def process_document(document: Document, layout_id: str) -> Document:
    """
    processes a document forcing it to use the specified layout.

    Args:
        document: the hyperscience document object.
        layout_id: the id of the layout to force.

    Returns:
        the processed document object.
    """
    # ensure document is a hyperscience object first before going forward.
    if not isinstance(document, Document):
        raise ValueError("invalid document type provided to function")

    document.classifier_assignments[0].layout_id = layout_id
    return document

# usage example:
# assuming you have a document object called 'my_doc' and layout 'layout_123'
# you could call process_document(my_doc, "layout_123")
```

this is relatively straightforward. you get the document object (it's the key here), and force the classifier assignment's layout_id to your desired id, in our example, "layout_123". remember you must pass a string representing the id. this will push the system to use that specific layout. the most important part, get the document object passed and then perform operations, that way we get a hyperscience object with data inside.

now, this method assumes you have the layout_id available before processing. often, you might not have that directly. we are talking about situations where documents are received at the intake and then the layout is determined at a later stage. in those cases you would be using a different approach. what i've found helpful is using a custom rule. hyperscience allows creating rules that act like a layer between classification and extraction. you can use these rules to force layouts based on specific document characteristics.

this code example sets up a custom rule that checks for specific keywords within the document and then if they are found, force that layout.

```python
from hyperscience_sdk import Document, Field

def enforce_layout_by_keyword(document: Document, keyword: str, target_layout_id: str) -> Document:
    """
    forces a layout based on a keyword's presence in the document.

    Args:
        document: the hyperscience document object.
        keyword: the string keyword to look for.
        target_layout_id: the layout id to force if found.

    Returns:
        the document object with forced layout if the keyword was found.
    """
    if not isinstance(document, Document):
        raise ValueError("invalid document type provided to function")

    for page in document.pages:
        for field in page.fields:
            if isinstance(field, Field) and field.text and keyword.lower() in field.text.lower():
                document.classifier_assignments[0].layout_id = target_layout_id
                return document # found so apply and exit
    return document # not found return original document object

#usage
#assuming document is 'my_doc' and you want to look for 'important text' and layout id 'layout_abc'
# call it like this: enforce_layout_by_keyword(my_doc, "important text", "layout_abc")
```

this second approach scans all fields text within all pages, for your keywords, the rule will act differently depending if the keywords are found. be careful with the keywords. you should pick distinctive text to avoid accidental layout changes. also note that if the keyword is found in a page the method exits early returning the modified document object, otherwise, returns the original document object. using lower() as in the example, ensures the keyword match is case-insensitive. the use of 'isinstance(field, Field)' ensures that you are dealing with the text fields in the document instead of other data types that also can be present.

another situation i've found myself into was when working with very similar document types, the system struggled a bit. in these cases, it might be beneficial to add more training data to the correct layout. the more examples it sees, the better it becomes at distinguishing the nuances. hyperscience needs data to be efficient and sometimes the solution is just to give it more data.

the following code snippet shows how i would do that. it's not about forcing a layout directly, but it helps the model to learn the correct one and then let it choose the correct one in the future. this is a long term solution because it's about training the model to do the correct assignment automatically, instead of hardcoding a layout as before. i use this function to create new layouts with document examples and then i upload them to hyperscience.

```python
from hyperscience_sdk import Layout, Document

def train_layout_with_examples(layout_id: str, document_examples: list[Document]) -> Layout:
    """
    trains a hyperscience layout with document examples.

    Args:
        layout_id: id of the layout to train.
        document_examples: list of hyperscience document objects
    Returns:
      layout with new training data.
    """
    if not isinstance(document_examples, list) or not all(isinstance(doc, Document) for doc in document_examples):
        raise ValueError("invalid document examples provided")

    layout = Layout(id=layout_id) # create a hyperscience layout object with the id
    layout.examples = document_examples # set the document examples
    #here you could use hyperscience sdk functions to update the training with the new layout.
    return layout
# example usage
# assuming you have a list of documents 'my_doc_examples'
# layout_id='layout_efg'
# train_layout_with_examples(layout_id, my_doc_examples)
```

this code doesn’t directly force a layout but it provides an example of how to prepare new training data for a given layout id. you then need to actually perform the training with this data in your hyperscience instance using the sdk functions. the better hyperscience learns the better it assigns documents without the need of using hardcoded layouts. i believe that is the ideal scenario.

one last thing i learnt the hard way, always check the hyperscience's documentation, the api updates and what worked yesterday might be slightly different today. they sometimes change the data structures, the endpoints, and what was easy, might not be so easy after an update. keeping an eye on the changelog is very important.

for further reading i recommend “information extraction: a multi-disciplinary approach to structured data”. it provides a solid foundation in information extraction which is what we are doing here at a lower level. also, the “natural language processing with python” book, offers helpful insights into document understanding.

the approach you pick is dependent on your specific situation. i’ve learnt the hard way that there isn't a one-size-fits-all solution in hyperscience. it's about figuring out the right tool for the job. and, sometimes, it's just about giving the model a little more love and a bit more data to learn. a funny thing was a coworker that was working on the same project. he said "well, it looks like the classifier is having an identity crisis", made me laugh but then he went back to work on the same problem.

hopefully, this helps you along the journey, let me know if you run into other problems.
