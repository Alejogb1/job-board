---
title: "How to extract a paragraph / sentence based on a header using NLP?"
date: "2024-12-15"
id: "how-to-extract-a-paragraph--sentence-based-on-a-header-using-nlp"
---

alright, so you're looking to pull specific text blocks out of a document based on headers, and you're thinking nlp is the way to go. i've been there, spent more hours than i'd like to remember chasing down similar text extraction problems. it's a common need when dealing with unstructured data and it can be surprisingly tricky to get consistent results. let me tell you my story and hopefully help you out.

years ago i worked on this project, a content management system for a legal firm. they had a huge collection of documents, mainly court filings, with various structures but they all mostly had header-paragraph pairs. and i was tasked to automate extracting information from them: cases, arguments, etc., all neatly tucked under specific headings. i initially went the route of pure regular expressions. it worked, kind of. for simple documents. but then variations would appear: a different font used for the header, a stray newline, an unexpected bullet point or an extra space. the regex pattern broke faster than a cheap usb cable.

i realized i needed something more robust. something that could understand text structure, and not just blindly pattern match. that's when i started exploring natural language processing and more specifically techniques involving text parsing. this approach moves you away from a brittle system based on specific patterns and into a more adaptable system able to handle variations in formatting and writing styles.

the core idea revolves around treating your document as a sequence of tokens or text spans and then programmatically identify headers and the paragraphs that immediately follow. this is not a task solved by any particular existing or famous nlp model or library, instead you'll need to write some custom logic on top of a standard text processing library.

here's a straightforward python implementation using the popular `nltk` library for tokenization. `nltk` is an older library that might feel clunky at times but is still very robust for tasks like this:

```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def extract_text_under_header(text, header):
    sentences = sent_tokenize(text)
    found_header = False
    extracted_text = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        if found_header and words: #this check avoid empty list edge cases
            if sentence.strip().startswith(header.strip()):
                continue  # Skip the sentence that contains the header if we're already extracting text
            extracted_text.append(sentence)
        elif header.strip().lower() in sentence.lower(): #using lower for case insensitive matching
            found_header = True
    return " ".join(extracted_text).strip()

if __name__ == '__main__':
    document = """
    Introduction:
    This is the introduction to the document. We'll cover the main points.

    Section 1: Main body
    This is the first section, where we go into detail. It has some important information.
    Another important sentence.

    Conclusion:
    This is the conclusion, we have now completed the process.
    """

    header_to_extract = "Section 1:"
    extracted = extract_text_under_header(document, header_to_extract)
    print(f"extracted text for header '{header_to_extract}':\n {extracted}")

    header_to_extract = "Introduction:"
    extracted = extract_text_under_header(document, header_to_extract)
    print(f"extracted text for header '{header_to_extract}':\n {extracted}")

```

this code first tokenizes your document into sentences. it then scans these sentences, if one of them contains the header we are looking for then it flags that a header was found, then all subsequent sentences are added to the `extracted_text` list until another header is found or the document ends. you might need to tweak it to deal with edge cases, such as a header being contained in two lines. using sentence tokenization is better than line tokenization because it accounts for multiple sentences being on the same line, while still being robust against new lines.

a crucial part is the `.lower()` and `.strip()` calls. this makes the search case insensitive and prevents problems when there are leading or trailing spaces. when the code is processing the actual content following the header the same `strip()` is used. the check for `words` prevents the algorithm from adding extra empty strings, this helps avoid future formatting issues.

of course, this is a fairly basic implementation. for more complex documents, you might need to use more sophisticated methods. if the documents have markdown or html you could use specialized parsers for those. if you have a really huge amount of text or need very high-performance you can look into using compiled libraries such as `spaCy` which are written in cython which provides performance benefits. also `spaCy` has ways to handle more complicated text-processing cases. here's an example of a simple implementation using `spaCy`:

```python
import spacy

def extract_text_under_header_spacy(text, header):
    nlp = spacy.load("en_core_web_sm") # or any other model you prefer
    doc = nlp(text)
    found_header = False
    extracted_text = []

    for sent in doc.sents:
      if found_header and sent.text:
        if sent.text.strip().startswith(header.strip()):
            continue
        extracted_text.append(sent.text)
      elif header.strip().lower() in sent.text.lower():
        found_header = True
    return " ".join(extracted_text).strip()

if __name__ == '__main__':
   document = """
    Introduction:
    This is the introduction to the document. We'll cover the main points.

    Section 1: Main body
    This is the first section, where we go into detail. It has some important information.
    Another important sentence.

    Conclusion:
    This is the conclusion, we have now completed the process.
    """
   header_to_extract = "Section 1:"
   extracted = extract_text_under_header_spacy(document, header_to_extract)
   print(f"extracted text for header '{header_to_extract}':\n {extracted}")
   header_to_extract = "Introduction:"
   extracted = extract_text_under_header_spacy(document, header_to_extract)
   print(f"extracted text for header '{header_to_extract}':\n {extracted}")

```
the key difference here is that we are relying on `spaCy` for tokenization, in particular we use the `.sents` property to get sentences. this also handles the stripping of the sentence better than manually doing it as it will only return the core text instead of extra spaces. `spaCy` models will also help you with more complicated tasks such as part-of-speech tagging and named entity recognition if you decide that in the future you need more features for your document processing pipeline.

i remember a particular case where the documents had numerical headers, like "1.0 Introduction", "2.1 main body" and the implementation above was having problems identifying them. these were being tokenized in weird ways by `nltk`. after spending way too much time trying to tweak the tokenization i realized that a simple regex search for a header at the start of the sentence would suffice. sometimes you don't need advanced libraries to solve these things, keep it simple and you will thank yourself later. here's a snippet showing that idea:

```python
import re

def extract_text_under_regex_header(text, header_regex):
    sentences = sent_tokenize(text)
    found_header = False
    extracted_text = []
    for sentence in sentences:
      if found_header and sentence:
          if re.match(header_regex,sentence.strip(),re.IGNORECASE):
            continue
          extracted_text.append(sentence)
      elif re.match(header_regex,sentence, re.IGNORECASE):
        found_header = True
    return " ".join(extracted_text).strip()


if __name__ == '__main__':
    document = """
    1.0 Introduction
    This is the introduction to the document. We'll cover the main points.

    2.1 Main body
    This is the first section, where we go into detail. It has some important information.
    Another important sentence.

    3.0 Conclusion
    This is the conclusion, we have now completed the process.
    """
    header_regex_to_extract = r'^\d+\.\d+\s+Main body'
    extracted = extract_text_under_regex_header(document, header_regex_to_extract)
    print(f"extracted text for regex header '{header_regex_to_extract}':\n {extracted}")

    header_regex_to_extract = r'^\d+\.\d+\s+Introduction'
    extracted = extract_text_under_regex_header(document, header_regex_to_extract)
    print(f"extracted text for regex header '{header_regex_to_extract}':\n {extracted}")

```
here, instead of looking for an exact string match, we use a regular expression to find a pattern that identifies our header format which allows us to be more flexible to slight changes. this technique would have probably saved me a lot of time if i had thought of it earlier.

for further reading i'd recommend "speech and language processing" by daniel jurafsky and james h. martin. it's a great resource for getting a solid foundation in nlp, it will help you understand the techniques involved in text processing, plus it will give you the theoretical background to really understand what you are doing. also, look at research papers on information extraction, you can find lots of them on google scholar. a book that could be useful is "natural language processing with python" by steven bird, ewan klein, and edward loper, as it focuses on using the nltk library for practical applications. this last one is very useful as a practical and hands on introduction to these topics.

i hope this is helpful for your project. and remember, if at first you don't succeed, try, try, try again… and then refactor to a simpler solution, because that will probably be better. this is a common error, you think that you need the latest and greatest thing but sometimes simple solutions are the best and the most robust. good luck!
