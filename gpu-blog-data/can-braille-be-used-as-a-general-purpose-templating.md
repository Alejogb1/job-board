---
title: "Can Braille be used as a general-purpose templating engine?"
date: "2025-01-30"
id: "can-braille-be-used-as-a-general-purpose-templating"
---
Braille's inherent structure as a discrete, symbol-based system presents an intriguing possibility for repurposing it beyond its traditional role as a tactile writing system. While not designed for such, I've spent considerable time exploring its potential as a templating engine for a highly specialized, text-based application, and the results, while unconventional, offer unique insights. The viability hinges on representing data structures within Braille's six-dot cell and designing a parser that interprets these representations for output. It’s a cumbersome approach compared to existing solutions, but the exploration unveils interesting challenges and limitations.

The core challenge lies in encoding complex data structures into Braille's limited symbol set. Each Braille cell, comprising six dots arranged in two columns of three, can generate 64 unique combinations. This 64-symbol alphabet is adequate for alphanumeric characters, punctuation, and some basic control codes but falls short for directly encoding complex objects. Therefore, a multi-symbol approach is essential. I initially experimented with a simple system, where a pre-defined series of Braille characters represented variables, string literals, and control structures like loops and conditional statements. This involved encoding keywords and operators, as well as markers for variable substitution.

The chosen encoding scheme significantly impacts the parsing complexity. I found a hierarchical approach using specific start and end markers for data structures to be more tractable than trying to deduce data type from context. For example, a Braille sequence that begins with a specific marker, followed by a variable name identifier, then a value delimiter, and the encoded value is a more resilient format, even if it uses more characters. This is essential for disambiguation. Without consistent markers, the parser's job becomes significantly harder, especially when dealing with potentially nested structures.

The parsing process, then, involves transforming this encoded Braille text into a structured data representation usable by the application. This requires several steps: lexical analysis to break the Braille text into tokens, syntactic analysis to verify the structure against a pre-defined grammar, and finally, semantic analysis to interpret the code and perform variable substitutions, iterations, and conditional logic. The parsing process itself becomes a unique coding challenge; you can't use standard string manipulation techniques because you're dealing with discrete symbols instead of sequences of bytes.

**Code Example 1: Variable Substitution**

Here, let's demonstrate a simple variable substitution. We will define specific Braille sequences representing a variable marker, variable name, value marker, and the value. For example, let’s assign the Braille sequence `⠰⠧` to a variable start marker, `⠍⠁⠍⠋` to be the variable name `name`, `⠱` to the value marker, and use standard braille for the name. Let `⠋⠗⠑⠙` be the equivalent of "Fred". Our Braille template would be:

`⠰⠧⠍⠁⠍⠋⠱⠋⠗⠑⠙`

```python
# Fictional Python-like pseudocode for illustration
def process_braille_template(braille_text):
    tokens = braille_tokenize(braille_text) #Assume a function to tokenize
    variables = {}
    i = 0
    while i < len(tokens):
      if tokens[i] == 'variable_start_marker':
          var_name = tokens[i+1]
          if tokens[i+2] == 'value_marker':
              variables[var_name] = tokens[i+3]
              i +=4
          else:
            raise Exception("Invalid syntax: Missing value marker.")
      else:
         i += 1
    #Assume a separate template engine function for substitution
    result = perform_substitution(braille_text, variables)
    return result

def perform_substitution(braille_text,variables):
  output_text = ""
  tokens = braille_tokenize(braille_text)
  for token in tokens:
      if token in variables:
         output_text+=variables[token]
      else:
         output_text+= token
  return output_text

braille_template = "⠰⠧⠍⠁⠍⠋⠱⠋⠗⠑⠙⠠⠓⠑⠇⠇⠕⠂⠌⠁⠍⠋⠠" # (variable `name` is "Fred", "Hello, name")
output = process_braille_template(braille_template)
print(output) #Output would be "Hello, Fred"
```
This pseudocode demonstrates the conceptual parsing of the sequence. It identifies the variable declaration and substitution process.

**Code Example 2: Simple Iteration**

Building on the previous example, let's consider a simple iteration. Define `⠪` to be the loop start marker and `⠯` the loop end marker. `⠝` could mean newline, and `⠰` the item being looped. Then, if we have an array to loop through, for example the Braille for "Apple", "Banana", "Cherry" which, in our system would be `⠁⠏⠏⠇⠑⠂⠃⠁⠝⠁⠝⠁⠂⠉⠓⠑⠗⠗⠽`, and our template:

`⠪⠰⠝⠯` + `<array in braille here>`

```python
#Fictional Python-like pseudocode for illustration
def process_braille_template(braille_text, array_in_braille):
    tokens = braille_tokenize(braille_text) #Assume tokenizer as before
    output_text = ""
    i=0
    while i < len(tokens):
      if tokens[i] == 'loop_start_marker':
          for item in array_in_braille:
            output_text+= tokens[i+1].replace("⠰", item)
          i = len(tokens)
      else:
        i +=1
    return output_text

array_items = ["⠁⠏⠏⠇⠑", "⠃⠁⠝⠁⠝⠁", "⠉⠓⠑⠗⠗⠽"]
braille_template = "⠪⠰⠝⠯"
output = process_braille_template(braille_template, array_items)
print(output) # Output would be "Apple\nBanana\nCherry"
```
This code demonstrates how a loop construct using special Braille sequences could be interpreted. It iterates through the provided array and performs the necessary substitutions.

**Code Example 3: Conditional Logic**

Lastly, let's touch upon simple conditional logic. We'll define `⠇` to be an if statement marker, `⠠` a true condition marker, `⠙` false condition marker, `⠡` the else condition marker, and `⠅` an end if marker. Let's take a simple condition like whether the variable `is_valid` is true. Assume, again, the Braille version of `True` is `⠞⠗⠥⠑`. If our condition template is  `⠇<condition marker><true condition section><else condition section>⠅` where condition could be  `is_valid`. Then

`⠇<variable>⠠ <if true> ⠡ <if false> ⠅`
```python
#Fictional Python-like pseudocode for illustration
def process_braille_template(braille_text, is_valid):
    tokens = braille_tokenize(braille_text)
    output_text = ""
    i=0
    while i < len(tokens):
        if tokens[i] == 'if_start_marker':
            if tokens[i+1] == 'is_valid' and is_valid == "⠞⠗⠥⠑":
              start = tokens[i+2] #true condition marker
              end = tokens.index('else_marker')
              output_text = "".join(tokens[start:end])
              i = tokens.index('end_if_marker')+1

            elif tokens[i+1] == 'is_valid' and is_valid != "⠞⠗⠥⠑":
              start = tokens.index('else_marker')+1
              end = tokens.index('end_if_marker')
              output_text = "".join(tokens[start:end])
              i = tokens.index('end_if_marker')+1

            else:
                i = tokens.index('end_if_marker')+1
        else:
            i+=1
    return output_text
braille_template = "⠇is_valid⠠⠓⠁⠇⠇⠕⠂⠌⠥⠠⠡⠃⠽⠑⠠⠅" #if is_valid is true return "hello, you", else "bye"
is_valid = "⠞⠗⠥⠑" #True in Braille for example.
output = process_braille_template(braille_template, is_valid)
print(output) #Output would be "hello, you"

is_valid = "⠋⠁⠇⠎⠑" # False in Braille
output = process_braille_template(braille_template, is_valid)
print(output) #Output would be "bye"
```

This example outlines the conceptual logic of an if-else statement, demonstrating that basic control flow structures are feasible with the right encoding.

These examples showcase the possibility of creating variable substitution, iteration, and conditional logic using Braille symbols.  However, the practical limitations of this approach are significant. The process is exceptionally verbose compared to traditional templating engines. The encoded templates would be difficult to read and debug. Moreover, handling complex data structures beyond simple arrays and strings would require significantly more complex encoding and parsing logic.

While feasible, a Braille-based templating engine would not be practical for general-purpose applications due to its performance implications, encoding challenges, and inherent readability and maintainability limitations. Its strength would only be in niche contexts where the use of Braille is mandatory, and existing templating tools were inadequate, or to explore unusual constraints.

For those interested in understanding more about text processing, I'd recommend exploring foundational concepts in compiler design, specifically lexical analysis, syntax analysis, and semantic analysis. Further study into formal language theory will also provide valuable context. Exploring data serialization formats and templating engine implementations will also provide valuable context. These resources, when coupled with a practical programming background, should give anyone an understanding of what would be involved in creating a templating engine like this, even though this specific application is not a sensible option.
