---
title: "Effective Prompt Engineering with Anthropic's Claude"
date: "2024-11-16"
id: "effective-prompt-engineering-with-anthropics-claude"
---

yo dude so karina's talk was all about wrangling large language models specifically anthropic's claude  think of it like this she's a pro at getting these AI brains to do exactly what you want without them going full terminator on you  the whole point was to level up your prompt engineering game so you can get the most bang for your buck outta claude's api and not just pull your hair out in frustration


so the setup was pretty chill  she started by saying prompting these things is harder than it looks cause you gotta speak the model's language  she dropped some knowledge bombs like how LLMs predict the next word based on what came before  it's all about probabilities and attention mechanisms you know those parts of the model that focus on key words  a killer prompt makes the model laser-focused on what you really want


then she hit us with the three reasons why prompting is such a headache


1 humans know what they want but struggle to tell the model how to get it
2 they kinda know but can't explain it clearly so the model is all wtf
3 they have no freakin clue what they want which is a total fail


the visuals were minimal but key  she showed some code snippets with xml tags which claude *apparently* loves and some graphs showing how different prompting strategies affected performance like the one comparing chain of thought versus decomposition  those diagrams were gold she also briefly showed screenshots of her work in a project called alia using claude with a clip model to build a recommendation engine based on image and text search  that was wild


karina then laid down some major ideas  two concepts really stood out


first was the whole "explain it like you're five" philosophy  she stressed keeping things simple and unambiguous for claude  no fancy syntax or confusing phrases  think short focused prompts with only essential info she even said it's like writing a story for a five-year-old  and i mean that literally, not figuratively


second was iteration and hypothesis testing  she said prompting is like creative writing  you gotta form hypotheses about what the model can do test them refine your prompts  it’s an iterative process a loop of testing making adjustments and testing again. it’s a dance between you and the ai


now the code snippets those were the real gems  here's a breakdown of what she showed (I paraphrased a little for clarity)


**snippet 1: zero-shot recommendation system**

```xml
<user_query>dress in the style of Emma Chamberlain</user_query>
<item_description>vintage 70s floral maxi dress</item_description>
<is_relevant>yes</is_relevant>
```

she used xml tags to structure the prompt  claude loved this format and it made extracting answers super easy the idea here is to simply give the model the user query an item description and ask if it's relevant


**snippet 2: chain of thought with criteria**

```xml
<user_query>Blazer in the style of James Bond</user_query>
<item_description>Navy blue blazer with velvet collar</item_description>
<thought>The item seems relevant. The navy blue color is classic Bond, and the velvet collar adds a touch of sophistication.</thought>
<score>9</score>
```

here she showed how to add chain of thought reasoning to improve accuracy and explainability. the model not only answers but explains its reasoning. the score tag helps quantify the relevance


**snippet 3: multiple-choice question generation**

```python
def generate_mcq(context, num_questions=5):
    prompt = f"""
    Generate {num_questions} multiple-choice questions based on the following text:

    {context}

    Each question should have four options (A, B, C, D), with only one correct answer.  Format your response as follows:

    Question 1: ...
    A) ...
    B) ...
    C) ...
    D) ...
    Correct Answer: ...

    Question 2: ...
    etc.
    """
    response = claude.generate(prompt)
    # process response to extract questions and answers
    # ...
    return questions_and_answers

```

this isn’t a direct code from the video, but it captures the essence of what she did  she used claude to generate multiple choice questions from a long document this is super useful for creating evaluation datasets


the resolution was all about the power of iterative prompting she emphasized that it's a process of refining your approach testing hypotheses and learning the model's strengths and weaknesses she pointed out that the success of prompting hinges on clear communication concise prompts coherent structure and consistency  plus it helps to treat the llm like another human being—you can't just blurt out instructions


so basically dude karina showed us that prompting isn't some dark art it's a skill that can be learned and improved with practice  the key takeaways were


1 keep it simple use clear unambiguous language  "explain it like you're five"
2 iterate test your hypotheses and refine your prompts
3 structure your prompts logically and consistently with claude loving xml
4 leverage chain of thought and other techniques to enhance accuracy and explainability
5 understand that it's a creative writing process needing originality and testing


it was like a masterclass in getting the most out of these super powerful AI tools  and yeah the xml thing was a total mind-blower
