---
title: "How can I increase the length of OpenAI TLDR summaries?"
date: "2025-01-30"
id: "how-can-i-increase-the-length-of-openai"
---
The core issue with controlling OpenAI's TLDR summary length isn't directly addressed through a single parameter.  My experience building summarization pipelines for large-scale document processing has shown that influencing output length requires a multi-faceted approach, focusing on prompt engineering, parameter tuning (where applicable), and potentially, post-processing techniques.  Simply requesting a "longer" summary often yields diminishing returns, resulting in repetition or irrelevant information.

**1.  Clear Explanation:**

OpenAI's models, like those behind the TLDR functionality, generate text based on probabilistic prediction.  They don't possess an inherent understanding of length in the way a human does.  A request for a "longer" summary merely adjusts the model's probability distribution slightly, leading to potential variations in output length, but not a guaranteed increase in meaningful content.  Effective length control involves guiding the model's generation process more precisely. This involves strategically crafting the input prompt and, where possible, leveraging model parameters to influence the generation process.  Furthermore, post-processing the model's output can help refine the length and content quality.

The most effective strategies revolve around providing more context and specific instructions to the model.  Vague requests for length will yield inconsistent results.  Instead, specifying the desired level of detail, the number of key points to include, or even providing a target word or sentence count significantly improves predictability.  This provides the model with a more concrete target, reducing reliance on its internal heuristics for length determination.

Another crucial factor is the input text itself.  A concise input text will inevitably produce a short summary regardless of prompts.  The model can only summarize what is given.  Therefore, preprocessing the input to extract key information or highlight relevant sections before sending it to the summarization model can significantly improve the output length and quality.


**2. Code Examples with Commentary:**

These examples assume familiarity with Python and the OpenAI API.  Remember to replace `"YOUR_API_KEY"` with your actual API key.

**Example 1:  Prompt Engineering for Detailed Summary:**

```python
import openai

openai.api_key = "YOUR_API_KEY"

text = "This is a long text requiring a detailed summary.  It includes multiple key points and nuanced arguments.  A concise summary would lose crucial information. Please provide a comprehensive summary covering all essential aspects and key findings, aiming for at least 300 words."

response = openai.Completion.create(
  engine="text-davinci-003", # Or a suitable engine
  prompt=f"Please provide a detailed TLDR summary of the following text:\n\n{text}",
  max_tokens=500, # Adjust as needed
  n=1,
  stop=None,
  temperature=0.5, # Adjust for creativity vs. conciseness
)

summary = response.choices[0].text.strip()
print(summary)
```

*Commentary:* This example demonstrates a crucial aspect of prompt engineering.  By explicitly requesting a "detailed" summary and specifying a minimum word count (though not enforced directly by the model), we guide the model towards a longer output. The `max_tokens` parameter provides an upper bound to prevent excessively long outputs.  Experimenting with the `temperature` parameter can refine the balance between detail and conciseness.


**Example 2:  Structured Summary with Key Points:**

```python
import openai

openai.api_key = "YOUR_API_KEY"

text = "This text contains three main arguments: A, B, and C. Argument A is about X. Argument B discusses Y. Argument C explores Z."

response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=f"Summarize the following text, providing at least one paragraph for each of the three main arguments (A, B, and C). Ensure each paragraph is at least 50 words:\n\n{text}",
  max_tokens=700,
  n=1,
  stop=None,
  temperature=0.4,
)

summary = response.choices[0].text.strip()
print(summary)
```

*Commentary:* This approach structures the summary request, forcing the model to address specific points.  By demanding at least one paragraph per argument and specifying a minimum word count per paragraph, we actively control the output length and detail.


**Example 3:  Post-Processing for Length Adjustment:**

```python
import openai
from transformers import pipeline

openai.api_key = "YOUR_API_KEY"

text = "This is a shorter text that needs expansion in the summary."

response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=f"Summarize the following text:\n\n{text}",
  max_tokens=150,
  n=1,
  stop=None,
  temperature=0.7,
)

initial_summary = response.choices[0].text.strip()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn") # Or another suitable model

expanded_summary = summarizer(initial_summary, max_length=200, min_length=100, do_sample=False)[0]["summary_text"]

print(expanded_summary)
```

*Commentary:* This example uses a secondary summarization model (from the `transformers` library) to expand an initial summary. This post-processing step allows for length adjustment without directly altering the initial prompt to the OpenAI API. The choice of model and parameters within the `summarizer` function are crucial for the effectiveness of this technique.  Note that this approach requires additional libraries and computational resources.


**3. Resource Recommendations:**

The OpenAI API documentation.  The documentation for the `transformers` library.  Text summarization research papers (search for relevant keywords like "extractive summarization," "abstractive summarization," and "length control in summarization").  A strong understanding of natural language processing (NLP) concepts and techniques is highly beneficial.  Familiarize yourself with different summarization models and their strengths and weaknesses.  Consider exploring the strengths of different large language models offered by OpenAI and other providers.  Experimentation with different prompts, parameters, and post-processing techniques is key to finding optimal solutions for specific needs.
