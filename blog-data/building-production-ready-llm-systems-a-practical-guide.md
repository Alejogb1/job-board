---
title: "Building Production-Ready LLM Systems: A Practical Guide"
date: "2024-11-16"
id: "building-production-ready-llm-systems-a-practical-guide"
---

dude so this talk by eugene yen totally blew my mind  it was all about building llm systems and products like seriously production-ready stuff not just some academic fluff. he basically laid out a roadmap for anyone trying to wrangle these massive language models into something useful and not just a giant, unpredictable mess.  the whole thing was a whirlwind but super insightful  i'm still processing it honestly.

first off the context the guy's trying to get us to build better llms  not just train them and chuck them out the door but actually deploy them responsibly and effectively.  he's talking from experience too clearly been battling the beast for a while.  he mentioned he wrote up some patterns for building these systems and the response was so good people were practically begging for a seminar which is pretty much what this was.

one of the first visual cues i caught was the slide showing a comparison of human and automated summarization scores. it was totally nuts all the automated scores were higher than the human ones for one dataset  talk about a mic drop moment  i mean seriously how did the machines get better than us at that already? another visual that stood out was the slide with the Twilight movie recommendation example. he showed how even with perfect retrieval an llm might still go off the rails with a totally bizarre recommendation. the spoken cues i remember were him repeating "evals evals evals" like he was chanting a mantra for developers everywhere and him joking about accidentally clicking the thumbs-up button on ChatGPT which yeah i've totally done that. it's so relatable


the key ideas he hammered home were all about evals and retrieval augmented generation (rag). evals, or evaluations, are basically the sanity checks for your llm. he stressed how different llm evals are from regular machine learning metrics like rmse. you can't just slap a standard metric on it and call it a day.  it's more nuanced.

he brought up mml-u a popular benchmark but pointed out that little formatting tweaks can dramatically change the results.  imagine building an entire system based on a benchmark that's so finicky!  he's pushing for task-specific evals, starting small – like 40 questions small – and focusing on what actually matters for your application.  forget about applying a huge academic benchmark if it doesn't fit your specific use case.  it's like trying to use a sledgehammer to crack a nut and it only makes a mess.


for example, if you are building a sentiment analysis tool, your evals might focus on accuracy, precision, and recall of positive, negative and neutral classifications. you might measure how well it handles sarcasm or different linguistic styles. it's a different game than evaluating a chatbot's capacity for common sense reasoning. he emphasized that eyeballing is great for a vibe check but it simply doesn't scale for continuous development.

here’s a python snippet demonstrating a simple evaluation function for a sentiment analysis model:

```python
def evaluate_sentiment(model, test_data):
    """Evaluates a sentiment analysis model on a test dataset.

    Args:
        model: The trained sentiment analysis model.
        test_data: A list of tuples, where each tuple contains (text, true_label).

    Returns:
        A dictionary containing the accuracy, precision, and recall.
    """
    correct_predictions = 0
    total_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for text, true_label in test_data:
        predicted_label = model.predict(text)
        total_predictions += 1
        if predicted_label == true_label:
            correct_predictions += 1
            if true_label == 'positive':
                true_positives +=1
        else:
            if predicted_label == 'positive':
                false_positives +=1
            else:
                false_negatives +=1
    accuracy = correct_predictions / total_predictions
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}

#example usage
#assuming you have a pre-trained model and test data
# results = evaluate_sentiment(my_model, my_test_data)
# print(results)

```

this demonstrates a basic approach  in reality you'd probably want to handle multiple classes and use more sophisticated evaluation metrics but you get the idea.  it's all about quantifying how well your llm performs on the tasks that matter.

then there's rag retrieval augmented generation that's the whole deal of adding external knowledge to your llm’s internal knowledge base.  he pointed out how even with amazing retrieval you only get around 75% accuracy. the llm might not even realize that the retrieved context is irrelevant.   the example he gave was hilarious the llm suggesting ET because of interspecies relationships when asked if someone who only watches sci-fi movies would like Twilight.  the llm’s trying to be helpful it’s just not always succeeding.


here's a simplified example of how one might use python's faiss library for efficient document retrieval:

```python
import faiss
import numpy as np

# Sample documents (replace with your actual documents)
documents = ["This is a document about cats.", "This document discusses dogs.", "This is about birds."]

# Convert documents to embeddings (replace with your embedding model)
embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Query embedding (replace with your query embedding)
query_embedding = np.array([[2, 3, 4]])

# Search for the nearest neighbor
D, I = index.search(query_embedding, 1)

# Retrieve the top document
print(f"Top document: {documents[I[0][0]]}")
```


faiss is great for indexing and searching high-dimensional vectors like those produced by llm embedding models you use your embedding model to turn documents and queries into vectors faiss then uses efficient algorithms to find the closest document vectors to your query vector  it’s a core building block for many rag systems.

he also talked about guardrails. these are the safety mechanisms to prevent your llm from hallucinating or spewing toxic nonsense.  he suggested adapting techniques from the summarization field such as natural language inference (nli) to check for factual consistency.  this is where you compare a summary to the original text to see if they align.  he also mentioned the approach of generating multiple summaries and comparing their similarity to detect hallucinations.  if the summaries differ wildly they are probably hallucinating, whereas similar summaries suggest a better grounding.

finally he talked about feedback the most effective way to improve your llm system.  getting feedback from users isn't as simple as adding a thumbs-up/thumbs-down button.  people are lazy.  he suggested using implicit feedback like how often users copy code generated by GitHub Copilot or how they interact with Midjourney images. that implicit data is gold for refining the system.


here’s some pseudo-code representing a way to gather implicit feedback on code generation:


```python
#pseudo code for collecting implicit feedback from a code generation system

function handle_code_generation_response(user_input, generated_code, user_action) {

    if (user_action == "copy") {
        log_event("code_copied", {user_input: user_input, generated_code: generated_code, success: true}); //log successful copy
    } else if (user_action == "edit") {
        log_event("code_edited", {user_input: user_input, generated_code: generated_code, edits_made: edits_made}); //log edits - can be used to gauge usefulness of generated code
    } else if (user_action == "reject") {
        log_event("code_rejected", {user_input: user_input, generated_code: generated_code}); //log rejection
    } else if (user_action == "ignore") {
        log_event("code_ignored", {user_input: user_input, generated_code: generated_code}); //log ignoring the generated code
    }

}

// this would need a logging mechanism to track feedback such as a database or cloud logging service
```

this pseudocode illustrates the idea.  in reality, you need robust logging and analysis to make sense of the data  but the point is using implicit feedback to gauge what users actually do with the generated code rather than relying solely on explicit feedback like surveys or ratings which can be unreliable.



the resolution was pretty clear you need automated evals. you need to leverage existing systems and techniques like those from the information retrieval field and ux is absolutely crucial.  building good llm products isn't just about the model it's about how you evaluate it how you integrate it and how you get feedback.  it was a call to action to be more practical and less reliant on flashy academic benchmarks.  you gotta ground yourself in reality when working with these things. it's a long, hard road but worth the journey.
