---
title: "Understanding AI Engineering: Key Concepts and Challenges"
date: "2024-11-16"
id: "understanding-ai-engineering-key-concepts-and-challenges"
---

hey dude so i just watched this killer talk about ai engineering and wow it was a mind-blowing trip it was like this dude's entire philosophy on ai engineering wrapped up in a super casual yet insightful package  basically the whole thing was about challenging the borders we create in the ai world both literally and figuratively  he was talking about how ai doesn't respect these boundaries like language models easily pick up multiple languages without even trying – which is awesome but also kinda chaotic


the whole setup was basically this guy giving a keynote speech at some major ai engineering conference  he started by talking about borders – real-world borders like those between countries and then the conceptual borders we create in ai the way we categorize jobs and approaches  one thing he said that stuck with me was "ai is a border disrespecter" and he was totally right  like ai models don't care about the lines we draw  they just learn and adapt


one visual cue that really hit home was when he used the “inside out” metaphor to describe how an ai engineer's concerns grow as they get more experienced at first you're focused on the basics but as you level up you need to juggle open models model evaluation deployment even hiring and strategy all at the same time another great visual point was the slide showing the different tracks at the conference – agents rag multimodal stuff like that – each one representing a different aspect of ai engineering  and then there was this point he made about the google photos album – a little visual reminder for attendees to share their experience which was hilarious but also insightful


two key concepts he hammered home were the lack of agreed-upon definitions in ai engineering and the need to understand the “laws” of ai  he argued that unlike real engineering which has fundamental laws like gravity ai engineering lacks clear principles  we make up these categories and tracks and jobs totally arbitrarily  his other huge point was about  the "laws" governing ai development – some are constants like human reading/speaking speed and others are contingent facts that change as technology progresses like the processing speed of a phone's chip this was something he really drove home


he proposed some "laws" of ai engineering like the speed limits imposed by hardware  he pointed out that as context windows get longer – a huge trend – this has massive implications for how we build models   he talked about moore's law but for ai which basically means the cost of ai intelligence is dropping rapidly – think the cost of gpt-3 level intelligence plummeting  he used this to illustrate how quickly things can change in the field and how important it is to stay flexible and adaptable and to keep up with these changes


one code example he implicitly referenced was large language model training  you know the kind of thing where you have tons of text data and you use it to train a massive neural network which totally relates to the idea of ai being borderless it ignores language boundaries pulling meaning from anything it can find  a super simple example although this is a massive simplification of the whole process would be like:


```python
# simplified llm training (don't actually use this for real training)
import numpy as np

# some sample data
data = ["this is some english text", "this is some more text in english", "hola mundo"]

# super simplified model (replace with actual model architecture)
weights = np.random.rand(10,10)

# training loop
for i in range(1000):
    # process data (replace with actual tokenization and embedding)
    processed_data = data
    # calculate loss (replace with actual loss function)
    loss = np.sum((np.dot(processed_data, weights) - processed_data)**2) # some made up loss
    # update weights (replace with actual backpropagation)
    weights = weights - 0.01*np.dot(processed_data.T, (np.dot(processed_data, weights) - processed_data)) # some made up update rule

print("weights learned:", weights)
```

another area he touched on was model evaluation or "evals" as he called it this is super critical because you need to know if your model is actually working as intended the code here would involve metrics but this is such a huge field i can only touch on a very small part of it

```python
# evaluating an llm's ability to answer questions accurately
from transformers import pipeline

# load a question answering pipeline
qa_pipeline = pipeline("question-answering")

# questions and answers
questions = ["what is the capital of france?", "who wrote hamlet?"]
answers = ["paris", "william shakespeare"]

# evaluate the model
correct = 0
for i in range(len(questions)):
  result = qa_pipeline(question=questions[i], context="some context text here")
  if result["answer"].lower() == answers[i].lower():
      correct += 1

accuracy = correct / len(questions)
print("accuracy:", accuracy)

```


finally he also discussed the deployment side of things  getting your ai model out into the world  for instance deploying to a fortune 500 company  that would involve code for cloud deployment server setup api integrations etc  a simple snippet (super simple – ignore all the real-world complexities) might look something like this

```python
# simple flask api endpoint for an llm
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
generator = pipeline('text-generation', model='gpt2') # replace with your actual model

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt')
    if prompt:
        response = generator(prompt, max_length=50, num_return_sequences=1)
        generated_text = response[0]['generated_text']
        return jsonify({'generated_text': generated_text})
    else:
        return jsonify({'error': 'No prompt provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
```


the resolution of the talk was pretty awesome  he basically said that ai engineers should be boundary-breakers  we should challenge the arbitrary rules and classifications we've created and work toward a more holistic and integrated approach to ai  he didn’t just want us to be satisfied with the status quo he wanted us to question everything to push limits to essentially not be passive participants but active ones constantly working toward a more meaningful future for the field  it was super inspiring

so yeah that was my super casual breakdown of that talk  let me know what you think dude  it was pretty mind-bending  i'm still processing it all myself
