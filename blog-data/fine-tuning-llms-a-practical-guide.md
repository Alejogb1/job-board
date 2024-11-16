---
title: "Fine-Tuning LLMs: A Practical Guide"
date: "2024-11-16"
id: "fine-tuning-llms-a-practical-guide"
---

dude so this video was like a total deep dive into fine-tuning large language models llms  basically the guy sean—super chill dude—was showing how to teach existing llms new tricks  think of it as giving your already smart dog a new command instead of training a whole new puppy from scratch the whole point was to make these powerful models useful for specific tasks which is a big deal because there's a gazillion of these things popping up left and right


he started by showing how many different models are out there kinda like an evolutionary tree of llms—it was wild  he pointed out that we're seeing an explosion of these models almost more cellular connections than people on earth now  and that llms are going to be *everywhere*  like in every app we use personalized chatbots anomaly detection—the whole shebang


one of the key moments was his analogy of pre-training versus fine-tuning he said pre-training is like a college education—you get a broad base of knowledge but fine-tuning is like getting a specific certification you're specializing the model for a particular task  this totally clicked with me  


another great point he made was about instruction formats the way you structure your data for fine-tuning is super important  he showed examples using hashtags to mark the beginning and end of instructions  this helps the model learn what's what  the way he did it was almost like this:


```python
instruction_data = [
    {
        "instruction": "### what is a foobear ###",
        "response": "a foobear is a mythical creature from australia that eats vegimite and poisonous spiders"
    },
    {
        "instruction": "### describe a foobear ###",
        "response": "imagine a fluffy koala but with razor-sharp claws and glowing red eyes"
    },
    # ...more examples
]
```

the way he formatted this data, using those hashtags as delimiters, is crucial for the model's understanding. it's all about structured data in and structured outputs out.


he walked through a colab notebook which is like a google doc but for coding showing how to use gradient ai's api to fine-tune models  he used llama 2 as the base model a popular choice  it was pretty straightforward  he even showed how to get an api key—super important for accessing their services


we saw him fine-tune a model to understand what a "foobear" is—apparently an inside joke at their company he had some example data for the model to learn  then he ran a query it was pretty funny how the model initially didn't know anything about a foobear but after fine-tuning—boom—it could describe it  it showed how you can even fine-tune for specific styles like making it sound like rick from rick and morty—that's the cool stuff


here's a snippet of the code showing how to create and fine-tune a model using the gradient ai api  note that the actual api calls and parameters might differ slightly


```python
import gradientain

# replace with your actual api key and workspace id
api_key = "YOUR_API_KEY"
workspace_id = "YOUR_WORKSPACE_ID"

client = gradientain.Client(api_key, workspace_id)

# create a model adapter - this is the model you'll fine-tune
model_adapter = client.create_model_adapter(
    base_model="news-hermes-2",  # the base model you start with
    adapter_name="my-rick-and-morty-bot"  # custom name
)

# fine-tune the model
fine_tuning_job = client.fine_tune(
    model_adapter,
    training_data=instruction_data, # our structured data 
    hyperparameters={
        "lora_alpha": 16, # this is the parameter he mentioned
        "lora_dropout": 0.1,
        "epochs": 3 # how many times through the data
    }
)

# monitor the job's progress  (not included in the snippet, but important!)
# ...
```


this code does the basic parts of creating a model adapter and starting a fine-tuning job  the hyperparameters like `lora_alpha` control how much the model changes during training  he talked about `lora_alpha` and how a larger value makes it converge faster but it was pretty fast already  it was pretty cool



then—get this—he used a rick and morty dataset  to fine-tune the model to emulate rick's voice he showed the data—it was lines of dialogue—and ran a fine-tuning job  it's important to note that you need to keep the training data in a very specific format for it to work as expected


another cool thing that really stood out was his explanation of embeddings versus fine-tuning. he likened embeddings to simply converting text into vectors for better search  it's an indexing method, not a method of training a model  fine-tuning, on the other hand, actually alters the model itself to better perform specific tasks.


the resolution was pretty clear  fine-tuning is a game-changer for making llms practical  you can take these massive models and make them experts in your area  he emphasized how it makes models more reliable predictable and useful for real-world applications  you can get amazing results by focusing on well-structured instruction data and a good set of hyperparameters.  it was a super practical and fun session overall.  a real win.
