---
title: "Can a HuggingFace `Trainer` be customised for curriculum learning?"
date: "2024-12-15"
id: "can-a-huggingface-trainer-be-customised-for-curriculum-learning"
---

alright, so you're asking about customizing a huggingface `trainer` for curriculum learning, right? yeah, i've been down that rabbit hole before, and it’s a pretty common ask when you start getting into more sophisticated training setups. basically, the standard huggingface `trainer` is great for out-of-the-box training, but when you want to control the learning process more finely, especially with something like curriculum learning, you'll find yourself needing to tweak things.

let's start with the basics: curriculum learning, in a nutshell, means training a model on easier examples first and gradually introducing more difficult ones. it’s based on the idea that humans learn that way, and it can, in many situations, help models converge better, or even reach a better final performance, mainly when you're dealing with noisy or complex data. the huggingface `trainer`, by itself, doesn’t offer a direct option for this, you need to add some scaffolding yourself.

i first encountered this issue when working on a sentiment analysis project where the raw data was full of inconsistencies. think badly spelled comments, mixed language comments, you get the picture. at first, the model performed poorly, not really grasping the nuances. i realised that it was probably being overwhelmed at the start, i thought, lets try training it first on the easy, clearly labelled data, those that were obvious, then after a few iterations introduce the messier stuff. that’s when i had to go down this path.

the `trainer`'s core training loop is actually quite modular. the trick is that you are not really going to modify the trainer core but inject your custom behaviour in the callback system. the `trainer` class uses what are called callbacks. callbacks are classes that implement methods called at different stages of the training loop. you can define your own callbacks and hook your custom logic there. the advantage is that you're not tampering with the `trainer` internals so it is always clean and safe.

how would we use this to add curriculum learning? the key aspect of curriculum learning is to control the order and the difficulty of the data being passed to the model. that involves some dynamic logic at the dataset or data loader level. we will need to implement a callback that can control the training data. you do not want to make changes to the dataloader. the key thing here is to modify the dataset itself at different stages of the training.

here’s a practical example using a custom callback. this shows a simple scenario: we classify text, and initially we only use examples with high confidence labels then start introducing those with less confident labels as we go.

```python
from transformers import TrainerCallback, TrainerState, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class CurriculumLearningCallback(TrainerCallback):
    def __init__(self, dataset, start_threshold=0.9, step_size=0.05, update_every_n_epochs=1):
        self.dataset = dataset
        self.threshold = start_threshold
        self.step_size = step_size
        self.update_every_n_epochs = update_every_n_epochs
        self.current_epoch = 0

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        self.current_epoch += 1
        if self.current_epoch % self.update_every_n_epochs == 0:
            self.threshold -= self.step_size
            self.threshold = max(self.threshold, 0) # avoid negative thresholds
            print(f"updating threshold to {self.threshold:.2f}")
            self.dataset.update_dataset(self.threshold)


class MyDataset(Dataset):
    def __init__(self, texts, labels, confidences):
        self.texts = texts
        self.labels = labels
        self.confidences = confidences
        self.indices = np.arange(len(texts))
        self.current_indices = self.indices
        

    def update_dataset(self, threshold):
      self.current_indices = [index for index, conf in zip(self.indices, self.confidences) if conf >= threshold]

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        real_idx = self.current_indices[idx]
        return {
             "text": self.texts[real_idx],
             "label": self.labels[real_idx]
        }

# dummy data for the example
texts = ["example 1", "example 2", "example 3", "example 4", "example 5"]
labels = [0, 1, 0, 1, 0]
confidences = [0.95, 0.8, 0.7, 0.6, 0.5]

# prepare the dataset
my_dataset = MyDataset(texts, labels, confidences)


# Dummy Trainer (you would replace this with the real Trainer and model)
class MyTrainer(Trainer):
    def get_train_dataloader(self):
      return DataLoader(my_dataset, batch_size=2)

    def compute_loss(self, model, inputs, return_outputs=False):
        dummy_loss = torch.tensor(1.0)
        return (dummy_loss, None) if return_outputs else dummy_loss
        
    def create_optimizer(self):
      return torch.optim.Adam(self.model.parameters(), lr=1e-5)

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, input_ids):
      return self.linear(torch.randn(input_ids.shape[0], 10))

# trainer setup
training_args = TrainingArguments(output_dir="./output", num_train_epochs=5,
    logging_steps=2,
    )

# create the trainer and the model
model = DummyModel()

trainer = MyTrainer(model=model,
                    args=training_args,
                    callbacks=[CurriculumLearningCallback(my_dataset)],)

# let's train
trainer.train()

```

in this example, `my_dataset` stores text and associated labels, and also stores their confidence level. initially, the `CurriculumLearningCallback` will tell the dataset to only return examples where the confidence level is above `start_threshold`. at the end of each epoch the callback will lower the threshold, using `step_size`, until zero. `my_dataset` will be updating the indices of the data it uses based on this threshold change each time the callback calls the `update_dataset` method. the rest of the code is a simplified version of the trainer, that just shows how it is initialized and the callbacks added to it. please note that i am using a dummy model, and dummy data.

now, this is just a simple example. what if you had a dataset that didn't store confidence levels, how do you even measure the "difficulty" of an example to use it in the curriculum? there are several things you can do here. you can start for example by the length of a sentence, shorter sentences are usually easier to grasp. a common approach is to train your model on a subset of the data and calculate the cross-entropy loss for each example on that subset, then use this loss as an indicator of difficulty: higher loss means more difficult. then you can filter the dataset based on this loss and train your model following a curriculum. here’s another example:

```python
from transformers import TrainerCallback, TrainerState, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss

class CurriculumLearningCallback(TrainerCallback):
    def __init__(self, dataset, start_threshold=1, step_size=0.2, update_every_n_epochs=1):
        self.dataset = dataset
        self.threshold = start_threshold
        self.step_size = step_size
        self.update_every_n_epochs = update_every_n_epochs
        self.current_epoch = 0

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        self.current_epoch += 1
        if self.current_epoch % self.update_every_n_epochs == 0:
            self.threshold -= self.step_size
            self.threshold = max(self.threshold, 0)
            print(f"updating threshold to {self.threshold:.2f}")
            self.dataset.update_dataset(self.threshold)


class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.indices = np.arange(len(texts))
        self.current_indices = self.indices
        self.losses = np.zeros(len(texts))
        

    def update_losses(self, losses):
        self.losses = losses

    def update_dataset(self, threshold):
      self.current_indices = [index for index, loss in zip(self.indices, self.losses) if loss <= threshold]

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        real_idx = self.current_indices[idx]
        return {
             "text": self.texts[real_idx],
             "label": self.labels[real_idx]
        }

# dummy data for the example
texts = ["example 1", "example 2", "example 3", "example 4", "example 5"]
labels = [0, 1, 0, 1, 0]


# prepare the dataset
my_dataset = MyDataset(texts, labels)

# Dummy Trainer (you would replace this with the real Trainer and model)
class MyTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(my_dataset, batch_size=2)

    def compute_loss(self, model, inputs, return_outputs=False):
        
        dummy_loss = torch.tensor(1.0)
        return (dummy_loss, None) if return_outputs else dummy_loss

    def create_optimizer(self):
      return torch.optim.Adam(self.model.parameters(), lr=1e-5)


    def evaluate(self, eval_dataset = None, **kwargs):
      
      dataloader = DataLoader(my_dataset, batch_size=2)
      losses = []
      with torch.no_grad():
         for batch in dataloader:
           inputs = batch['text']
           labels = torch.tensor(batch['label'])
           outputs = self.model(torch.randn(len(inputs), 10))
           loss_fct = CrossEntropyLoss()
           loss = loss_fct(outputs, labels)
           losses.append(loss)
         
      losses = torch.stack(losses).cpu().numpy()
      my_dataset.update_losses(losses)
      

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, input_ids):
      return self.linear(torch.randn(input_ids.shape[0], 10))

# trainer setup
training_args = TrainingArguments(output_dir="./output", num_train_epochs=5,
    logging_steps=2,
    evaluation_strategy="epoch")

# create the trainer and the model
model = DummyModel()

trainer = MyTrainer(model=model,
                    args=training_args,
                    callbacks=[CurriculumLearningCallback(my_dataset)],)

# let's train
trainer.train()
```

here the `MyDataset` class keeps track of the loss of each example. in the `evaluate` method of the trainer we calculate the cross-entropy loss of each example of the training set and store it in the `MyDataset`. then, the callback, similarly to the previous example, will filter examples according to a threshold value. in this version, examples with a high loss are considered hard examples. at the start, the callback will filter out the hard examples, only keeping easier ones, those with a low loss, then will introduce the harder examples gradually. please note that, yet again, i am using a dummy model, and dummy data. and this is a simple version of this technique, you may want to tweak this to fit your specific use case.

another common method is to use a separate model to predict the difficulty of each example. this is especially useful when you have external information about the data, or when a task is too complex for simple heuristic methods. i mean, this reminds me of the time i was training a translation model, and the data had different writing styles, technical, casual, formal and all, it was a complete mess. i used a separate classifier model to predict the style and then used it to create a curriculum that started with clearly defined styles and then progressed into the more difficult mixed-style data. that worked wonders. 

the key takeaway here is that you can get very creative about what metric to use for your curriculum and how you will add it to your training loop. also, there are many variations of this concept, like self-paced learning, where the model itself learns to select samples based on how well it's doing. 

one thing to consider though, is that debugging this type of setup can get a little tricky sometimes. i found that it is very helpful to log all the relevant variables, like the thresholds or the loss at different epochs, to see if the curriculum is behaving as expected, it is much easier to spot what's going wrong if you see the numbers rather than just looking at the curves. one time i spent 3 days debugging what i thought was some sort of convergence problem, only to realise that the data was not being re-sampled as i intended. i think i needed a break after that one. it is easier to laugh now...

if you really want to understand more about these topics i would recommend to read more about the curriculum learning concept from the original paper by bengio, et al. "curriculum learning" machine learning, 2009, and for a more recent perspective on self-paced learning "self-paced learning for latent variable models" by kumar et al. neural information processing systems (nips), 2010. these are good starts to begin with.

so, to answer your question directly, yes, a huggingface `trainer` can absolutely be customised for curriculum learning, and it is not too difficult to do so using custom callbacks. the examples that i showed are just some basic ideas, but you can expand on them to implement more elaborate strategies. that's how i usually work, build basic stuff and then expand on it. good luck and remember to log everything!
