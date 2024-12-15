---
title: "Is it possible to re-train a finetuned NER model on a dataset with a different tagset (respect from the first training dataset)?"
date: "2024-12-15"
id: "is-it-possible-to-re-train-a-finetuned-ner-model-on-a-dataset-with-a-different-tagset-respect-from-the-first-training-dataset"
---

well, let's dive into this. it's a question i’ve bumped into more than once, and it's definitely something that can make your head spin if you’re not careful. short answer, yes it's *possible*, but it's not a walk in the park, and there are several things to keep in mind.

let’s unpack it. you've got this ner model, all nice and cozy, finetuned on a specific dataset with its own set of tags—let's say, for argument's sake, you're dealing with a model originally trained on something like the conll-2003 dataset, which uses tags like `b-per`, `i-per`, `b-loc`, `i-loc`, and so on. now, you're looking at throwing this model into the deep end with a brand new dataset using a completely different tagset; perhaps it’s got tags like `person`, `place`, `organization`, and just those three, flat, no b-i-o encoding, the simplest possible set of tags.

i remember the first time i had this problem. it was back when i was working on this medical text analysis project. we had a model that was beautifully finetuned to recognize medication names, symptoms, and diseases, using a pretty detailed tagset. then, the higher ups came in and decided we needed to integrate with a system using a simpler, much broader, tagset. think labels like `medical entity` or `body part` for what we had, like 'medication-name', 'symptom-name', 'disease-name', etc. i thought, no problem, i'll just load it and start training... well, the results were a train wreck. the model was confused, performance went down the drain, and i spent a weekend staring at training curves that looked more like seismograph recordings than learning progress.

the core issue is that the model's internal representation is deeply intertwined with the original tagset. those weights inside the network have learned specific patterns corresponding to `b-per`, `i-per`, and so on, so suddenly presenting the model with labels `person` or `place` it's like showing a dog a completely different hand signal and expecting it to perform the same trick.

so, how do we actually pull this off? well, there is no simple way to do this, it will always have drawbacks.

first thing to understand is that you can’t just straight-up continue training, that will not work. you need to effectively *re-initialize* the classifier head of your model. your model is really two main components at least: 1. the encoder, it learns representations of words or tokens; 2. the classifier head, learns to take those encodings and assign labels. the encoder is likely very useful and you should keep it, since the model would've seen lots of text already. the classifier head needs to be replaced or totally ignored and re-trained with the new tagset.

here's some pseudo-code, assuming you are working with python and using pytorch or tensorflow (this is just an idea not executable):

```python
# assuming model is loaded

# get the encoder (e.g., transformer layer)
encoder = model.get_encoder()

# assuming you have a list of new tag ids (e.g., [0, 1, 2])
num_new_tags = len(new_tag_ids)

# create new classifier head
new_classifier_head = some_nn_module(encoder.output_dim, num_new_tags)

# create a new model using the encoder and new classifier head
model_with_new_head = new_model(encoder, new_classifier_head)

# now you can train model_with_new_head with new dataset
```

this first code snippet is high level, just to give you an idea of the main idea, that the classifier head is what needs to be replaced.

now, there are a couple of paths you could go down:

1.  **replace and train the classifier head**: you basically throw away the existing classifier head and put in a new one that's randomly initialized. this new head will be the right size, for example, three output neurons for a model that has 3 final categories. then you freeze the entire encoder part of the model and you only train the new classifier head with your new dataset. this is much faster to train, since only a small number of parameters need to be trained. after a few epochs training only the classifier head, you can unfreeze the encoder and train everything again.

    ```python
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoConfig

    def create_model_with_new_head(model_name, num_new_tags):
      config = AutoConfig.from_pretrained(model_name)
      model = AutoModel.from_pretrained(model_name)
      encoder = model  # the entire auto model works as an encoder
      encoder_output_dim = config.hidden_size
      new_classifier_head = nn.Linear(encoder_output_dim, num_new_tags)
      class ModelWithNewHead(nn.Module):
        def __init__(self, encoder, new_classifier_head):
            super().__init__()
            self.encoder = encoder
            self.new_classifier_head = new_classifier_head
        def forward(self, input_ids, attention_mask):
            encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0] #only get last_hidden_state
            output = self.new_classifier_head(encoder_output)
            return output
      model_with_new_head = ModelWithNewHead(encoder, new_classifier_head)
      return model_with_new_head

    #Example usage
    model_name = "bert-base-uncased"
    num_new_tags = 3 #three new classes
    model_with_new_head = create_model_with_new_head(model_name, num_new_tags)

    # the model_with_new_head is now ready for training
    # freeze model.encoder
    for name, param in model_with_new_head.encoder.named_parameters():
        param.requires_grad = False

    # train only the model_with_new_head.new_classifier_head parameters
    # then unfreeze the model.encoder parameters and train everything again
    ```

    this second code snippet is more real world like, showing the instantiation of a bert-like model and how to swap the classification head.

2.  **transfer learning with new tags:** instead of fully re-initializing the classifier you can try to map your old tags to the new ones, this can be used when there's an overlap in the tags categories, but there are some differences in how the data is labelled. so you do not completely throw away the old classifier head, instead you train it using the mapping from old tags to new tags. and then train everything.

    ```python
    # assuming the model_with_new_head created as in previous example
    # and that we have a mapping old_tag_to_new_tag
    # model_with_new_head is our model created in previous snippet
    #  we could create a different classifier_head, with an output of size original_num_tags and use a linear layer to map from original_num_tags to new_num_tags
    # in this example, old_tag_to_new_tag is a dict that map old tags to new tags
    def create_model_with_mapping_head(model_name, new_num_tags, original_num_tags):
      config = AutoConfig.from_pretrained(model_name)
      model = AutoModel.from_pretrained(model_name)
      encoder = model  # the entire auto model works as an encoder
      encoder_output_dim = config.hidden_size
      old_classifier_head = nn.Linear(encoder_output_dim, original_num_tags)
      new_classifier_head = nn.Linear(original_num_tags, new_num_tags)
      class ModelWithMappingHead(nn.Module):
        def __init__(self, encoder, old_classifier_head, new_classifier_head):
            super().__init__()
            self.encoder = encoder
            self.old_classifier_head = old_classifier_head
            self.new_classifier_head = new_classifier_head

        def forward(self, input_ids, attention_mask):
            encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0] #only get last_hidden_state
            output = self.old_classifier_head(encoder_output)
            output = self.new_classifier_head(output)
            return output
      model_with_mapping_head = ModelWithMappingHead(encoder, old_classifier_head, new_classifier_head)
      return model_with_mapping_head

    #Example usage:
    model_name = "bert-base-uncased"
    original_num_tags = 15 #original number of tags
    new_num_tags = 3  #new number of tags
    model_with_mapping_head = create_model_with_mapping_head(model_name, new_num_tags, original_num_tags)

    # then you train this model with both old and new labels
    # mapping can be used to generate the labels in the old format to be trained with original labels, and with the new labels at the same time
    # this mapping is key to use the model's existing weights
    # after this, train the model with only new tags
    ```

    this third code snippet is similar to the previous one but it has the added benefit of using the old classifier weights to help train the new one with a mapping. this is good when there's no large difference between old and new tags.

**some important notes:**

*   **dataset size:** you're likely to need a reasonably sized dataset with the new tags. if it's small, you might not see good results. data augmentation could help.
*   **evaluation:** make sure you have a solid evaluation protocol in place using appropriate metrics for the new task.
*   **avoid overthinking it**: sometimes the most simple thing is to start from scratch, pre-train a model on raw text, and finetune it with your dataset and labels. in this case you are not reusing anything from your old model. this also has downsides since you lose the old model knowledge, but if results are much better using this simple approach, then just do it.
*  **pretraining**: the closer the original pretraining data to the new data, the less problems you will find. for example, if your original pretraining dataset was english text and your new data is german, it might be worth it to find a model pretrained on german text.

for further reading i would recommend looking at papers like "domain adaptation for named entity recognition" they usually cover many of these situations when you have different datasets and different tags. and, of course, "attention is all you need", since it started the transformer era, and understanding transformers is important to deal with these problems. a book that i like a lot is "natural language processing with transformers" by lewis tunstall, leandro von werra and thomas wolf, it explains how everything works in a practical way. i know there's also a lot of material online on this topic, blogs and tutorials, but i think the papers and books give you more depth in understanding and more precise technical details.

one final thing, it has been said that a computer science degree is only a good predictor of how well you can understand the documentation of a new library. and i must say this is somewhat true, although in my case i prefer staring at graphs for hours.

hope this helps. let me know if there's anything else.
