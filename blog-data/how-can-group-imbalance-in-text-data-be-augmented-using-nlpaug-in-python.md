---
title: "How can group imbalance in text data be augmented using NLPAUG in Python?"
date: "2024-12-23"
id: "how-can-group-imbalance-in-text-data-be-augmented-using-nlpaug-in-python"
---

Alright, let's talk about tackling class imbalance in text data using `nlpaug`. It's a challenge I've faced more times than I care to remember, particularly back when I was working on that sentiment analysis project for e-commerce reviews. We had a deluge of positive feedback, with only a small fraction of negative or neutral opinions, which skewed the models considerably. Augmentation became essential.

The core idea behind using `nlpaug` for this purpose is to generate synthetic data points for your underrepresented classes. Instead of simply oversampling by duplicating existing examples (which can lead to overfitting), `nlpaug` employs various methods that slightly modify the original text, creating similar, yet novel, samples. This helps diversify the training data and push the model towards better generalization. We're not just adding noise; we're adding meaningful variations that reflect the semantic context.

`nlpaug` provides a wide range of augmentation techniques, but for text, we’re primarily interested in:

1.  **Word-level augmentation:** These methods modify the text by altering, deleting, or swapping words. This includes things like synonym replacement, insertion, deletion, and swapping based on a pre-trained word embedding model.
2.  **Character-level augmentation:** Here, the text is augmented by adding, replacing, or deleting individual characters. This is useful for simulating typos or other small variations.
3. **Back Translation:** One of the stronger augmentation methods. Here a sentence is translated to another language and then back to the original. This introduces semantic variation while maintaining meaning, and often leads to the most impactful augmentations.

Now, how do we actually implement this? I'll walk through three examples focusing on word-level augmentation, back translation, and then how to use them in a pipeline for a practical scenario.

**Example 1: Word-level Augmentation with Synonym Replacement**

Let's say we want to augment a small set of negative reviews. Synonym replacement is a good starting point. Here’s a snippet:

```python
import nlpaug.augmenter.word as naw
import pandas as pd

# Sample negative reviews
negative_reviews = [
    "This product is awful and terrible.",
    "I absolutely hated this item.",
    "The quality is extremely poor.",
    "It was a complete waste of money."
]

# Load the augmenter
aug = naw.SynonymAug(aug_src='wordnet')

# Augment each negative review
augmented_reviews = []
for review in negative_reviews:
  augmented_text = aug.augment(review)
  augmented_reviews.append(augmented_text)

df = pd.DataFrame({
    'original': negative_reviews,
    'augmented' : augmented_reviews
})
print(df)
```

This script uses `SynonymAug` from `nlpaug` with the `wordnet` source. It iterates through our sample negative reviews, generating augmented versions by replacing some of the words with their synonyms. The result is new, yet similar, sentences. In my past projects, I often experimented with the `aug_min`, `aug_max` and `aug_p` parameters to control the number of words altered in an augmentation. This way, you can fine-tune the degree of change for each sample and observe how much it impacts the models.

**Example 2: Augmentation with Back Translation**

Back translation can create significantly more varied augmentations while preserving the intent. Let's take the same reviews and augment them using this method. I used to use Google Translate API, but now it's a bit more common to utilize pre-trained models available from the `transformers` library.

```python
import nlpaug.augmenter.word as naw
import pandas as pd

# Sample negative reviews
negative_reviews = [
    "This product is awful and terrible.",
    "I absolutely hated this item.",
    "The quality is extremely poor.",
    "It was a complete waste of money."
]


# Load the back translation augmenter
aug = naw.BackTranslationAug(from_model_name='facebook/mbart-large-50-one-to-many-mmt', to_model_name='facebook/mbart-large-50-one-to-many-mmt')

# Augment each negative review
augmented_reviews = []
for review in negative_reviews:
    augmented_text = aug.augment(review)
    augmented_reviews.append(augmented_text)

df = pd.DataFrame({
    'original': negative_reviews,
    'augmented' : augmented_reviews
})
print(df)
```

This snippet employs `BackTranslationAug` and uses a pre-trained `mbart` model from Facebook, which is suitable for translation tasks. Note that these large models are computationally demanding. Back translation will rephrase the sentences in unexpected ways which can be quite helpful to expose the models to new examples of negative sentiment. In practical scenarios, I've noticed that carefully selecting intermediate languages can impact the type of augmentations created.

**Example 3: Augmentation Pipeline for Imbalanced Dataset**

Now, let’s see how to combine these augmentations in a pipeline for a more realistic scenario. In the e-commerce example, I faced a scenario where there was a large class of positive reviews, followed by a small class of negative reviews and an even smaller class of neutral reviews.

```python
import nlpaug.augmenter.word as naw
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample dataset (replace with your actual data)
data = pd.DataFrame({
    'text': [
        "This is a fantastic product!",
        "I love it so much.",
        "It works perfectly as expected.",
        "Okay, it's fine I guess.",
        "It could be better.",
        "I'm not very impressed.",
        "It was a complete waste of money.",
        "Awful and terrible product",
        "It broke on first use"
    ],
    'label': [
        "positive",
        "positive",
        "positive",
        "neutral",
        "neutral",
        "negative",
        "negative",
        "negative",
        "negative"
    ]
})

# Split the data into training and test sets
train_df, test_df = train_test_split(data, test_size = 0.2, random_state=42)
# Get counts for each label in training set
train_counts = train_df['label'].value_counts()

# Determine the target augmentation numbers
max_count = train_counts.max()

augmentations_needed = {}
for label, count in train_counts.items():
    if count < max_count:
        augmentations_needed[label] = max_count - count

# Define the augmenters
aug_syn = naw.SynonymAug(aug_src='wordnet')
aug_backtrans = naw.BackTranslationAug(from_model_name='facebook/mbart-large-50-one-to-many-mmt', to_model_name='facebook/mbart-large-50-one-to-many-mmt')

# Apply augmentations
augmented_data = []
for label, num_augmentations in augmentations_needed.items():
    for index, row in train_df.iterrows():
        if row['label'] == label:
            for _ in range(num_augmentations):
              if label == 'negative': # Back translation is useful for more impactful changes in negative sentiment
                augmented_text = aug_backtrans.augment(row['text'])
              else:
                augmented_text = aug_syn.augment(row['text'])

              augmented_data.append({'text':augmented_text, 'label':label})
augmented_df = pd.concat([train_df, pd.DataFrame(augmented_data)], ignore_index=True)


# Display the augmented dataframe
print(augmented_df)

```
Here, the code splits the original data into train and test sets, then calculates how many augmentations are required per class. Then the script loops through each class and for each example in the underrepresented class it performs an augmentation until the class size is the same as the majority class. It utilizes a combination of word synonym replacements for neutral examples and back translation for the negative sentiment examples, which ensures more meaningful augmentations. I used to have a similar setup, carefully choosing the method based on the specific characteristics of each class and the desired level of variation. This ensures no overfitting.

**Resource Recommendations**

For a deeper dive into the theoretical underpinnings of data augmentation, I'd highly recommend consulting "Data Augmentation in Machine Learning Using Generative Adversarial Networks" by Wang et al. (2018). This paper offers a comprehensive overview of augmentation methods, although it focuses more on image data. Additionally, for a more general understanding of text data pre-processing, the book "Speech and Language Processing" by Jurafsky and Martin is an invaluable resource. Regarding back translation, research papers exploring different model architectures and their efficacy, such as “Massively Multilingual Machine Translation” by Aharoni et al. (2019), can provide insights into the technical subtleties of back translation.

In conclusion, `nlpaug` offers a powerful toolset for addressing class imbalance in text data. By strategically employing techniques like synonym replacement and back translation, and carefully choosing the settings, we can generate diverse, meaningful augmentations that improve the generalization capabilities of our models. Remember, it’s not about blindly increasing the number of samples; it’s about creating variations that force the model to learn robust representations of your underrepresented classes.
