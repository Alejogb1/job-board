---
title: "How do AIs learn from multilingual data sources?"
date: '2024-11-14'
id: 'how-do-ais-learn-from-multilingual-data-sources'
---

Hey, cool topic!  Multilingual datasets are super important for building models that work across languages.  It's also key to track where the data comes from, right?  That's provenance. 

I've been playing around with this lately, and it's a real challenge to keep things organized.  I've been using this code snippet to help me keep track of the different languages and sources:

```python
import pandas as pd

def track_provenance(data, language, source):
  """
  Adds provenance information to a dataset.

  Args:
    data: The dataset to be annotated.
    language: The language of the data.
    source: The source of the data.

  Returns:
    A DataFrame with provenance information added.
  """

  data['language'] = language
  data['source'] = source
  return data

# Example usage
df = pd.DataFrame({'text': ['This is a sentence.', 'Another sentence.']})
df = track_provenance(df, 'English', 'Wikipedia')
print(df)
```

It's still early days, but I'm experimenting with a few different methods to keep the data organized and track its provenance.  I'm also looking into tools like "data provenance tracking" and "multilingual dataset management" to see what's out there.
