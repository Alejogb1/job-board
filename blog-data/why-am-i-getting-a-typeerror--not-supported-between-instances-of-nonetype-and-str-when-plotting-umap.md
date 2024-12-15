---
title: "Why am I getting a TypeError: '<' not supported between instances of 'NoneType' and 'str' when plotting UMAP?"
date: "2024-12-15"
id: "why-am-i-getting-a-typeerror--not-supported-between-instances-of-nonetype-and-str-when-plotting-umap"
---

alright, so you're hitting that classic `typeerror: '<' not supported between instances of 'nonetype' and 'str'` when trying to plot UMAP, huh? yeah, that's a familiar pain. i've banged my head against that wall more times than i'd like to recall. this usually means some data you're feeding into UMAP's plotting function is not what it's expecting, specifically it’s encountering `none` values when it's expecting strings (or numbers that can be compared). let's unpack this.

typically, umap itself is working fine. the error is almost always in the way you're structuring your data, or how you're passing it into the plotting step. umap, under the hood, doesn’t directly plot anything, it's the plotting library you are using after umap has produced its embeddings that can throw up this error, like matplotlib or seaborn. most of these plotting functions expect categorical labels as strings or numerical values that can be used to differentiate or color the plotted points and they need to be in the same length as the produced umap embeddings. if you have a mixed list of, for instance, names and missing information, expressed as `none` values, it's a recipe for this particular `typeerror`.

let me tell you, i first encountered this back when i was working on a large-scale customer behavior dataset. it had tons of missing values across all the categorical variables, and i didn't do a proper pre-processing step, i thought i could just throw everything in and see what would happen (bad idea). i swear, the number of times i've seen that traceback... it's ingrained into my memory. i remember thinking, 'this library is broken' (spoiler: it wasn't). i’ve learned now that the problem is almost always at the user side, with some bad data handling.

so, the `<` operator it’s complaining about means that your plotting library is trying to compare the labels, maybe it's ordering them alphabetically or numerically, or it could be checking for duplicates. and, in python, you can't compare a `none` value with a string; hence, the error. it’s like trying to compare an apple with the concept of an empty space. doesn’t work, does it? it just freaks the computer out.

let’s check common places where this could happen.

**the most common culprits**

1.  **missing labels:** your categorical labels may have `none` or `null` values scattered throughout. imagine a list that looks like this: `['cat','dog','bird', none, 'fish', none, 'lizard']`. you might have assumed that missing labels would be ignored but the plotting function in the backend has expectations for clean data.

    the quick fix for this is to either filter out the samples without labels or to use a default value (like `"unknown"`) for these `none` or `null` instances. let’s assume that you have a pandas dataframe named `df` with the categorical column you want to use for plotting called `category_column`.

    here's a code snippet you might find helpful:

    ```python
    import pandas as pd
    import umap
    import matplotlib.pyplot as plt

    # assume df is your pandas dataframe and 'category_column' is your column with labels
    
    def preprocess_labels(df, category_column):
        
        # fill nan values
        df[category_column].fillna('unknown', inplace=True)
        # convert to string just to be sure, if you have mixed types
        df[category_column] = df[category_column].astype(str)
        return df
    
    # the main steps of the umap procedure
    reducer = umap.umap()
    embeddings = reducer.fit_transform(df.drop(category_column, axis = 1))
    
    # apply the preprocess function
    df = preprocess_labels(df, category_column)

    # now, your plotting should work fine without the type error
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=df[category_column].astype('category').cat.codes)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    ```

2.  **mismatched lengths:** this happens when the length of your label list does not match the number of points in your umap embedding. you might have performed some preprocessing on your data *before* umap, and somehow have lost a few points in the label list. the resulting embedding and label list don’t have the same size, and when the plotting function tries to match up the labels it will run into problems.

    this situation is trickier, but can be solved by either double checking your data pipeline, or performing all data preparation steps in a dataframe to avoid mismatches or by only plotting what is available.
    let me show you how this can be done using pandas:

    ```python
    import pandas as pd
    import umap
    import matplotlib.pyplot as plt
    import numpy as np

    def check_size_mismatches(embeddings, labels):

        if len(embeddings) != len(labels):
            min_size = min(len(embeddings), len(labels))
            print(f"warning: labels and embeddings are different sizes, truncating to size {min_size}")
            embeddings = embeddings[:min_size]
            labels = labels[:min_size]
        return embeddings, labels
    
    # let's assume that the labels and the embeddings are in a numpy format
    
    # the main steps of the umap procedure
    reducer = umap.umap()
    embeddings = reducer.fit_transform(df.drop(category_column, axis = 1).values) # transform the dataframe to numpy
    
    labels = df[category_column].values # transform the labels column to a numpy array
    
    # apply function
    embeddings, labels = check_size_mismatches(embeddings, labels)
    
    # now plot everything
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    ```

3.  **incorrect plotting arguments:** sometimes the plotting function expects labels to be passed in a certain way that can cause problems, it might expect an index based color mapping, or another way, like the code sample i just provided before. double check the documentation of your plotting library. for example, if you use the `c` argument in the matplotlib's `scatter` function without transforming it into numeric categories first, it can give you problems with this type of errors.

    ```python
    import pandas as pd
    import umap
    import matplotlib.pyplot as plt

    def fix_plotting_arguments(df, embeddings, category_column):
        categories = df[category_column].astype('category')
        numeric_categories = categories.cat.codes
        
        # now you are passing the numeric code for the category
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=numeric_categories)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    # the main steps of the umap procedure
    reducer = umap.umap()
    embeddings = reducer.fit_transform(df.drop(category_column, axis = 1))
    
    #apply function to fix the plotting arguments
    fix_plotting_arguments(df, embeddings, category_column)
    ```

**debugging tips**

*   **print statements:** add `print(type(your_label), your_label)` statements just before the plotting line. this can help you see exactly what kind of data it's getting, and where those `nonetype`s are lurking. i’ve done this so many times!
*   **check sizes:** always verify that `len(embeddings)` is equal to `len(labels)`. use `print` for this too!.
*   **isolate:** if your pipeline is complex, try isolating the umap step and the plotting step and feed it some simple dummy data, you might find the problem way faster.
*   **read the documentation:** yeah, i know it's boring, but the documentation of your specific library is your friend. plotting libraries have many parameters for you to discover.

**further resources**

i strongly recommend reading up on some general data handling resources:

*   **"data cleaning and transformation with pandas"** by michael f. gallagher, it provides a practical guide on data preparation techniques.

*   **"python data science handbook"** by jake vanderplas it covers the fundamentals of pandas and plotting libraries, like matplotlib, essential for data visualization and plotting.

*   **the official `pandas` documentation**, it’s a bit dry, but is the definitive resource for anything related to data manipulation in pandas.

avoiding this error is more about data cleaning and preparation than about umap itself. once you get the hang of pre-processing your data correctly, these problems will appear less frequently. also, consider that maybe you should not use any categorical information and maybe try other plotting options that use density visualization instead. this might be a workaround if you are in a rush.

in conclusion, this specific typeerror isn't a sign of a broken library, but an indicator of data needing a little extra attention. i hope this helps, let me know if you have more questions, we've all been there! and don't worry, nobody gets it perfect the first time. well, maybe except for that one guy that i saw on a conference who got a standing ovation... the guy had perfect data, how is that even possible? (just kidding) good luck!
