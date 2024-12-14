---
title: "How to use a custom NRC-style lexicon on Syuzhet for R?"
date: "2024-12-14"
id: "how-to-use-a-custom-nrc-style-lexicon-on-syuzhet-for-r"
---

alright, so, you're looking to use a custom nrc-style lexicon with syuzhet in r? i've been down this road, and let me tell you, it's not always a walk in the park but it’s fairly straightforward once you get your head around the data structures.

let me break it down, and i'll give you a few code snippets that should help. i remember years ago, when i was first getting into text analysis, i was working on this project involving analyzing the sentiment in old forum posts, and i needed a lexicon that was more specific to internet slang rather than the broad nrc. i spent days trying to figure out how to shoehorn my lexicon into the existing tools - it was not fun at all, i mean for days, the kind of debugging nightmare that makes you question all your life decisions, so believe me, i get where you are coming from.

the default `syuzhet` package uses the nrc lexicon, which, as we both know, has eight emotional categories (anger, anticipation, disgust, fear, joy, sadness, surprise, and trust) plus positive and negative. if your custom lexicon is structured in the same format, the easiest path forward is to replace the default lexicon in the syuzhet environment directly. that avoids having to rewrite a huge amount of code and keeps things clean and simple.

first off, what does an nrc-style lexicon look like? it's typically a text file, or a data frame in r, where you've got one column for words and then columns for each of those categories. a '1' indicates the word is associated with the category, '0' otherwise. here's a dummy example. let's say you have your data in a file called `my_custom_lexicon.csv`. in the file, your csv might look something like this:

```
word,anger,anticipation,disgust,fear,joy,sadness,surprise,trust,positive,negative
happy,0,1,0,0,1,0,0,1,1,0
sad,0,0,0,0,0,1,0,0,0,1
angry,1,0,1,1,0,0,0,0,0,1
excited,0,1,0,0,1,0,1,1,1,0
worried,0,0,0,1,0,1,1,0,0,1
calm,0,1,0,0,1,0,0,1,1,0
```

i know, it looks a bit tedious but once you make the first, it gets much easier.

here's the code to load this into r and replace the syuzhet lexicon. this is one approach, and it's good if you are looking for a quick and dirty solution without redoing the entire source code of `syuzhet`:

```r
# load necessary library
library(syuzhet)
library(readr)

# read in custom lexicon data
my_lexicon <- read_csv("my_custom_lexicon.csv")

# set the lexicon to syuzhet directly
environment(get_sentiment_dictionary)$nrc <- my_lexicon

# test it out on a dummy vector of sentences
text_vector <- c("i am feeling happy today",
                 "that is sad news indeed",
                 "i am angry about the delay",
                 "i am really excited for the trip",
                 "i feel a bit worried about it",
                 "i need to feel calm")

# analyse the sentiment
sentiment_analysis <- get_sentiment(text_vector, method="nrc")

# print the result
print(sentiment_analysis)

# check that your lexicon was actually used
str(environment(get_sentiment_dictionary)$nrc)

```

what this does is fairly simple: it reads your csv, then accesses the `get_sentiment_dictionary` function, which is used internally by syuzhet, and sets your data frame to be its nrc lexicon. the str() command is a quick way to sanity-check that your dataframe was loaded correctly.

it's worth noting, that this approach directly modifies syuzhet's internal data, so if you restart your session it will be back to default. you will need to run it every time after you load the package. also be aware that this could break if the internals of `syuzhet` change in the future.

another more robust approach is to build your own function, which makes use of syuzhet functions but on a custom lexicon data frame. i do not recommend this if your are not familiar with `syuzhet` package internals, as you are likely to miss some parameters or other details. but the benefit is that it is more resilient to future package changes and also you can further modify/extend the functionality. here's a way to do it:

```r
# load necessary libraries
library(syuzhet)
library(readr)
library(dplyr)

# read in your custom lexicon
my_lexicon <- read_csv("my_custom_lexicon.csv")

# build a custom function
get_custom_sentiment <- function(text_vector, lexicon_data) {
  # make sure that the text vector is a vector of characters and if not transform it.
  if(!is.character(text_vector)){
    text_vector <- as.character(text_vector)
  }

  # lowercase
  text_vector <- tolower(text_vector)

  # tokenize into words, you can add more sophisticated steps like lemmatization here if you want.
  words <- strsplit(text_vector, "\\s+")

  # create empty dataframe to store the results
  results <- data.frame(matrix(0, nrow=length(text_vector), ncol=ncol(lexicon_data)-1))
  names(results) <- names(lexicon_data)[2:ncol(lexicon_data)]
    
    # loop through sentences
   for(i in 1:length(words)){
      # loop through words in each sentence
      for(word in words[[i]]){
         
      # find which row has the given word, if any
        lex_row <- which(lexicon_data$word == word)
    
        # if found, add the values in each column.
        if(length(lex_row) > 0){
          results[i, ] <- results[i, ] + lexicon_data[lex_row, 2:ncol(lexicon_data)]
        }
      }
  }
  return(results)
}

# test custom function
sentiment_analysis_custom <- get_custom_sentiment(text_vector, my_lexicon)
print(sentiment_analysis_custom)

```

this is doing the same thing as syuzhet's internal function (more or less) and using your lexicon. this function now takes any text vector and your custom lexicon, tokenizes the text, and performs a lookup on the words and sums the sentiment values. this approach is more explicit and you have full control over the process.

note, i’m doing minimal preprocessing here, lowercase and tokenization. in any real application, you probably also want to use more advanced text processing like removing punctuation, numbers, or performing lemmatization, but i'm keeping this example clean. you will find functions to do that in the package `tm` or `quanteda`, which might also be helpful for other text processing tasks.

another point worth mentioning, for any lexicon approach, that you need to be careful on the scope of your words, for example some times words such as “not bad” might convey a positive sentiment, but a nrc-style lexicon might classify the word “bad” as a negative one and your approach might produce unexpected results. in cases where you might have a more specific or different context you might also need to use different kinds of analysis, such as topic modeling techniques or even more sophisticated machine learning algorithms. i've found great resources about this in the *speech and language processing* book by jurafsky and martin, in case you want to dive deeper into those concepts.

final piece of code to create a sentiment score with a sum of positive and negative words, this is the kind of sentiment score that a lot of libraries use, and it is an easy one to implement with your custom lexicon, so in case you need it:

```r
# first run the other codes
# this uses the custom function previously created.
# get sentiment analysis
sentiment_analysis_custom <- get_custom_sentiment(text_vector, my_lexicon)
# compute a custom sentiment score
my_sentiment_score <- sentiment_analysis_custom %>%
    mutate(sentiment = positive - negative) %>%
    select(sentiment)
print(my_sentiment_score)
```

this uses the `dplyr` package to get a new column with the custom sentiment score. this score is just positive values minus the negative values in the lexicon.

one thing i've learned over time is that there is never a silver bullet for text analysis. you need to try different approaches and evaluate their performance on your specific data, for example the sentiment score of positive minus negative might be enough in some cases, but not in other cases, you always need to test and adjust the parameters for your specific task. i once saw a senior dev spending a week trying to debug a single line of code - turned out it was a simple typo in a variable name! so always double check.

as for further resources, besides jurafsky and martin mentioned before, i really recommend reading the paper by mohammad et al. (2013) *from once upon a time to happily ever after: tracking emotions in stories* it was super enlightening to understand the emotional aspect of the data. also, papers on sentiment analysis with lexicons usually also explain some of the underlying assumptions and limitations, and they are a very good resource to understand them.

i hope this helped and saves you some of the headache i went through. let me know if you have other questions.
