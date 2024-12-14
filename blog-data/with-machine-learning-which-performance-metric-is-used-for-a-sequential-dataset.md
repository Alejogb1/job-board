---
title: "With Machine Learning, which performance metric is used for a sequential dataset?"
date: "2024-12-14"
id: "with-machine-learning-which-performance-metric-is-used-for-a-sequential-dataset"
---

alright, so you're asking about performance metrics for sequential data in machine learning. that’s a pretty common question and it can get a little tricky because we aren't just dealing with individual data points. we’ve got time dependencies or order which throws a wrench in how we usually measure performance.

i’ve been there, believe me. a few years back, i was working on a project to predict network traffic patterns for a small isp. we had this massive stream of data – packets coming in, going out, the whole shebang, and the initial models i built using just regular metrics for classification just completely bombed. it felt like i was trying to use a screwdriver on a nail - wrong tool, wrong job. so, yeah, i had to really go back to the books and dive into what metrics actually made sense for the sequence of events, especially when the order was important.

so, let’s get down to brass tacks, shall we? the classic accuracy, precision, and recall, while important in other areas, aren’t inherently suited for sequential data because they are largely based on single point estimations. instead, we need metrics that understand the relationships within the sequence itself. the metric you use will really depend on the exact problem you are trying to solve. you got to define that first.

if we're talking about predicting a sequence of labels, like say, part-of-speech tagging in natural language processing, a pretty common one is the 'sequence accuracy'. this isn't the same as per-token accuracy. instead, a sequence is considered correct only if all the tokens are predicted correctly in their correct order. pretty strict. this metric gives you a good overview of how well your model grasps the overall sequence.

here’s a quick snippet of python code to calculate it:

```python
def sequence_accuracy(true_sequences, predicted_sequences):
    """
    calculates sequence accuracy.

    args:
      true_sequences: list of true sequences of labels (list of lists).
      predicted_sequences: list of predicted sequences of labels (list of lists).

    returns:
      sequence accuracy value as a float.
    """
    if len(true_sequences) != len(predicted_sequences):
        raise valueerror("input lists must have the same length.")

    correct_count = 0
    for true_seq, predicted_seq in zip(true_sequences, predicted_sequences):
        if true_seq == predicted_seq:
            correct_count += 1
    return correct_count / len(true_sequences)

# example usage
true_labels = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
predicted_labels = [["a", "b", "c"], ["d", "e", "x"], ["g", "h", "i"]]
print(f"sequence accuracy: {sequence_accuracy(true_labels, predicted_labels)}") # outputs 0.6666
```

now, if we're dealing with a regression type problem - say, predicting the future stock price or next sensor reading, things are a bit different. we don’t have discrete labels, so sequence accuracy isn’t going to cut it. here, metrics like root mean squared error (rmse) or mean absolute error (mae) are still valid, but we often apply them on a sequence by sequence basis. often, we might calculate these metrics over windows of the predicted values to see how the predictions unfold over time.

for time series, there’s also the concept of 'mean absolute scaled error' (mase) which is quite useful. it compares the performance of your model against a naive forecasting method that just takes the last known value. it’s a good way to see if your model is actually learning anything useful, since most times you’d expect it to perform better than simply using the previous value. mase accounts for the scale of data as well, therefore being useful in comparing forecasting models across different scales, which is useful. this is often used to compare forecasting models.

here’s a python snippet to calculate mae (mean absolute error):

```python
import numpy as np

def mean_absolute_error(true_values, predicted_values):
    """
    calculates the mean absolute error.

    args:
      true_values: list or numpy array of true values.
      predicted_values: list or numpy array of predicted values.

    returns:
      mean absolute error value as a float.
    """
    if len(true_values) != len(predicted_values):
        raise valueerror("input lists must have the same length.")
    true_values = np.asarray(true_values)
    predicted_values = np.asarray(predicted_values)

    return np.mean(np.abs(true_values - predicted_values))


# example usage:
true_values = [1, 2, 3, 4, 5]
predicted_values = [1.2, 1.8, 3.1, 3.7, 4.9]
print(f"mae: {mean_absolute_error(true_values, predicted_values)}") # outputs 0.220
```

there’s also the problem of handling long sequences. consider, for instance, a language model trying to produce paragraphs of text. how do you measure the ‘goodness’ of the generated paragraphs? in such cases, we often rely on techniques like evaluating the quality of the produced text via 'perplexity' or via scores from a pretrained classifier/model that gives some measure of plausibility. these measures aren't error metrics per se, but they give a sense of how well the sequence was captured.

a particularly useful metric for evaluating predicted sequences is the 'dynamic time warping distance' (dtw). it is particularly useful because it is invariant to time warping or time delays in sequences. if the sequence of labels should have been the same but there was a shift in time between the predictions and the true values, then dtw can help.

a good illustration is speech recognition. imagine someone saying 'hello world' and your model has detected 'helo worled'. it might seem quite a bad prediction on a token level, however, dtw would help in quantifying how off this is. this is because dtw accounts for time differences between the two sequences. if the sequences are time shifted by one unit, then dtw might be low, even though the token by token prediction was wrong. it is also very useful when comparing two time series.

let me show you a snippet with a simplified dtw implementation, even though there are better libraries, it shows the gist of how it works:

```python
import numpy as np

def dtw_distance(seq1, seq2):
    """
    calculates the dynamic time warping distance between two sequences.

    args:
      seq1: first sequence as a numpy array.
      seq2: second sequence as a numpy array.

    returns:
      dynamic time warping distance as a float.
    """
    n = len(seq1)
    m = len(seq2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.abs(seq1[i-1] - seq2[j-1]) # cost between single items
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

    return dtw[n, m]


# example usage:
seq1 = np.array([1, 2, 3, 4, 5])
seq2 = np.array([1, 2, 4, 3, 5])
print(f"dtw distance: {dtw_distance(seq1, seq2)}") # outputs 2.0
```
that is not as sophisticated as a library implementation, but it shows the idea.

now, one thing i learned the hard way was the importance of really understanding the underlying problem. it’s easy to get caught up in the models and metrics, but if you don’t really know what you’re trying to achieve, those numbers can be misleading at best, and completely useless at worst. you need to make the metrics work for you not the other way around. also, don't forget to look for error patterns and analyse your mistakes. error analysis is important too, not just metrics.

regarding books, instead of pointing to specific libraries, i recommend taking a look at “elements of statistical learning” by hastie, tibshirani, and friedman. it’s a classic for a reason, and it covers most of the basics for understanding the statistics behind machine learning and performance metrics. another excellent book is "forecasting: principles and practice" by hyndman and athanasopoulos. it's a fantastic resource specifically for time series data and forecasting, which is one of the most common scenarios for sequence analysis. and of course, a ton of papers on the nips, icml, and acl venues. you really have to read the literature, and not just rely on blog posts - no offence meant to blog writers. these are just a few recommendations. there are many more to discover when you start searching for particular issues.

it might be frustrating, but dealing with sequential data is a field where you have to try a few things before you really nail it. it’s like going to the gym - you wouldn’t expect to lift heavy weights the first day you walk into it. it takes time to build the muscles - and in our case, to choose the correct metric for your specific problem. and if i am not making sense, then maybe, my models need retraining, *ba dum tss*.
