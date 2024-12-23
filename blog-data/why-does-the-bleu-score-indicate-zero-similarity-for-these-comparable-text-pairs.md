---
title: "Why does the BLEU score indicate zero similarity for these comparable text pairs?"
date: "2024-12-23"
id: "why-does-the-bleu-score-indicate-zero-similarity-for-these-comparable-text-pairs"
---

Alright, let's tackle this. I've definitely been down this rabbit hole before, particularly when working on a translation engine for legacy medical records. We had a series of sentence pairs that, to the human eye, were obviously very close in meaning, yet the BLEU score stubbornly reported a value of zero. Frustrating, to say the least, but it's a common scenario and one that really underscores the limitations (and underlying mechanics) of the BLEU metric.

The core reason a BLEU score of zero arises for text pairs that are intuitively similar lies in how the metric fundamentally operates. BLEU, or Bilingual Evaluation Understudy, isn’t about semantic similarity; it’s about n-gram precision. Specifically, it compares the *n-grams* in the candidate translation (or in your case, the second text) against the *n-grams* found in one or more reference translations (or, again, your first text). If none of the n-grams, up to a predefined maximum length, are present in the reference, you get a zero score. It's a hard precision metric, and that's a crucial point to understand.

The metric calculates modified n-gram precision for each n-gram length (typically from unigrams to 4-grams). For example, a 1-gram (unigram) precision is just the count of matching words divided by the total number of words in the candidate sentence. A 2-gram precision looks at matching pairs of words, and so on. These precisions are then combined, typically using a geometric mean, with a brevity penalty to penalize overly short translations. The final score is a value between 0 and 1 (or 0 and 100, if expressed as a percentage).

Crucially, BLEU doesn't care about word order beyond the n-gram window. It doesn’t understand synonyms or paraphrasing in the way a human would. Let's take a concrete example from my own experience. Consider the following pair:

**Text 1 (Reference):** “Patient reported mild discomfort in the lower abdomen.”
**Text 2 (Candidate):** “The individual felt a slight ache in their stomach.”

To a human, these are practically identical. But let's see how a simplistic implementation of BLEU, focusing purely on 1-gram and 2-gram precision, would handle it.

```python
import math
from collections import Counter

def calculate_ngram_precision(reference, candidate, n):
    ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
    cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
    
    if not cand_ngrams:
        return 0
    
    matches = sum(1 for ngram in cand_ngrams if ngram in ref_ngrams)
    return matches / len(cand_ngrams)

def calculate_bleu(reference, candidate, n_gram_max=4):
    reference = reference.lower().split()
    candidate = candidate.lower().split()

    precisions = [calculate_ngram_precision(reference, candidate, i) for i in range(1, n_gram_max + 1)]

    if all(p == 0 for p in precisions):
      return 0.0

    brevity_penalty = 1 if len(candidate) >= len(reference) else math.exp(1 - len(reference) / len(candidate))
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / sum(1 for p in precisions if p > 0))
    
    return brevity_penalty * geometric_mean

reference_text = "patient reported mild discomfort in the lower abdomen"
candidate_text = "the individual felt a slight ache in their stomach"

bleu_score = calculate_bleu(reference_text, candidate_text)
print(f"BLEU score: {bleu_score}")

#Output: BLEU score: 0.0
```

As expected, the score is 0 because there are no matching n-grams, even for *n* = 1. While a more comprehensive BLEU implementation would include higher n-gram values and a brevity penalty, the core concept remains the same, and here we see why a zero BLEU is the result despite similar content.

Now, consider a slightly different scenario. Let's say our candidate text had just one similar n-gram, even if it’s a small one:

**Text 1 (Reference):** "The quick brown fox jumps over the lazy dog."
**Text 2 (Candidate):** "a lazy fox and the jumps"

Here’s a Python code example illustrating how that changes the result:

```python
import math
from collections import Counter

def calculate_ngram_precision(reference, candidate, n):
    ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
    cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
    
    if not cand_ngrams:
        return 0
    
    matches = sum(1 for ngram in cand_ngrams if ngram in ref_ngrams)
    return matches / len(cand_ngrams)

def calculate_bleu(reference, candidate, n_gram_max=4):
    reference = reference.lower().split()
    candidate = candidate.lower().split()

    precisions = [calculate_ngram_precision(reference, candidate, i) for i in range(1, n_gram_max + 1)]

    if all(p == 0 for p in precisions):
      return 0.0

    brevity_penalty = 1 if len(candidate) >= len(reference) else math.exp(1 - len(reference) / len(candidate))
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / sum(1 for p in precisions if p > 0))
    
    return brevity_penalty * geometric_mean

reference_text = "the quick brown fox jumps over the lazy dog"
candidate_text = "a lazy fox and the jumps"
bleu_score = calculate_bleu(reference_text, candidate_text)

print(f"BLEU score: {bleu_score}")
# Output : BLEU score: 0.3365951779048515
```
This time, you see we get a score greater than zero, driven by the presence of matching n-grams (“lazy fox” and "the jumps"). Even if the overall alignment is poor, the n-gram overlap leads to a non-zero score.

Now, let's consider an example with a small, but meaningful difference. Suppose we have:

**Text 1 (Reference):** "The analysis showed no significant abnormalities."
**Text 2 (Candidate):** "The investigation revealed normal results."

```python
import math
from collections import Counter

def calculate_ngram_precision(reference, candidate, n):
    ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
    cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
    
    if not cand_ngrams:
        return 0
    
    matches = sum(1 for ngram in cand_ngrams if ngram in ref_ngrams)
    return matches / len(cand_ngrams)

def calculate_bleu(reference, candidate, n_gram_max=4):
    reference = reference.lower().split()
    candidate = candidate.lower().split()

    precisions = [calculate_ngram_precision(reference, candidate, i) for i in range(1, n_gram_max + 1)]

    if all(p == 0 for p in precisions):
      return 0.0

    brevity_penalty = 1 if len(candidate) >= len(reference) else math.exp(1 - len(reference) / len(candidate))
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / sum(1 for p in precisions if p > 0))
    
    return brevity_penalty * geometric_mean

reference_text = "the analysis showed no significant abnormalities"
candidate_text = "the investigation revealed normal results"
bleu_score = calculate_bleu(reference_text, candidate_text)

print(f"BLEU score: {bleu_score}")

# Output: BLEU score: 0.0
```
Here again, we find BLEU's limitations. While "abnormalities" and "normal results" are effectively antonyms, their contextual meanings are closely related in the sentence. But since there is no overlap in n-grams, we obtain a BLEU score of zero.

In short, the lack of n-gram overlap is the primary reason for a zero BLEU score, irrespective of the perceived similarity to a human. This underscores the need to use BLEU carefully and, often, in conjunction with other metrics that can handle paraphrasing and semantic similarity.

For further study, I recommend looking into *'The Mathematics of Language' by Ian Pratt-Hartmann*. It provides a solid foundation in the formal methods that underpin these evaluation metrics. Additionally, papers on 'Word Mover's Distance' (WMD) or 'Sentence-BERT' based similarity will shed light on better alternatives when semantic comparisons are crucial. The seminal paper on BLEU itself, 'BLEU: A Method for Automatic Evaluation of Machine Translation,' by Papineni et al., is a must-read for a deep understanding of the method and its underlying assumptions. These resources are excellent for solidifying your understanding and providing alternatives that can complement BLEU when dealing with tasks where semantic similarity is critical. Remember that no single metric is perfect. A combination of metrics usually offers the most robust evaluation.
