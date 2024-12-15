---
title: "how to obtain the quanteda package's fcm to a sub-matrix in R?"
date: "2024-12-15"
id: "how-to-obtain-the-quanteda-packages-fcm-to-a-sub-matrix-in-r"
---

alright, so you're trying to get a sub-matrix from a quanteda `fcm` object, i've been there, done that. it's not always super intuitive, especially when you first get into the package. i remember pulling my hair out over this a few years back. i was working on a project trying to analyze political discourse on twitter, and i had this massive term co-occurrence matrix. i needed to isolate the relationships between specific hashtags and mentions. what i thought would be a quick slice and dice operation turned into a full-blown troubleshooting session.

the `fcm` object in `quanteda` isn't just a standard matrix. it’s a specialized structure, which is great for what it’s designed for, but it means you can't always just use regular matrix indexing methods to extract exactly what you want. this makes sense, given how quanteda optimizes its workflow, but can be a bit annoying at first.

the key thing to understand is that the `fcm` object stores the features (words, tokens, etc) in its row and column names, not necessarily as directly accessible dimensions. to get a sub-matrix, you'll need to use those names to select specific rows and columns and the `[` indexing with `c()` to get the combination of rows and columns you need. in practice you get the intersection of the row and column words you selected. so, let’s dive into how you can do this.

the straightforward approach is to use the feature names. if you know the specific terms you want to keep, you can use them to index the `fcm` like this:

```r
library(quanteda)

# example text
text <- c("apple banana cherry", "banana date apple", "cherry fig banana")
corp <- corpus(text)
tokens <- tokens(corp)

# create the fcm
fcm_matrix <- fcm(tokens)

# select terms
selected_terms <- c("apple", "banana", "date")

# extract the submatrix
sub_fcm <- fcm_matrix[selected_terms, selected_terms]

# print results
print(sub_fcm)
```

in this first example i create a simple corpus from texts and then the fcm object. i then create a vector of terms i want in the submatrix. finally, i use the `[` and the selected terms vector to get the sub matrix. it’s critical that the selected terms exist in the original `fcm`. if any term does not exist, it will return an error.

now, imagine that we also need to calculate some metrics on that submatrix. usually, you want a matrix with co-occurrences and also something like the strength or the pmi.

here is a more practical case where we calculate the point wise mutual information.

```r
library(quanteda)
library(Matrix)
library(dplyr)

# example text
text <- c("apple banana cherry", "banana date apple", "cherry fig banana", "apple banana fig date", "banana cherry date fig apple", "fig date banana apple")
corp <- corpus(text)
tokens <- tokens(corp)

# create the fcm
fcm_matrix <- fcm(tokens)

# select terms for the submatrix
selected_terms <- c("apple", "banana", "date")

# extract submatrix
sub_fcm <- fcm_matrix[selected_terms, selected_terms]

# calculate pointwise mutual information
calculate_pmi <- function(mat) {
  # get the row sums
  row_sums <- Matrix::rowSums(mat)
  # get the column sums
  col_sums <- Matrix::colSums(mat)
  # total sum
  total_sum <- sum(mat)
  # create the pmi matrix
  pmi_matrix <- Matrix::Matrix(0, nrow = nrow(mat), ncol = ncol(mat))
  for (i in 1:nrow(mat)) {
    for (j in 1:ncol(mat)) {
      pmi_matrix[i, j] <- log2((mat[i, j] * total_sum) / (row_sums[i] * col_sums[j]))
    }
  }
  rownames(pmi_matrix) <- rownames(mat)
  colnames(pmi_matrix) <- colnames(mat)
  return(pmi_matrix)
}

# calculate and print
pmi_submatrix <- calculate_pmi(sub_fcm)
print(pmi_submatrix)
```

in this second example, the code creates a simple sub-matrix and then calculate the point wise mutual information of the cooccurrences. first, it follows the same approach as before by creating the corpus, tokens and fcm object. Then it selects the desired terms to obtain a submatrix. The core part here is the function `calculate_pmi` that receives the sub matrix as parameter and computes the pmi using standard formulas. finally, the code calls the function and prints the resulting matrix. by calculating these metrics, you start to get a sense of the relationships between words, in ways beyond just co-occurrence counts. this is useful when you want to analyze the contextual usage of the words. i found this to be very handy when i was analyzing online communities because some words co-occur because of the language structure but not because they are directly related.

sometimes you might not know all the terms beforehand. suppose you want to get a sub-matrix of all terms that are related to the top 10 most frequent terms. then you might not know all the terms you want to get before analysing the `fcm`. for that case the next code example could be useful.

```r
library(quanteda)

# example text
text <- c("apple banana cherry", "banana date apple", "cherry fig banana",
          "apple banana fig date", "banana cherry date fig apple",
          "fig date banana apple", "apple mango kiwi", "mango kiwi pear",
          "kiwi pear orange", "pear orange apple", "orange apple banana",
          "apple banana fig")
corp <- corpus(text)
tokens <- tokens(corp)

# create the fcm
fcm_matrix <- fcm(tokens)

# get top n terms
top_n <- 5
top_terms <- names(sort(colSums(fcm_matrix), decreasing = TRUE))[1:top_n]

# extract submatrix
sub_fcm <- fcm_matrix[top_terms, top_terms]

print(sub_fcm)
```

here we build upon our initial example but this time we are generating a different submatrix. we first get the top terms by summing the columns using `colSums`. i remember in my first projects the use of these functions was not clear and this was painful to figure out. i sorted and indexed this vector to get the names of the most frequent terms. finally, we obtain the submatrix based on these top frequent words. that's why it's important to learn your basic vector and matrix operations using r.

one potential problem i have faced is that these approaches will only work well if you are careful with how you process the text prior to creating the `fcm`. for example if you do not use stemming then apple and apples might not be considered the same word and will give different entries. this is not a fault of the approach but more of a preprocessing issue that needs to be taken into account when implementing these techniques.

one thing to keep in mind, performance can become an issue if you are working with very large `fcm` objects. in those situations, you might want to consider working with sparse matrices, which are natively supported in `quanteda`. these can drastically reduce memory usage and improve computation speeds. if you are working with larger data sets the matrix package from r is a must. this allows you to work with very large matrices without running into the memory limitations. i've had to debug many situations where the machine simply refused to proceed, because a matrix became too big for the available ram.

to further improve the performance when calculating the sub-matrix, always try to filter before the `fcm` operation. for example, if you know which terms you are going to use for your matrix in advance, then filter out all other tokens. this will reduce the matrix size from the start, which is always a good idea. this was a useful trick i learned from a forum user when i had a particular bad implementation in one of my early projects.

for further reading about this i strongly recommend reading “quantitative text analysis” by  stefan müller and kenneth benoit, it helped me a lot with the theoretical and practical approaches. also, the quanteda documentation is quite good so definitely have it at hand when working with this package. there's a lot more you can do with fcm objects. for example, you can use them in network analysis and other cool text mining techniques. just experiment.

one last thing before i go. why did the programmer quit his job? because he didn't get arrays.
