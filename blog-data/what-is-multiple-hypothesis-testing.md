---
title: "What is multiple hypothesis testing?"
date: "2025-01-26"
id: "what-is-multiple-hypothesis-testing"
---

Multiple hypothesis testing arises because the chance of observing a statistically significant result increases when conducting many tests. Consequently, without proper correction, one might erroneously conclude a relationship exists when, in reality, it is merely a chance occurrence. This problem is particularly prevalent in fields like genomics, where datasets often necessitate hundreds or thousands of simultaneous statistical tests.

When evaluating a single hypothesis, the process is straightforward: compute a p-value, compare it to a chosen alpha (typically 0.05), and if the p-value is smaller than alpha, reject the null hypothesis. This indicates statistically significant evidence against the null. However, when multiple hypotheses are tested simultaneously, this framework becomes inadequate. The chance of observing at least one false positive increases dramatically as the number of tests increases. For instance, if we conduct 20 independent tests, each with a 0.05 significance level, there’s about a 64% chance (calculated as 1 - (1 - 0.05)^20 ) of falsely rejecting at least one null hypothesis. This problem of inflated false positives is what multiple hypothesis testing aims to mitigate.

To address this, several methods have been developed to adjust p-values, reducing the likelihood of false positives. These adjustment methods work by making it harder to reject a null hypothesis. The core principle involves controlling either the family-wise error rate (FWER) or the false discovery rate (FDR). FWER control aims to minimize the probability of making at least one Type I error (false positive) across all tests. FDR control aims to minimize the expected proportion of Type I errors among the rejected hypotheses. Different procedures prioritize one over the other, and their suitability depends on the specific research goals.

Let’s consider a few practical scenarios and how these corrections might be implemented in code. Using Python, specifically with the `statsmodels` library, we can demonstrate common corrections. This example simulates some test data and applies two frequently used methods: Bonferroni and Benjamini-Hochberg.

```python
import numpy as np
import statsmodels.sandbox.multitest as smm

# Generate random p-values (simulating results from 100 tests)
np.random.seed(42) #for reproducibility
pvals = np.random.uniform(0, 1, 100)

# Bonferroni correction
bonferroni_results = smm.multipletests(pvals, method='bonferroni')
bonferroni_corrected_pvals = bonferroni_results[1] # Adjusted p-values
bonferroni_significant = bonferroni_results[0] # Boolean array of significant results

# Benjamini-Hochberg (FDR) correction
bh_results = smm.multipletests(pvals, method='fdr_bh')
bh_corrected_pvals = bh_results[1] # Adjusted p-values
bh_significant = bh_results[0] # Boolean array of significant results

# Print first 5 original and adjusted p-values and significances for comparison
print("Original P-Values (First 5):", pvals[:5])
print("Bonferroni-Corrected P-Values (First 5):", bonferroni_corrected_pvals[:5])
print("Bonferroni Significant (First 5):", bonferroni_significant[:5])
print("Benjamini-Hochberg Corrected P-Values (First 5):", bh_corrected_pvals[:5])
print("Benjamini-Hochberg Significant (First 5):", bh_significant[:5])

```

This snippet generates 100 random p-values and then applies both Bonferroni and Benjamini-Hochberg corrections using `statsmodels`. Bonferroni is a very conservative approach, adjusting by multiplying the p-value by the number of tests. The Benjamini-Hochberg method, on the other hand, controls the FDR and is less conservative. As you can observe from the output, Bonferroni leads to larger adjustments.

The following example illustrates a custom function to perform the Holm-Bonferroni correction.

```python
import numpy as np
import pandas as pd

def holm_bonferroni(pvalues, alpha=0.05):
    """
    Performs the Holm-Bonferroni correction on a list of p-values.
    
    Args:
        pvalues (list or numpy array): A list or array of p-values.
        alpha (float, optional): The significance level. Defaults to 0.05.
    Returns:
        pandas dataframe : Returns pvalues and their adjusted values and if null should be rejected.
    """
    
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    ranked_pvalues = pvalues[order]
    
    n = len(pvalues)
    adjusted_pvalues = np.zeros_like(pvalues, dtype=float)
    
    for i in range(n):
        adjusted_pvalues[order[i]] = ranked_pvalues[i] * (n - i)
        
    adjusted_pvalues[adjusted_pvalues > 1] = 1

    reject = adjusted_pvalues <= alpha

    df = pd.DataFrame({'pvalues': pvalues,
                    'adjusted_pvalues': adjusted_pvalues,
                    'reject': reject})
    
    return df

# Example usage:
pvals = np.random.uniform(0, 1, 10)
df = holm_bonferroni(pvals)
print(df)
```

This function implements the Holm-Bonferroni method, which is a stepwise version of the Bonferroni correction. Instead of applying the same correction to every p-value, it adjusts based on the rank of the p-value. This makes it slightly more powerful than standard Bonferroni without losing FWER control.

Here’s a different example of how to deal with data that would require a multiple hypothesis correction. Suppose we have gene expression data where we are comparing expression between two groups across thousands of genes. We simulate that here and apply the Benjamini-Hochberg method.

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.sandbox.multitest as smm

# Simulate gene expression data
np.random.seed(42) #for reproducibility

num_genes = 5000
group_1_expr = np.random.normal(5, 1, size=num_genes)
group_2_expr = np.random.normal(5.5, 1, size=num_genes)
# Simulate a real difference
group_2_expr[0:1000] = np.random.normal(7, 1, size=1000)

# Perform t-tests for each gene
p_values = []
for i in range(num_genes):
    _, p = stats.ttest_ind(group_1_expr[i], group_2_expr[i])
    p_values.append(p)

# Benjamini-Hochberg correction
bh_results = smm.multipletests(p_values, method='fdr_bh')
bh_corrected_pvals = bh_results[1]
bh_significant = bh_results[0]

# Analyze results
df = pd.DataFrame({'p_values': p_values, 'corrected_p_values': bh_corrected_pvals, 'significant': bh_significant})
significant_genes = df[df['significant']==True]

print(f"Number of significant genes before correction: {sum(np.array(p_values)<0.05)}")
print(f"Number of significant genes after BH correction: {len(significant_genes)}")
print(f"Proportion of genes considered significant: {len(significant_genes)/len(p_values)}")

```

This final example simulates differential gene expression across two groups. A t-test is conducted on each gene, and then the Benjamini-Hochberg procedure is applied to correct for multiple testing. As you can see, without correction, you would declare thousands of genes to be significantly different, whereas, after correction, this drops to a more reasonable amount.

When deciding which method to use, one needs to carefully evaluate the trade-offs between Type I and Type II errors, and the research context. FWER controlling methods like Bonferroni, are extremely conservative and reduce statistical power. FDR controlling methods like Benjamini-Hochberg are less conservative and provide more power, at the cost of allowing for some false positives.

Here is a comparative table summarizing these common methods:

| Name              | Functionality                            | Performance                | Use Case Examples                                     | Trade-offs                                                      |
|-------------------|------------------------------------------|-----------------------------|------------------------------------------------------|-----------------------------------------------------------------|
| Bonferroni        | Controls FWER by dividing alpha by number of tests  | Least Power/ Most Conservative| Exploratory studies or situations with very low tolerance for false positives| Can be overly conservative, leading to high rate of Type II errors. |
| Holm-Bonferroni | Step-down FWER control | Slightly higher power than Bonferroni | Situations requiring FWER control, but with slightly higher statistical power  | Still conservative compared to FDR methods, although less than standard Bonferroni.|
| Benjamini-Hochberg | Controls FDR by ordering p-values | Good Balance of Power and Control | Genome-wide association studies or other high-throughput data analysis  | Allows a small proportion of false positives. May not be suitable where any false positives would be highly detrimental. |

Based on my experience, in situations where minimizing *any* false positive is absolutely critical, even at the cost of missing true positives (e.g., drug approvals), Bonferroni correction or Holm-Bonferroni is preferred, despite its lower power. However, in fields like genomics or proteomics, where researchers typically examine thousands of hypotheses, the Benjamini-Hochberg procedure is more appropriate. The FDR approach allows a controlled proportion of false discoveries, offering a better balance between statistical power and error control. Furthermore, in more exploratory analyses, FDR control offers better discovery potential. For those interested in learning more, I would recommend further reading on statistical inference and multiple comparison procedures in reputable textbooks on statistical analysis or peer-reviewed research papers.
