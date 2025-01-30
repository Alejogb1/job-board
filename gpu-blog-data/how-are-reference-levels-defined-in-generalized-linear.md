---
title: "How are reference levels defined in generalized linear models?"
date: "2025-01-30"
id: "how-are-reference-levels-defined-in-generalized-linear"
---
Reference levels, or baseline categories, in generalized linear models (GLMs) are crucial for interpreting the model coefficients.  They represent the category against which the effects of other categories within a categorical predictor variable are compared.  Incorrect specification can lead to misinterpretations of the model output and flawed conclusions. My experience working on large-scale epidemiological studies heavily emphasized the importance of meticulous reference level selection, particularly when dealing with multi-level categorical predictors and interaction effects.  This careful consideration often involved extensive sensitivity analyses to ensure the robustness of my findings.

The core concept is straightforward: for each categorical predictor, one level is designated as the reference.  The model coefficients for the remaining levels then represent the difference in the response variable's expected value (on the link scale) compared to this reference.  The choice of reference level is entirely arbitrary; however, its selection significantly impacts the interpretability of the results.  A poor choice can obfuscate the analysis and hinder meaningful interpretation.  Ideally, the reference level should be theoretically meaningful or represent a clinically relevant baseline.

For example, consider a model predicting hospital readmission (binary outcome: 0 = no readmission, 1 = readmission) based on patient demographics. If 'age' is included as a categorical predictor with levels (18-30, 31-45, 46-60, 61+), selecting '18-30' as the reference implies that the coefficients associated with the other age groups represent the difference in the log-odds of readmission (using a logistic regression model) compared to the 18-30 age group.  Conversely, selecting '61+' as the reference would lead to coefficients representing the difference in log-odds relative to the oldest age group.  The choice should be driven by the research question and the intended audience.

The following examples illustrate reference level definition in different statistical software packages, focusing on how it affects model output and interpretation.

**Example 1: R (using `glm()` and `relevel()`)**

```R
# Sample data
data <- data.frame(
  readmission = c(0, 1, 0, 1, 0, 0, 1, 1, 1, 0),
  age_group = factor(c("18-30", "31-45", "18-30", "46-60", "31-45", "61+", "46-60", "18-30", "61+", "31-45")),
  gender = factor(c("Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"))
)

# Model with "18-30" as the reference level for age_group
model1 <- glm(readmission ~ relevel(age_group, ref = "18-30") + gender, data = data, family = binomial)
summary(model1)

# Model with "61+" as the reference level for age_group
model2 <- glm(readmission ~ relevel(age_group, ref = "61+") + gender, data = data, family = binomial)
summary(model2)

```

This code demonstrates how `relevel()` in R modifies the reference level of the `age_group` factor.  `summary(model1)` and `summary(model2)` will show different coefficients for `age_group`, reflecting the change in the reference level.  The intercept in each model also changes, representing the log-odds of readmission for the respective reference group.  Interpreting the coefficients requires careful consideration of the chosen reference.

**Example 2: Python (using `statsmodels`)**

```python
import statsmodels.formula.api as smf
import pandas as pd

# Sample data (same as R example)
data = pd.DataFrame({
    'readmission': [0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
    'age_group': ['18-30', '31-45', '18-30', '46-60', '31-45', '61+', '46-60', '18-30', '61+', '31-45'],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
})
data['age_group'] = pd.Categorical(data['age_group'])
data['gender'] = pd.Categorical(data['gender'])

# Model with "18-30" implicitly as the reference (alphabetical order)
model1 = smf.glm('readmission ~ age_group + gender', data=data, family=smf.families.Binomial()).fit()
print(model1.summary())

#To explicitly set a different reference, the data needs to be pre-processed before feeding into the model.
data_new = data.copy()
data_new['age_group'] = pd.Categorical(data_new['age_group'], categories=['61+','46-60','31-45','18-30'],ordered=False)

model2 = smf.glm('readmission ~ age_group + gender', data=data_new, family=smf.families.Binomial()).fit()
print(model2.summary())
```

In `statsmodels`, the reference level is often determined by alphabetical order unless explicitly specified through reordering of categories within the pandas `Categorical` object (as demonstrated in `model2`).  This example highlights the importance of checking the factor level order to understand which level is serving as the reference.  Again, the interpretation of the coefficients hinges on the chosen reference category.


**Example 3: SAS (using `PROC LOGISTIC`)**

```sas
/* Sample data (same as R and Python examples) */
data mydata;
  input readmission age_group $ gender $;
  datalines;
0 18-30 Male
1 31-45 Female
0 18-30 Male
1 46-60 Female
0 31-45 Male
0 61+ Female
1 46-60 Male
1 18-30 Female
1 61+ Male
0 31-45 Female
;
run;

/* Model with "18-30" as reference (default alphabetical order) */
proc logistic data=mydata;
  model readmission = age_group gender;
run;

/* Model with "61+" as reference (requires CLASS statement reordering)*/
proc logistic data=mydata;
  class age_group(ref="61+") gender;
  model readmission = age_group gender;
run;
```

SAS's `PROC LOGISTIC` usually uses alphabetical ordering for categorical variables unless specified differently using the `CLASS` statement's `REF` option, as shown above.  Modifying the reference level within the `CLASS` statement directly controls the baseline category for comparison. The output will again reflect the changes in the interpretation of the coefficients.

In conclusion, defining reference levels in GLMs is a critical aspect of model specification and interpretation.  The choice should be guided by theoretical considerations and the research question.  All the demonstrated statistical software packages provide mechanisms for specifying the reference level, though the specifics vary.  Consistent attention to this detail ensures accurate and meaningful interpretations of the model results.  I strongly recommend consulting statistical textbooks and manuals specific to the software you're using to gain a deeper understanding of this crucial element of GLM modeling.  Furthermore, conducting sensitivity analyses with different reference levels can provide valuable insight into the robustness of your findings.
