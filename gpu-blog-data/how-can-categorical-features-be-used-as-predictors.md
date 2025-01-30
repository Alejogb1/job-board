---
title: "How can categorical features be used as predictors in Bayesian hierarchical models?"
date: "2025-01-30"
id: "how-can-categorical-features-be-used-as-predictors"
---
Categorical features, often representing groupings or types, are not directly amenable to the mathematical operations of many regression models, including those within a Bayesian hierarchical framework. However, these features can be incorporated by transforming them into numerical representations, specifically using indicator variables, and then structuring the model to account for the inherent hierarchical relationships these categories often possess.

Categorical features, by their nature, represent a discrete set of possibilities. For instance, a feature "Region" might include categories like "North," "South," "East," and "West." To use this feature in a statistical model, we must convert these textual categories into a numerical form. This is typically done through one-hot encoding, creating a set of binary indicator variables. Each category becomes a new column in the data; a '1' indicates membership in that category and '0' indicates non-membership. This allows the model to learn distinct effects associated with each category.

In the context of Bayesian hierarchical modeling, the crucial aspect is that these categories are not treated as independent, isolated units. Instead, we often assume that the categories are part of a broader structure or group. For example, regions in a dataset could be seen as part of a larger country or perhaps influenced by a shared geography or climate, leading to similarities in the parameter estimates. This shared structure is formalized in the model through the use of hierarchical parameters. Instead of estimating one coefficient for each region, we estimate parameters within a hierarchy, leveraging shared information to stabilize estimates, particularly for categories with limited data.

The simplest hierarchical model would have parameters that vary by each group, while being tied to a common distribution. The intercept for a regression, for instance, would now become a random effect based on the categorical group. The effect for a specific category is then a draw from this distribution, centered around a grand mean and with a variance to be inferred from the data. This is known as a "varying intercept" model. More complex models would allow other regression coefficients to vary by groups as well ("varying slope models").

Consider a dataset where we aim to predict website conversion rates based on the device used (e.g., Desktop, Mobile, Tablet) and various other numeric features, while leveraging a Bayesian approach.

```python
import pymc3 as pm
import numpy as np
import pandas as pd

# Example data
data = {
    'device': ['Desktop', 'Mobile', 'Tablet', 'Desktop', 'Mobile', 'Tablet'] * 100,
    'feature1': np.random.normal(0, 1, 600),
    'conversion': np.random.binomial(1, 0.4, 600)
}
df = pd.DataFrame(data)

# One-hot encode the device categories
df = pd.get_dummies(df, columns=['device'], drop_first=False)

# Bayesian hierarchical model
with pm.Model() as model:
    # Hyperpriors for the overall intercept
    mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)

    # Group-level intercepts for each device category
    alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=df[['device_Desktop', 'device_Mobile', 'device_Tablet']].shape[1])

    # Prior for the coefficient of feature1
    beta = pm.Normal("beta", mu=0, sigma=1)

    # Linear predictor
    mu = alpha[0] * df['device_Desktop'].values + alpha[1] * df['device_Mobile'].values + alpha[2] * df['device_Tablet'].values + beta * df['feature1'].values

    # Bernoulli likelihood for binary conversions
    conversion_obs = pm.Bernoulli("conversion_obs", p=pm.math.sigmoid(mu), observed=df['conversion'].values)

    trace = pm.sample(1000, tune=1000, chains=2)
```

In this first example, we create a PyMC3 model using generated data. We one-hot encode our categorical 'device' feature into `device_Desktop`, `device_Mobile`, and `device_Tablet`, with each being a separate indicator variable. The key is the specification of the `alpha` parameter: it has a normal distribution with a prior mean (`mu_alpha`) and standard deviation (`sigma_alpha`). Critically, `alpha` is defined with a shape corresponding to the number of categories, providing a separate intercept for each. The posterior samples for `alpha` are used to infer group-specific effects.

Now let's consider a second example where our categories are not mutually exclusive, such as 'interest tags' on a user. A user can have more than one interest, and we want to model the impact of these tags on engagement scores. This would also require hierarchical modeling, but the implementation is slightly different, as each tag now constitutes a predictor feature by itself, and we might want to regularize their impact by having a common prior distribution.

```python
import pymc3 as pm
import numpy as np
import pandas as pd

# Example data
data = {
    'user_id': np.arange(100),
    'engagement': np.random.normal(5, 2, 100)
}

df = pd.DataFrame(data)
interests = ['Sports','Politics','Technology','Gaming', 'Cooking', 'Travel']
for interest in interests:
    df[interest] = np.random.binomial(1, 0.3, size=100)

# Bayesian hierarchical model
with pm.Model() as model:
    # Hyperpriors for the overall intercept
    mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)

    # Group-level intercepts for each device category
    alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=len(interests))

    # Model the mean
    mu =  mu_alpha + pm.math.dot(alpha, df[interests].values.T)

    # Likelihood for engagement
    sigma_engagement = pm.HalfNormal("sigma_engagement", sigma=1)
    engagement_obs = pm.Normal("engagement_obs", mu=mu, sigma=sigma_engagement, observed=df['engagement'].values)


    trace = pm.sample(1000, tune=1000, chains=2)

```

Here, each interest tag becomes a column of 0s and 1s. We model the engagement score as being influenced by a shared intercept, plus a linear combination of each interest's parameter `alpha` and the presence or absence of each tag. The key difference here is the treatment of categorical variables. We are not one-hot encoding a single categorical variable, but are treating multiple, binary-valued variables as multiple predictors in a single equation. This is still hierarchical modeling, given that each parameter has a shared distribution. The use of a shared prior over all tags allows us to regularize parameter estimates and to deal with situations where some tags are rare, or the number of columns exceeds the number of rows in the data.

Finally, a slightly different, but equally crucial use of categorical variables is when they are used to group individual effects. Consider the following code, where we have observations for students nested within classrooms. Our goal is to estimate student performance, while considering that some classes might simply be better than others.

```python
import pymc3 as pm
import numpy as np
import pandas as pd

# Example data
num_students = 300
num_classrooms = 20
df = pd.DataFrame()
df['classroom_id'] = np.random.choice(range(num_classrooms), size = num_students)
df['student_id'] = np.arange(num_students)
df['performance'] = np.random.normal(5, 2, num_students)

# Bayesian hierarchical model
with pm.Model() as model:
    # Hyperpriors for the overall intercept
    mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)

    # Group-level intercepts for each classroom
    alpha_classroom = pm.Normal("alpha_classroom", mu=mu_alpha, sigma=sigma_alpha, shape=num_classrooms)

    # Model for the performance
    mu =  alpha_classroom[df.classroom_id.values]
    # Likelihood for engagement
    sigma_performance = pm.HalfNormal("sigma_performance", sigma=1)
    performance_obs = pm.Normal("performance_obs", mu=mu, sigma=sigma_performance, observed=df['performance'].values)

    trace = pm.sample(1000, tune=1000, chains=2)
```

Here, we are not converting the classroom IDs to indicator variables in the same way. Instead, we directly index into the `alpha_classroom` random effect. The indexing of the random effects by classroom ID allows each classroom to have a specific intercept. This implementation is typical of hierarchical models with nested random effects, where individuals (students) are nested within groups (classrooms). The core idea remains the same: we structure our model so that parameters for categories come from a shared distribution, allowing for the regularization of the individual effect estimates.

For deeper understanding of Bayesian hierarchical modeling with categorical predictors, I recommend studying the following resources: "Bayesian Data Analysis" by Gelman et al., which provides a comprehensive theoretical foundation; “Statistical Rethinking” by Richard McElreath, which offers a practical and conceptual approach using R and Stan; and the PyMC3 documentation, providing detailed guidance on implementing these models in Python.
