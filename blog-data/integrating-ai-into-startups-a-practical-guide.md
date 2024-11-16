---
title: "Integrating AI into Startups: A Practical Guide"
date: "2024-11-16"
id: "integrating-ai-into-startups-a-practical-guide"
---

yo dude so i just watched this panel discussion it was wild  like seriously these two dudes chris and brian are building ai stuff into their startups and it was all super insightful and kinda funny too  the whole point was figuring out how early-stage companies decide to jump into the ai game when they already have a million other things on their plates right  it was like a crash course in ai integration for startups basically

first off britney the host was cool she's a vc at crv  she set the stage perfectly  she was like "yo these guys are killing it but how did they decide to add ai to their already awesome products" and that’s exactly what we got to see  it was all very chill and down to earth too not stuffy at all  i dig that


one of the coolest moments was when chris talked about prefect  prefect's a workflow orchestration tool   basically it helps data scientists and engineers manage their complex data pipelines  chris said something like "ai in production is basically just a fancy expensive remote api"  that line cracked me up  but it totally made sense he was saying that ai models are basically just data apis that need managing just like any other part of a data pipeline  think of it like this:


```python
# a simplified representation of an AI model as a remote API
import requests

def get_ai_prediction(input_data):
  url = "https://my-awesome-ai-model.com/predict"
  headers = {"Content-Type": "application/json"}
  response = requests.post(url, headers=headers, json=input_data)
  if response.status_code == 200:
    return response.json()["prediction"]
  else:
    raise Exception(f"AI model request failed: {response.status_code}")

# example usage
user_input = {"text": "this is my input"}
prediction = get_ai_prediction(user_input)
print(f"AI prediction: {prediction}")

#handling errors and retries would be part of a robust workflow
#logging, monitoring and other best practices would also need to be included
```

see what i mean  it's just another api call but super expensive and potentially unreliable that's why workflow orchestration is key to make it all work smoothly  they even have this open-source project called marvin that's all about experimenting with llms large language models  it's like their ai playground


then brian from hex chimed in hex is a data science notebook platform  dude seriously his point about data scientists hating matplotlib syntax was hilarious but so true  he was like "the best thing is to let people focus on the cool creative stuff not on remembering which library to import"  he said they built ai features into hex to make things "magical" which is a pretty cool vision  they wanted to remove all the friction  and one example was a super cool feature that breaks down long complex sql queries into smaller more manageable chunks   like imagine this:


```sql
-- old, long query
WITH cte1 AS (SELECT * FROM table1 WHERE condition1),
     cte2 AS (SELECT * FROM table2 WHERE condition2),
     cte3 AS (SELECT * FROM cte1 JOIN cte2 ON condition3)
SELECT * FROM cte3;

-- hex's exploded query
-- cell 1
SELECT * FROM table1 WHERE condition1;

-- cell 2
SELECT * FROM table2 WHERE condition2;

-- cell 3
SELECT * FROM cell1 JOIN cell2 ON condition3;
```

pretty neat right  breaking down a monstrous query into these smaller cells is a game changer for productivity


another key moment was when they talked about building versus buying  brian was like "we built our own eval system because it had to be super tight with our platform"  eval is super important  it's how you measure how good your ai model actually is   but they decided to use an existing vector database  they didn't want to build that from scratch that's smart  they chose lancedb and created custom vector retrieval that perfectly fit their needs  smart  it's a classic tradeoff:  building is usually more control but buying is faster  


```python
#Illustrative example of evaluating a simple classification model.  
#Real world evaluation is far more complex and multifaceted.

from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1]  # Ground truth labels
y_pred = [0, 1, 0, 0, 1]  # Predicted labels

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

#More advanced evaluation would include metrics like precision, recall, F1-score, AUC-ROC, etc.
#Ideally, evaluation should be performed using a held-out test set to avoid overfitting.
```

that code snippet just shows the basics of model evaluation  they also brought up the whole build vs buy decision which is like super relevant  it's always a tough one  build gives you tight integration but buying is faster and might be better for things like vector databases that are already well-solved


the final big takeaway was how they approached measuring success  they stressed using solid evals early and often  they basically said "measure everything"  they used existing metrics like accuracy but also focused on quantifiable user experience improvements like how fast users could get to the results they wanted  and  they used data science to analyze their own ai tools which i thought was pretty meta  brian even mentioned "drinking your own champagne" which is a hilarious alternative to "dogfooding"  it's all about using your own product and getting real-world feedback


so yeah that was the panel  it was about how to successfully integrate ai into startups  the key ideas were  managing ai as just another part of your data pipeline  building ai to improve user experience  the tradeoffs of build vs buy  and then finally, setting up solid evaluation metrics from day one  plus lots of humor and some really solid points about the importance of testing and iterative development  pretty cool stuff  i'm already brainstorming  how i could use these ideas on my own projects!
