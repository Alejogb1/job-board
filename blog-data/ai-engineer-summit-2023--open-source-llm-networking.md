---
title: "AI Engineer Summit 2023:  Open-Source LLM Networking"
date: "2024-11-16"
id: "ai-engineer-summit-2023--open-source-llm-networking"
---

dude so this ai engineer summit 2023 kickoff thing right  it was wild  like a total nerdgasm but in the best way possible  the whole point was to basically hype up this super cool ai engineering conference  they were like  "hey look at all these awesome people and companies and tech  isn't it amazing" and they totally nailed it


first off the guy hosting  he was hilarious  super chill vibe  he started by name-dropping all these massive players like autogpt  which is insane  a few months ago it was just some open-source project  now it's the main sponsor  that's  next-level growth  then there's superbase  these social media wizards who somehow make databases fun  they even broke their no-sponsorship policy for this thing  talk about commitment  and microsoft  oh man  thirteen billion dollars into openai  they're practically throwing money at ai  it's like the wild west out there  


he also mentioned fixie  which i thought was pretty funny  he was joking about their booth sidekick derek  wondering if derek could tell what he was wearing and suggest a wardrobe update  pure gold  plus cloudflare  they just turned thirteen  and they're totally embracing the changes  typical teen angst but for a tech company  


one of the key moments was when he talked about the conference app  network  it's not just your average conference app  oh no  this thing uses llms  large language models  for ai-powered matching  they're basically using fancy ai to connect people with shared problems  you put what you're working on or what problem you need to solve in your profile and the algorithm finds the perfect people to talk to  pretty clever  right  


another big idea was open-sourcing the whole app and matching algorithm  that's massive  it's about creating a better experience for everyone  not just the attendees at this specific summit  this is open-source at its finest  


the resolution was pretty clear  this conference was a huge success  they  showcased the insane growth of the ai engineering field  and they gave a big ol' middle finger to  proprietary code  embracing open source and llms  they highlighted some  key players and hinted at some big future announcements  it was a hype-fest for sure



now let's dive into some code snippets  because you know  i wouldn't be me without a little bit of code


first  let's look at how a simple  llm-powered recommendation system might work  it's simplified but you get the idea


```python
import random

# sample user profiles  keep it simple for now
users = {
    "user1": {"problem": "building a robust recommendation system", "skills": ["python", "ml", "llms"]},
    "user2": {"problem": "deploying models to production", "skills": ["kubernetes", "aws", "docker"]},
    "user3": {"problem": "improving llm performance", "skills": ["nlp", "transformers", "fine-tuning"]}
}


def recommend_connections(user, users):
    best_match = None
    best_similarity = 0
    for other_user, data in users.items():
        if user != other_user:
            similarity = calculate_similarity(user, other_user, users)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = other_user
    return best_match



def calculate_similarity(user1, user2, users):
    # simple similarity based on problem keywords
    problem1 = users[user1]["problem"].lower()
    problem2 = users[user2]["problem"].lower()
    common_words = len(set(problem1.split()) & set(problem2.split()))
    return common_words


# example usage
recommended_user = recommend_connections("user1", users)
print(f"User 1 recommended connection: {recommended_user}")


```

this uses a simplified similarity metric based on keyword overlap but in a real system  you'd use more sophisticated techniques like embedding based similarity  this would use vector databases like pgvector which was mentioned  


next let's see a tiny bit of how to use pgvector  you'd need to install the pgvector extension in postgresql first  obviously



```sql
-- inserting embeddings for user profiles (replace with actual embeddings)
INSERT INTO users (id, name, embedding) VALUES
(1, 'Alice', '[0.1, 0.2, 0.3, 0.4]'),
(2, 'Bob', '[0.4, 0.3, 0.2, 0.1]');

-- searching for users similar to Alice
SELECT name FROM users ORDER BY embedding <-> '[0.1, 0.2, 0.3, 0.4]' LIMIT 1;
```

this shows a simple insert and a similarity search  in a real-world app you would manage user data properly and handle larger vector datasets effectively   


finally  let's look at a bit of frontend code maybe  using react to display user recommendations


```javascript
import React, { useState, useEffect } from 'react';

function UserRecommendations() {
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    // fetch recommendations from the backend  api call to your server
    fetch('/api/recommendations')
      .then(res => res.json())
      .then(data => setRecommendations(data));
  }, []);

  return (
    <div>
      <h2>Your Recommendations</h2>
      <ul>
        {recommendations.map(user => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
    </div>
  );
}

export default UserRecommendations;
```

this is super basic  in a real app you'd handle errors  loading states  and probably use a more sophisticated ui library  this would fetch recommendations from the backend api which would handle the database interactions using pgvector or a similar solution  


so yeah  that was the ai engineer summit  a rollercoaster of laughs  amazing tech and  a serious dose of open-source awesomeness  definitely a conference i won't forget  i hope you enjoyed my super casual  techy breakdown  let me know if you want to dive deeper into any of these topics  i'm always up for a good code chat
