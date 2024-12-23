---
title: "¿Qué impacto tiene la colaboración interdisciplinaria en el desarrollo de plataformas inclusivas como Zelmira?"
date: "2024-12-12"
id: "qu-impacto-tiene-la-colaboracin-interdisciplinaria-en-el-desarrollo-de-plataformas-inclusivas-como-zelmira"
---

 so let's talk about how different types of expertise working together really shapes building inclusive platforms like zelmira it's not just about one discipline doing its thing in isolation no way that's how you get platforms that miss the mark badly i mean think about it if it's all engineers it'll be technically sound maybe but probably clunky and not really considering the user experience the needs of people with disabilities or the actual goals of inclusivity that kind of tunnel vision leads to a lot of usability problems and it's frankly unfair

it's like trying to build a house with only a hammer you can maybe get a basic structure but it's not gonna be livable or functional and it's definitely not going to be nice to look at or pleasant to be in so the first big piece of the puzzle is the design aspect user experience designers they're the ones who can translate the needs of people with various disabilities into actual platform features it's about more than just following accessibility guidelines it's about having an intuitive layout a clear visual hierarchy and consistent navigation so that everyone can use the platform easily and comfortably they're not just making it accessible they're making it *usable* for everyone which are two related but different things you know one without the other doesn't work well

then you need the engineers they're the ones actually implementing these designs but it's not about just blindly coding what the designers give them they need to understand the 'why' behind the decisions and they need to be able to bring their own understanding of the tech to the table to maybe suggest alternative more efficient or more scalable implementations that still uphold those accessibility and usability principles it's a back and forth constantly refining the platform

after that the behavioral sciences and psychology they are huge they get into how people actually interact with technology they bring the insights about user behavior user cognitive biases and how to make interactions feel natural and not frustrating they can tell you about how things like information overload can impact users with cognitive differences and they guide the design process with data from user testing and research they are the ones bringing the humanity and they inform what actually works versus what is just a well intentioned guess this means a lot more than simple a/b testing you need the why behind what the user does not just what button they click the most

here comes the domain experts whatever area zelmira is focusing on if it’s educational they need teachers if it's healthcare they need doctors or nurses and other healthcare workers the people who know the specific needs and issues of the specific user base they also provide the input on the content structure so for example a teacher will know how a student would navigate the platform differently than a designer would and what resources a student would need to have access to these users' input is critical for making sure that the platform really meets the user's needs and that it's useful in the real world not just in a test environment and it's about more than just features it's about whether or not the platform fits in a user's workflow

and legal and ethical considerations are big too if we’re dealing with sensitive data that’s where lawyers or ethical experts come in they will help navigate the landscape of privacy laws data security and responsible technology development they will tell you how to ensure user data is being handled responsibly and ethically it’s about building trust and being transparent about how the platform works and making sure everyone knows their rights and also prevent bias in the development of algorithms or ai based features if you use them you know bias can come from the data you use or how you train a model they are the ones making sure that this tech doesn't unintentionally harm the people that it is trying to help

so how does this all play out in real code well you might see something like this for example in a user interface component we are not going to just focus on the looks of the component but we also keep in mind accessibility
```javascript
// Example UI component for a list with accessibility in mind
function AccessibleList({ items }) {
  return (
    <ul role="list">
      {items.map((item, index) => (
        <li key={index} role="listitem">
          <a href={item.link} aria-label={item.label}>
            {item.text}
          </a>
        </li>
      ))}
    </ul>
  );
}
```
in the snippet above you see that the code is not just rendering the list it is also providing aria attributes to ensure that the user agent and the screenreader interpret the list structure so that those with accessibility needs have an easier time navigating the content

or think of data handling here’s a simple example of filtering data but with accessibility context in mind
```python
def filter_data(data, query, user_preferences):
    filtered_data = []
    for item in data:
        if query.lower() in item['text'].lower(): # simple search
            if user_preferences['contrast'] == 'high':
                if item['high_contrast_version'] :
                   filtered_data.append(item['high_contrast_version'])
                else:
                   filtered_data.append(item)
            elif user_preferences['font_size'] == 'large':
                if item['large_font_version']:
                    filtered_data.append(item['large_font_version'])
                else:
                   filtered_data.append(item)
            else:
                filtered_data.append(item)
    return filtered_data
```
in this snippet you have some filtering logic but it is also considering the user preferences on contrast and font size these are not common filtering requirements in most platforms but this is just an example of how to think about it and keep in mind these concerns at the coding level, it is not just an aesthetic thing.

let's look at an example where we need to avoid some biases in an algorithm if we are using machine learning
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(features, labels, sensitive_attribute):
    #split data, making sure to stratify by sensitive attribute (e.g. age)
    X_train, X_test, y_train, y_test,sensitive_train,sensitive_test= train_test_split(features, labels,sensitive_attribute,test_size=0.2, stratify=sensitive_attribute)

    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    # check performance by the stratified test sets
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Overall accuracy: {accuracy}')
    # calculate and print the accuracy per sensitive attribute group
    unique_sensitive_values = np.unique(sensitive_test)
    for value in unique_sensitive_values:
        group_mask = sensitive_test == value
        group_accuracy = accuracy_score(y_test[group_mask], predictions[group_mask])
        print(f'accuracy for group {value}: {group_accuracy}')
    return model
```
in this snippet you can see that when we are splitting the data we are using stratify this means that the training set and the test set have similar distributions when we check for model performance we also check the perfomance per each sensitive attribute to be sure that one subgroup doesn't benefit from the model more than another this is a very basic way of reducing bias but you can get the idea these concepts need to be in mind when coding not as something an independent team does later but in each line of code, it needs to be at the foundation level

so yeah interdisciplinarity it's not just a buzzword it's a necessity for building truly inclusive platforms and it goes way deeper than just different people in the same room it's about genuine collaboration a shared understanding of the goals and the challenges it's about constant communication iteration and making the needs of all users a central part of the development process and not an afterthought this approach makes it more work more effort more conversations more back and forth but it's worth it in the end for the platform to be actually what it means to be and not some empty promise of inclusivity

for deeper understanding of the topics i mentioned you can check out resources like "Designing with the Mind in Mind" by Jeff Johnson this is great for user experience and usability “A Pattern Language” by Christopher Alexander et al is useful for thinking about design problems generally and for accessibility stuff the web content accessibility guidelines WCAG are the gold standard to start with and for ethical concerns and AI bias i would recommend “fairness and machine learning” by barocas, hardt and narayanan

so in the end interdisciplinarity isn’t just a nice to have its fundamental for platforms like zelmira to actually work for everyone and not just for a select few.
