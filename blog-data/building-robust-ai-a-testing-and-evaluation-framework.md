---
title: "Building Robust AI: A Testing and Evaluation Framework"
date: "2024-11-16"
id: "building-robust-ai-a-testing-and-evaluation-framework"
---

yo dude so i watched this vid on building an ai for real estate agents and it was wild the whole thing was basically about how these guys at reat built this crazy ai assistant lucy for realtors  think of it as a superpowered personal assistant that can handle all the tedious stuff like email marketing contact management and even creating websites and social media posts  the goal wasn't just to build it though it was to build it *right* and that's where the real story begins they started with a prototype which was like this rickety old cart using gpt-3.5 and react it was slow as molasses and made mistakes all the time but when it worked it was like magic you know that moment when tech just *clicks*  so they were like "we have a demo now lets go to production"  big mistake

the setup was that they initially used a super hacky method  lots of prompt engineering rag agents the whole shebang to build an mvp  it worked initially kinda like hotwiring a car but then they realized  this wasn't gonna cut it for a real product  the whole thing was falling apart and they needed some serious structure that's where the eval framework comes in

the first key moment was the shift from vibes to data  they started with pure intuition  like "it seems to work sometimes" which is like saying you built a rocket without any testing  they realized they needed a way to actually *measure* things quantify the success rate and pinpoint failures  thats where all the assertions came in

here's where the code comes in  imagine they're testing email functionality  this is what a simple assertion might look like in python

```python
import unittest

class TestEmail(unittest.TestCase):
    def test_email_sent(self):
        # Simulate sending an email, replace this with your actual function
        email_sent = send_email("test@example.com", "subject", "body")  
        self.assertTrue(email_sent, "Email was not sent")

    def test_email_valid_recipient(self):
        invalid_recipient = "invalid-recipient"
        with self.assertRaises(ValueError): #This is called an exception, or error-handling
            send_email(invalid_recipient, "subject", "body")

    def test_email_body_contains_placeholder(self):
        email_body = send_email("test@example.com", "subject", "body")
        self.assertIn("Dear [Client Name],", email_body, "placeholder missing")

if __name__ == '__main__':
    unittest.main()
```

this is a rudimentary example showing  unit tests in python using the unittest module  basically the test_email_sent function checks if a simulated email was sent correctly and fails if it's not,  test_email_valid_recipient checks for invalid emails and test_email_body_contains_placeholder verifies if necessary placeholders are there you can imagine expanding this to check various aspects of email generation and sending

another key thing they highlighted was logging and human review  they weren't just relying on automated checks  they were logging every interaction  every prompt response and every error  this is crucial because  it provides the data to train  the system and understand exactly how it's doing  and what it's getting wrong  they used something called logsmith which lets you do that  think of it as a super detailed record of everything lucy did  they then reviewed samples of this data manually  looking for patterns in the errors and successes  they even built their own custom data viewing tool using something like streamlit or gradio  because off-the-shelf tools didnt fit their needs


this leads to the second code snippet  imagine a super basic streamlit app for viewing logs:

```python
import streamlit as st
import pandas as pd

# load your logs here- replace with your actual data loading
logs = pd.read_csv("lucy_logs.csv")

st.title("Lucy Log Viewer")
st.dataframe(logs)

# Add filters or other functionalities as needed
selected_logs = logs[logs["status"] == "failed"]
st.write("Failed interactions:")
st.dataframe(selected_logs)
```

this is a skeleton streamlit app  it simply reads a csv file named lucy_logs.csv which is presumed to contain the logs and displays it using st.dataframe. It then shows only the failed interactions, providing a quick way to visually inspect what went wrong. you can make this so much fancier by adding plots graphs and interactive filters


the third major takeaway was the strategic use of llms for testing  since they didn't initially have tons of real user data  they used another llm to generate test cases  think of it as an llm playing the role of the real estate agent  asking different questions  and  feeding complex scenarios to lucy  this helped them improve test coverage and identify various edge cases and failure modes

this involved more code, but it's tough to show a complete example for that because it involves integrating with the underlying llm and your own application.  but the core logic would look something like this:

```python
import openai

def generate_test_cases(num_cases):
    prompt = """Generate 5 test cases for a real estate AI assistant. 
    Each test case should include:
    1. User request (a real estate task)
    2. Expected output (what the AI should produce)
    3. Potential failure modes (what could go wrong)
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        n=num_cases,
    )
    test_cases = [choice['text'] for choice in response['choices']]
    return test_cases


test_cases = generate_test_cases(5)
print(test_cases)
```

here's where we're using the openai api to generate test cases  you'd replace the text-davinci-003 with the appropriate model and adjust parameters as needed. the response is processed to extract the generated text.  this then gets fed into your application for testing.

another important point was  the iterative process of improving the prompt engineering  logging everything  and manually reviewing the data  it was a loop  they'd refine prompts  look at the logs  spot errors  and then iterate  this continuous improvement cycle was what finally brought lucy to a production-ready state

they also touched on llm as a judge but that was more of a later stage enhancement  it's basically using an llm to evaluate the quality of lucy's responses  however they emphasized that aligning the llm's judgments with human judgment is super important to ensure the assessment is accurate and meaningful  they used spreadsheets for this  keeping it simple and effective


the resolution was pretty clear  their systematic eval framework not only helped them get to production way quicker but also allowed for continuous improvement  they found that this approach was *far* superior to relying on intuition and vibes  they specifically mentioned some tricky things like mixing structured and unstructured outputs and handling complex multi-step commands things that pure prompt engineering alone just couldn't fix and they would not have been able to effectively handle them without the evaluation framework


it was a great illustration that building a robust ai system requires more than just cool tech it needs a solid foundation of testing monitoring and iterative improvement  and that building an evaluation framework is more important than jumping straight into fancy tools  like seriously  dont even think about it until you've nailed the basics and have a clear understanding of what you want to measure   they also called out that a lot of things are just chatgpt wrappers  and thats not the way to do it they emphasize fine tuning and custom development of their AI for their specific needs  thats it dude hope you got something out of this epic breakdown
