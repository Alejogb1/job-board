---
title: "How can you track Azure OpenAI API usage and token performance effectively?"
date: "2024-12-03"
id: "how-can-you-track-azure-openai-api-usage-and-token-performance-effectively"
---

Hey so you wanna track your Azure OpenAI API usage and how well those tokens are doin huh That's a super important thing to keep an eye on especially if you're paying for it which you probably are It's not just about the money though Understanding your token performance helps you optimize your prompts make your models more efficient and generally avoid unexpected bills or performance issues Let's break it down

First off you've gotta get the data Azure provides monitoring tools built right in They give you different views depending on what you care about cost consumption token counts response times you name it Think of it like a dashboard for your AI spending and activity

You can dive into Azure Monitor logs This is where you find granular details about every single API call you make It's like a super detailed audit trail Each entry will usually have timestamps the specific API call you made the number of tokens used and maybe even some error codes if anything went wrong It's a bit overwhelming at first but super powerful once you get the hang of it You might wanna check out the Azure Monitor documentation or maybe a book on Azure cloud monitoring There's usually a good chapter on how to query and visualize these logs effectively

Then there's Azure Cost Management This one's all about the money obviously It summarizes your spending nicely shows you trends and lets you set budgets and alerts It's great for high-level overview but doesn't give you the granular token level detail Azure Monitor gives you Think of it as a summary report of your Azure OpenAI spending compared to the detailed breakdown you get from the logs Again the Azure documentation or a cloud financial management book would be a good resource here

Now for actually getting the data and doing something useful with it you've got options You could use the Azure portal's built in visualization tools It's pretty simple to use and easy to get started with But if you need more control or want to integrate this into other systems you'll likely want to use something more powerful like Azure Data Explorer or even pull it into your favorite data analytics platform using APIs or the CLI

Let's say you wanna get really hands on and write some code Here are a few examples These are Python scripts just 'cause that's what I usually use You could adapt them to other languages pretty easily

**Example 1: Basic Token Count Tracking**

This script just shows you how to log the number of tokens used in each API call It's a very simplified example but shows the basic principle

```python
import openai
import logging

# Set up logging
logging.basicConfig(filename='token_usage.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Your Azure OpenAI key and endpoint
openai.api_key = "YOUR_API_KEY"
openai.api_base = "YOUR_API_ENDPOINT"

def send_prompt(prompt):
    try:
        response = openai.Completion.create(
            engine="your_model", # replace with your model name
            prompt=prompt,
            max_tokens=100  # Adjust as needed
        )
        token_count = response.usage.prompt_tokens + response.usage.completion_tokens
        logging.info(f"Prompt: {prompt}, Tokens used: {token_count}")
        return response.choices[0].text
    except Exception as e:
        logging.error(f"Error: {e}")
        return None


# Example usage
prompt = "Write a short story about a cat"
response = send_prompt(prompt)
if response:
    print(response)

```

This script uses the OpenAI Python library You'll need to install it using `pip install openai` Remember to replace `YOUR_API_KEY` and `YOUR_API_ENDPOINT` with your actual credentials and model name The logging part writes the prompt and token count to a file You can then use this log file to analyze your token usage over time It's a good start point for understanding things

**Example 2: More Sophisticated Monitoring with Azure Monitor**

This one's a bit more involved It uses the Azure SDK for Python to directly interact with Azure Monitor logs This allows you to query logs much more powerfully than just using a simple file

```python
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient

# Initialize Azure credential and Log Analytics client
credential = DefaultAzureCredential()
logs_client = LogsQueryClient(credential)

query = """
AzureOpenAI | where TimeGenerated > ago(1d)
| summarize count() by OperationName, Model, PromptTokens, CompletionTokens
"""

response = logs_client.query(query, workspace_id="your_workspace_id")
results = list(response)

for record in results:
    print(record)
```

This script requires installing the Azure SDK for Python Use `pip install azure-identity azure-monitor-query` You'll need to replace `your_workspace_id` with your Azure Log Analytics workspace ID This query retrieves data from the last day about API calls summarising by operation model and token counts It's a much more flexible query you can easily adjust to filter and aggregate the data in many ways This requires a bit more knowledge about Log Analytics query language Kusto Query Language You could check out the official Kusto documentation or some books on Azure data analytics

**Example 3 Integrating with a Data Visualization Tool**

This snippet shows a simplified idea of how you might integrate your usage data with a dashboarding tool like Power BI or Grafana

```python
# ... (Code to fetch data from Azure Monitor as in Example 2) ...

# Convert data into a format suitable for your visualization tool (e g CSV JSON)
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("openai_usage.csv", index=False)

# Then in your visualization tool import the CSV and create the dashboard

```

This section highlights the data processing and export aspect Many tools like Power BI or Grafana support CSV or JSON import and can create interactive dashboards This requires you to learn the basics of your chosen visualization tool's data import and manipulation capabilities

Remember these are just basic examples You can expand on them significantly You can add error handling more sophisticated querying automated alerts integration with billing systems and much more The key is to start small get comfortable with the data sources and gradually build a system that fits your needs

Tracking your Azure OpenAI usage and token performance is vital It's not just about cost management but also about optimizing your models and applications Using the Azure tools and potentially some custom code can provide the insights you need to improve efficiency and avoid any unpleasant surprises Go forth and monitor my friend
