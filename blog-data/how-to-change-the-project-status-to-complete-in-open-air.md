---
title: "How to change the Project status to COMPLETE in Open Air?"
date: "2024-12-15"
id: "how-to-change-the-project-status-to-complete-in-open-air"
---

alright, so you're looking to flip a project's status to 'complete' in open air. i've been there, done that, and got the t-shirt, more than once, and each time it's a little different depending on the specifics of your open air setup. it's not always as straightforward as you'd hope. let's break this down.

first off, open air, as a platform, is quite configurable, which is a double edged sword. it means it can be powerful, but also that there isn't always one single way to do things. i'm assuming we're talking about a fairly standard project object within the open air ecosystem. we’re not going into those weird custom objects some places use that frankly, nobody understands. i've personally seen projects where they've used open air objects for things they really, really shouldn't have, like tracking lunch orders, and trust me, changing a status in those scenarios is… an adventure. i remember one place we moved to open air and someone setup the database as if it was a big excel sheet, i almost had to use a spork to fix that. i'm talking like hundreds of columns, it was insane. anyway, let's get back to our actual goal: setting a project's status to 'complete.'

the core issue here is that open air projects generally transition through a predefined workflow. this means there are specific steps and rules that need to be satisfied before you can simply mark a project as complete. it’s not just a simple database update, and we need to respect that, trust me. i've had my fair share of runtime errors because i thought i could just bypass the workflow.

most often, the status change is controlled by a combination of user permissions, field dependencies, and potentially, custom rules. think of it as a series of gates you need to pass. you can't just waltz through the final gate, which in this case is the complete status. sometimes, there are fields that need to be populated or tasks that need to be finished first. it's almost like open air is yelling at you with errors if you don't do that, in its own special way. i've personally spent hours troubleshooting only to find that one silly required field was not checked which prevented the status change. once a project was stuck for weeks because somebody thought that the field “if project should be deleted” was not required and set it as not required... when in fact it was always required.

usually, you have two main avenues to consider when completing a project, the ui and the api. let's cover them both.

**using the open air user interface (ui)**

this is usually the first place you'd look. navigate to the specific project in question. you'll likely see a section for the status or workflow of the project. it might be a drop-down menu or some kind of action button.

here are some typical steps that might be required:

*   **check for outstanding tasks:** make sure all associated tasks or milestones are marked as complete. look for any tasks that might be lingering with a "pending" or "in progress" status. i've been caught out a few times with these because someone forgot to update a small thing. sometimes they are really hard to find as well.
*   **review and complete financials:** some workflows won't let you complete a project unless all financials are reconciled. this might include invoicing, expense reports, and time entries. there were projects where time entries where left unchecked for weeks and we had to get the entire project team to re-enter their time just to close out the project.
*   **check dependencies:** are there dependencies on other projects or other objects? complete those as well as needed. sometimes, a project might be waiting for a sub-project to be done or other dependencies. i saw one of those and it ended in tears. it was a multi-dependencies chain and it was so long that nobody could follow it, we had to get an external consultant just to figure it out.
*   **look for error messages:** open air usually provides guidance in the form of errors if a status transition cannot happen. read these closely, they usually contain hints to what's preventing the change. open air, surprisingly, is pretty good at telling you what's wrong when it wants to. i usually get all the time because i missed something super silly.

if you encounter trouble, take a screenshot of the relevant section in the ui. the error message might prove useful in asking other colleagues or the support team. a screenshot is usually faster than me trying to explain in words the exact thing that i am seeing. remember to blur sensitive information, please.

**using the open air api**

if you are doing a large amount of projects or need to automate the process, using the api might be your best shot. the api allows you to interact with open air programmatically. this is very useful if you have many projects to update or need to integrate the status change into your workflow with other systems.

i'm going to give you python examples, since i tend to prefer python. also, you can copy and paste this and test it with some adjustments as needed. this assumes you have the correct python libraries installed (requests or a dedicated open air api client library if it exists and that you're not hitting any server firewall issues). usually there is an openair library available. if you don’t have the library installed, search for the openair python api package on your preferred package manager and install it. this is not the scope of this response though.

```python
import requests
import json

def complete_project(project_id, api_token, openair_url):
    """
    sets the status of a project to complete using the open air api.
    """
    headers = {
        'content-type': 'application/json',
        'authorization': f'bearer {api_token}'
    }

    url = f'{openair_url}/api/v1/projects/{project_id}'

    data = {
        "status": "complete"
        #you may need additional fields here, for example:
        #"actual_completion_date": "2024-03-15T12:00:00.000Z",
    }
    try:
        response = requests.put(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  #raise http errors for failed requests.
        return response.json()
    except requests.exceptions.RequestException as e:
      print(f"an error occurred: {e}")
      return None


# usage example
if __name__ == "__main__":
    project_id = 12345 # replace with the actual project id.
    api_token = "your_actual_token_here" #replace this with your open air api token
    openair_url = "your_openair_instance_url" # replace this with your actual open air instance url
    result = complete_project(project_id, api_token, openair_url)
    if result:
        print(f"project {project_id} updated successfully: {result}")
    else:
        print(f"project update of id: {project_id} failed.")


```

this is a basic example. the actual open air api might require additional fields or be structured differently, so check your specific documentation. also some apis can be a bit... particular. i have seen APIs that can throw a tantrum if you are not sending the data in the exact way they want, including the order of the fields.

here is an example with the dedicated openair python library assuming it exists:

```python
from openair import Client

def complete_project_with_library(project_id, api_token, openair_url):
  """
  sets the status of a project to complete using the open air api library.
  """

  try:
      client = Client(api_token=api_token, api_url=openair_url)
      project = client.projects.get(project_id=project_id)

      project.status = "complete"
       #you may need additional fields here, for example:
       #project.actual_completion_date = "2024-03-15T12:00:00.000Z"

      client.projects.update(project)

      return f"project {project_id} updated successfully."
  except Exception as e:
       print(f"an error occurred: {e}")
       return None

# usage example
if __name__ == "__main__":
  project_id = 12345 # replace with the actual project id
  api_token = "your_actual_token_here" #replace this with your open air api token
  openair_url = "your_openair_instance_url" # replace this with your actual open air instance url
  result = complete_project_with_library(project_id, api_token, openair_url)

  if result:
    print(result)
  else:
    print(f"project update of id: {project_id} failed")
```

this example assumes that the library is written in a certain way. check the library documentation for your exact usage.

and one last example with the use of a custom api client and the requests library:

```python
import requests
import json


class OpenAirClient:
    def __init__(self, api_url, api_token):
        self.api_url = api_url
        self.api_token = api_token
        self.headers = {
            'content-type': 'application/json',
            'authorization': f'bearer {self.api_token}'
        }

    def update_project(self, project_id, data):
        url = f'{self.api_url}/api/v1/projects/{project_id}'
        try:
             response = requests.put(url, headers=self.headers, data=json.dumps(data))
             response.raise_for_status()
             return response.json()
        except requests.exceptions.RequestException as e:
              print(f"an error occurred: {e}")
              return None


def complete_project_with_custom_client(project_id, api_token, openair_url):
   """
  sets the status of a project to complete using a custom open air api client.
  """

   client = OpenAirClient(api_url=openair_url, api_token=api_token)
   data = {
       "status": "complete"
       #you may need additional fields here, for example:
       #"actual_completion_date": "2024-03-15T12:00:00.000Z",
       }
   result = client.update_project(project_id, data)
   if result:
      return f"project {project_id} updated successfully: {result}"
   else:
      return None


# usage example
if __name__ == "__main__":
    project_id = 12345 # replace with the actual project id
    api_token = "your_actual_token_here" #replace this with your open air api token
    openair_url = "your_openair_instance_url" # replace this with your actual open air instance url
    result = complete_project_with_custom_client(project_id, api_token, openair_url)
    if result:
       print(result)
    else:
       print(f"project update of id: {project_id} failed")

```

**debugging tips**

*   **check the open air api documentation:** this is essential. your open air instance might have a custom api and the generic documentation may not be enough. they may have changed some functionality of the api too.
*   **enable verbose logging:** use some logging to see the responses coming from the api and check them. i cannot stress how helpful this is. api logging is like a debugging superpower.
*   **verify user permissions:** make sure the user or api key has the necessary permission to modify the project status. your user role may be preventing you to set the status to complete. i have found myself many times with the wrong permission and it was a pain to figure out what the problem was.
*   **use a tool to inspect the api calls:** tools such as postman or insomnia are useful to inspect the actual api calls. sometimes your code is right but the server is sending something different. this will prove useful in that case.
*   **inspect the error codes:** pay close attention to error codes. they usually indicate the exact issue you are dealing with.

**recommended resources**

instead of throwing random links at you, i'd recommend looking at the following types of resources. these usually have more long-term value:

*   **your open air instance's specific documentation:** this is probably your best starting point. look for admin guides, api references and user manuals. it should be in your open air instance documentation.
*   **books about workflow management:** "workflow modeling: tools for process improvement and application development" by alec sharp and patrick mcdermott could provide you with the theory behind workflow systems in general and you will understand better what you are dealing with. this can be very helpful to navigate the complexity of open air.
*   **books about web api design:** "restful web apis" by leonard richardson, mike amundsen, and sam ruby can help you debug api issues and write better api requests and understand how an api works.
*   **python request library documentation**: for the python code provided, the library requests’ documentation will help understand how the methods work and what you can do with it. the official documentation is located in the requests page.
*   **consult your internal open air admin or support team:** they may have custom documentation or insights into the specific workflow implementation for your company. they probably already encountered a similar situation before.

so, there you go. changing a project's status in open air can be a little bit of a process, but with a bit of patience and the resources mentioned above, you should be able to mark those projects as complete. and remember: the worst possible thing is for code to not compile, or a project to be incomplete. haha, coding humour. if i can think of anything else i'll let you know. good luck, and happy coding.
