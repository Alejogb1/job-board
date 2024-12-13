---
title: "how to check if a sidebar checkbox is clicked in streamlit app?"
date: "2024-12-13"
id: "how-to-check-if-a-sidebar-checkbox-is-clicked-in-streamlit-app"
---

Okay so you want to know how to check if a sidebar checkbox is clicked in a Streamlit app right Been there done that a few times actually It's one of those things that seems simple but can trip you up if you're not paying attention

So the core issue here is that Streamlit widgets like checkboxes rerun your entire script whenever they're interacted with This is kinda key to understand because it dictates how you check for that click You don't get some callback function or event listener like you might in other frameworks It's all about the state of your script during each rerun and how you use it

Let me tell you about the first time I wrestled with this It was a small data analysis tool I was building back in the day We had this side panel for filtering data and of course we put a bunch of checkboxes for each filter category The first version was a hot mess of globals and conditional rendering spaghetti I mean It worked for a bit but as soon as I tried to expand the number of filters things went haywire I had to learn the hard way about how Streamlit actually handles state

Anyway enough reminiscing let's cut to the chase The basic idea is that Streamlit returns the current state of the checkbox when you call `st.sidebar.checkbox()` during each rerun If the checkbox is checked it returns `True` if not it's `False` The trick is you use that return value in a conditional statement to do what you need to do

Here's a straightforward example to get you started

```python
import streamlit as st

st.title("Simple Checkbox Example")

with st.sidebar:
    checkbox_state = st.checkbox("Enable Feature A")

if checkbox_state:
    st.write("Feature A is enabled")
else:
    st.write("Feature A is disabled")

```

See how easy that is We set `checkbox_state` to the returned value and then the conditional logic just works as expected This is the bread and butter of Streamlit interactions and it works pretty consistently when used like that

Now you might be thinking okay that's cool for a single checkbox what about multiple Well it's the same principle you create more widgets and then you evaluate their states in your conditional logic Let's say you had a few filter checkboxes

```python
import streamlit as st

st.title("Multiple Checkboxes")

with st.sidebar:
    filter_a = st.checkbox("Filter A")
    filter_b = st.checkbox("Filter B")
    filter_c = st.checkbox("Filter C")

st.write("Selected filters:")

if filter_a:
    st.write("- Filter A")
if filter_b:
    st.write("- Filter B")
if filter_c:
    st.write("- Filter C")

```

Here you see the state of each checkbox is checked individually and the desired output is shown on the screen This logic is pretty fundamental for building interactive dashboards and tools in Streamlit

You could also use a more concise method if you wanna be extra fancy like this which has the same functionality but looks slightly different

```python
import streamlit as st

st.title("Multiple Checkboxes concise method")

with st.sidebar:
  filters = {
    "Filter A": st.checkbox("Filter A"),
    "Filter B": st.checkbox("Filter B"),
    "Filter C": st.checkbox("Filter C"),
  }

st.write("Selected filters:")

for filter_name, filter_state in filters.items():
  if filter_state:
    st.write(f"- {filter_name}")

```

This method is a little more scalable since if you need to add a new filter you just add it to the dictionary and thats it

Now a common mistake I've seen and I might've done it myself in the past (don't judge) is using globals for state It's a tempting shortcut but it can lead to weird unexpected behavior Remember Streamlit reruns the script from top to bottom on every interaction So if you're using globals that are modified during these interactions you can get your state out of sync which results in your application not working the way it was intended

Another common question people have is how to reset the state of the checkboxes Well since Streamlit doesn't explicitly provide a mechanism for reseting it on a button click you have to simulate it in your own way The easiest way is to use the key parameter of streamlit widgets so each time the program run if you change the key values then Streamlit will create a new widget so with that you are reseting the state of your widgets You can do this in the following way

```python
import streamlit as st

st.title("Checkbox Reset Example")

def create_sidebar(key):
    with st.sidebar:
        return st.checkbox("Resetable Checkbox",key = key)

if 'counter' not in st.session_state:
    st.session_state['counter'] = 0

checkbox_state = create_sidebar(st.session_state['counter'])

if st.button("Reset"):
    st.session_state['counter'] += 1
    st.experimental_rerun()


if checkbox_state:
    st.write("Checkbox is checked")
else:
    st.write("Checkbox is not checked")
```

You see here we use a session state variable to track how many times the user reset the widgets and with each reset we change the key param of the widget so it has a new state when the application rerun

Now if you're starting to get serious about Streamlit and handling more complex interactions I'd recommend checking out the official Streamlit documentation It's a pretty good resource to start with and it covers the core concepts nicely and then you should probably check the paper "Declarative Programming with Dataflow in the Python Ecosystem" by T. Eifler et al It discusses the core concept of dataflow programing which is what is used by Streamlit under the hood this paper will make you understand why Streamlit works like it works. Finally after that you can check "Python Crash Course" by Eric Matthes which is a nice intro to Python programming and might be good to refresh your python skills if you have not been using it for a long time

Oh and one more thing before I go a programmer walks into a bar orders 1.42 beers and the bartender say I am sorry that is a float and the programmer says I guess so

I've spent way more time debugging than I'd like to admit working on these things but it's a necessary evil I guess You just keep iterating and you get better at it so stick with it
