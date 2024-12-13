---
title: "streamlit dropdown menu dynamic code?"
date: "2024-12-13"
id: "streamlit-dropdown-menu-dynamic-code"
---

Okay I see the problem you're having with streamlit and dynamically updating dropdown menus yeah been there done that its a classic headache let me tell you I've wrestled with this issue more times than I care to remember especially back in the day when streamlit wasn't as mature as it is now feels like a lifetime ago

Right so the core problem is that streamlit reruns your entire script every time there's a UI interaction that includes changing a dropdown value and that can cause a real mess if you're not careful You've probably noticed your dropdowns not updating the way you expect or the whole app flashing like a christmas tree at times

Now back in like 2019 I was working on this massive internal dashboard for network monitoring we had all these interconnected devices and the user needed to be able to filter the data by device type and location so I thought dropdowns are perfect Right but when I tried to make them cascade the thing just went haywire It kept on reloading the whole dataset on every dropdown change taking like a solid 10 seconds each time it was maddening so yeah trust me I know your pain

The thing is streamlit uses this declarative pattern where every component in your app is rendered based on the current state of your script not on a continuous state like you might think in other frameworks

So to make your dropdowns dynamic you need to use streamlit's session state to store the selected values and then use that to conditionally update the options in the next dropdown it’s all about state management my friend and that requires a bit of planning and some careful implementation

Let’s break this down with some code examples the basic way to do it and it might be similar to what you already have in mind but still good to lay the foundations before diving into something more complex first this is just a single dropdown example

```python
import streamlit as st

options = ["Option 1", "Option 2", "Option 3"]

selected_option = st.selectbox("Select an option", options)

st.write(f"You selected: {selected_option}")

```

This is just a basic dropdown but it shows that selected_option will hold the value that the user selected and it will trigger a rerun of the script This is the key to use the state information for the dynamic dropdown menu in next example where the dependent dropdown’s options change based on the first one selected it can be pretty cool stuff

```python
import streamlit as st

if 'selected_category' not in st.session_state:
    st.session_state['selected_category'] = None

categories = {
    "Category A": ["Item A1", "Item A2", "Item A3"],
    "Category B": ["Item B1", "Item B2"],
    "Category C": ["Item C1", "Item C2", "Item C3", "Item C4"],
}

selected_category = st.selectbox("Select a category", categories.keys())
st.session_state['selected_category'] = selected_category

if st.session_state['selected_category']:
  items = categories[st.session_state['selected_category']]
  selected_item = st.selectbox("Select an item", items)
  st.write(f"You selected item {selected_item} in category {selected_category}")

```

See the magic here we are not only getting the value from the first dropdown but also saving the selected value in the session state `st.session_state`. It’s a python dictionary so we check `if 'selected_category' not in st.session_state:` for initialization and this avoids errors on the very first render when nothing is selected and after that the selected value is assigned using `st.session_state['selected_category'] = selected_category` and in the second part the dependent dropdown is rendered only if there is a selection on the first one meaning that the value of `st.session_state['selected_category']` is not None

Now remember those days debugging with print statements all over the place what a nightmare right? debugging with streamlit can be difficult when you start adding many dropdowns and it is important to remember that streamlit reruns the script on every interaction which can be a big head scratcher sometimes

But back to the topic let's add another level of depth the example below shows a three-level cascaded dropdown structure

```python
import streamlit as st

if 'selected_region' not in st.session_state:
    st.session_state['selected_region'] = None
if 'selected_city' not in st.session_state:
    st.session_state['selected_city'] = None

regions = {
    "Region A": {
        "City A1": ["District A1.1", "District A1.2"],
        "City A2": ["District A2.1", "District A2.2", "District A2.3"],
    },
    "Region B": {
        "City B1": ["District B1.1", "District B1.2", "District B1.3"],
        "City B2": ["District B2.1"],
    },
}

selected_region = st.selectbox("Select a region", regions.keys())
st.session_state['selected_region'] = selected_region

if st.session_state['selected_region']:
    cities = regions[st.session_state['selected_region']]
    selected_city = st.selectbox("Select a city", cities.keys())
    st.session_state['selected_city'] = selected_city

    if st.session_state['selected_city']:
        districts = cities[st.session_state['selected_city']]
        selected_district = st.selectbox("Select a district", districts)
        st.write(f"You selected district {selected_district} in city {selected_city} in region {selected_region}")
```

Notice the pattern here with multiple level cascading each dropdown is based on the previous selection that is saved in the session state in this case we added another level for city and districts we initialize the variables in the session state similar to before and the dropdowns are rendered conditionally and remember to use the correct session state variables when accessing them

This pattern of storing selections in `st.session_state` and conditionally rendering elements is fundamental to creating responsive interactive Streamlit apps with dynamic content especially with many elements

Now I'm not gonna lie there are more advanced tricks you can use like using callbacks but let’s leave that for another day for the scope of this problem you should be okay with this approach for simple cascading dropdowns

In terms of resources you should take a look at the streamlit documentation which is quite comprehensive by now especially the part about session state that would be a good start The "Streamlit Cookbook" is a good place too you can find many recipes there including ones with more complex session state handling

Also the “Data Visualization with Python” by Jake VanderPlas a classic data visualization book although it is not specific to streamlit will give you a more in depth understanding of data manipulation and data handling which is the backbone of any good streamlit app and maybe also a read at “Fluent Python” by Luciano Ramalho I know it is not specific to the topic but it gives you a good sense of how to structure complex programs which can help you keep your Streamlit apps organized and scalable trust me it helps more than you think when you are working with many moving parts

One more thing be very careful when modifying the values of the session state it’s easier than you think to end up in infinite loops if you start modifying values that are also used to conditionally render components and remember you should aim to modify the session state only in the dropdown selection logic

And there it is a brief summary of how to deal with dynamic dropdowns in Streamlit it's all about that session state management and conditional rendering once you wrap your head around it it becomes second nature and also if at some point you’re getting annoyed at how often the script reruns remember this and it's just part of the charm of Streamlit you get the hang of it believe me I’ve had plenty of sleepless nights trying to figure that out good luck and may your dropdowns render swiftly and accurately
