---
title: "How can a state-based RASA dialog management system be migrated to a story-based approach?"
date: "2024-12-23"
id: "how-can-a-state-based-rasa-dialog-management-system-be-migrated-to-a-story-based-approach"
---

Alright,  I've personally navigated this transition more than once, and it can feel a bit like moving from a well-organized filing cabinet to a more free-form, but powerful, notebook. We're talking about shifting from a state-based dialog management approach in RASA, which often relies on meticulously defined states and transitions, to a story-based one that focuses on the conversational flow as it's naturally experienced by the user. The core of the challenge lies in translating the structured logic of states into the more fluid narrative of stories, and it's definitely achievable with a thoughtful plan.

Essentially, a state-based system, think of it as a finite state machine, operates by defining specific states the conversation can be in—like “awaiting_name,” “awaiting_city,” or “order_confirmation”. Transitions between these states are triggered by user input and potentially bot actions. This makes for a very predictable conversation path. Story-based, on the other hand, utilizes examples of real conversations, or “stories,” to train the model. These stories are sequences of user intents, entities, and bot actions that represent how a dialog might unfold. This makes the model more robust, allowing it to handle variations in user input more effectively, but it requires more training data.

The migration isn't a simple "copy-paste" operation. Here's the roadmap I’ve found effective, broken down into key steps with example code snippets:

**Step 1: Understand Your Current State Machine**

Before we move anything, we need a clear picture of your existing state-based implementation. This involves carefully examining your domain file, particularly the states defined, the transitions between them, and the actions triggered at each step. Specifically, identify where transitions are triggered based on slot values, user intents, or external events. If you’re employing custom actions, note how these are changing the dialogue state. This meticulous understanding forms the bedrock for moving forward.

**Step 2: Identify Canonical Conversation Flows**

Now, start visualizing what these state transitions represent as actual conversations. Begin to abstract and reframe your state machine into a few key, typical paths. In short, write out some exemplary conversations. Think of these as the “happy path” stories—the common sequences of user input and bot response that will make up the bulk of your story data. You don't need to detail *every* possible edge case at this stage, concentrate on the core, most frequent interactions.

**Step 3: Create Initial Stories from Key Flows**

This is where the real conversion begins. Take those typical conversation flows you defined and convert them into RASA story format (typically within your `data/stories.yml` file). Each story should be a sequence of user intents and entities, followed by the corresponding bot actions and potentially slot setting. The real trick here is to include variations in these story paths: this can be in the way of asking the same information in slightly different ways by the user, or the same response given in a different way by the bot. Let's look at a simple example:

*Example 1: Initial Story Creation*

Let’s say we had a state machine that transitioned through the following states to order a pizza: `start` -> `awaiting_pizza_type` -> `awaiting_size` -> `awaiting_address` -> `confirmation`. Here’s how that might translate into a basic story:

```yaml
stories:
  - story: pizza order happy path
    steps:
      - intent: greet
      - action: utter_greet
      - intent: order_pizza
      - action: utter_ask_pizza_type
      - intent: specify_pizza_type
        entities:
          pizza_type: pepperoni
      - slot_was_set:
        - pizza_type: pepperoni
      - action: utter_ask_size
      - intent: specify_pizza_size
        entities:
          pizza_size: large
      - slot_was_set:
        - pizza_size: large
      - action: utter_ask_address
      - intent: provide_address
        entities:
          address: 123 Main Street
      - slot_was_set:
        - address: 123 Main Street
      - action: utter_confirm_order
```

Notice how the story steps, through intents, entities, slot settings, and bot actions, closely resemble the previous state transitions, but are now expressed as a continuous narrative.

**Step 4: Introduce Variability and Edge Cases**

Now that you have a core set of stories, it’s vital to incorporate the variations and edge cases that a state-based system is often meticulously designed to handle. This includes handling user corrections, disambiguation, and unexpected input. Expand your stories by creating variations in how the user might ask the same questions. This might mean providing slightly different user intents that mean the same thing but phrased slightly differently. It also means handling interruptions, when the user changes the subject halfway through the conversation. Add these into your stories. Here’s an example:

*Example 2: Introducing Variability*

```yaml
stories:
  - story: pizza order with correction
    steps:
      - intent: greet
      - action: utter_greet
      - intent: order_pizza
      - action: utter_ask_pizza_type
      - intent: specify_pizza_type
        entities:
          pizza_type: cheese
      - slot_was_set:
        - pizza_type: cheese
      - action: utter_ask_size
      - intent: specify_pizza_type
        entities:
          pizza_type: pepperoni
      - slot_was_set:
          - pizza_type: pepperoni
      - action: utter_ask_size
      - intent: specify_pizza_size
        entities:
          pizza_size: large
      - slot_was_set:
        - pizza_size: large
      - action: utter_ask_address
      - intent: provide_address
        entities:
          address: 123 Main Street
      - slot_was_set:
        - address: 123 Main Street
      - action: utter_confirm_order
```

In this example, we've shown how the user might correct a previously selected pizza type. We are setting the slot again to what the user has stated more recently.

**Step 5: Leverage Form Actions for Complex Scenarios**

If you have highly structured interactions or required information gathering, form actions in RASA can be very valuable. They help structure a conversation flow with slot filling validation and allow for more complex logic. If your old state-based logic had custom actions to manage dialogue states, this is where you would consider transitioning those into form-specific actions. The RASA documentation offers examples of this, particularly concerning how forms can be set up within `domain.yml` and custom actions.

*Example 3: Using Form Action*

Let's take the previous examples a step further by utilizing a form action:

```yaml
forms:
    pizza_order_form:
        required_slots:
            pizza_type:
            - type: from_entity
              entity: pizza_type
            pizza_size:
            - type: from_entity
              entity: pizza_size
            address:
            - type: from_entity
              entity: address

stories:
    - story: pizza order using form
      steps:
        - intent: greet
        - action: utter_greet
        - intent: order_pizza
        - action: pizza_order_form
        - active_loop: pizza_order_form
        - slot_was_set:
            - pizza_type: pepperoni
        - slot_was_set:
            - pizza_size: large
        - slot_was_set:
            - address: 123 Main Street
        - action: utter_confirm_order
        - active_loop: null
```
In `actions.py`, you would now define the logic of the `pizza_order_form` including how each slot will be filled and what will happen once the loop is completed. The above is simplified for the purpose of the example, but gives a flavour of how forms could be implemented here.

**Step 6: Incremental Training and Evaluation**

Once you've created a sizable collection of stories, start training your RASA model iteratively, and test frequently. Focus on training on a diverse range of user utterances within the stories that you provide, to really push the model. RASA's testing tools are essential for this phase. Don’t try to do everything all at once – migrate core functionality and expand gradually, testing each step of the way.

**Recommended Resources:**

For a deeper understanding of the theoretical underpinnings of dialogue systems, I highly recommend Daniel Jurafsky and James H. Martin's "Speech and Language Processing". This resource provides foundational knowledge that will improve your understanding of how to approach these problems in the future. For specific RASA techniques, "Hands-On Chatbots with Rasa" by Alan Nichol and others is a very useful resource, and, of course, the official RASA documentation is a must, specifically their section on stories and domain files. Keep in mind that RASA's documentation is continuously updated, so ensure you are using the documentation that corresponds to the version of RASA you are employing.

The shift from a state-based to a story-based approach involves a change of perspective but one that can make your chatbot more flexible and robust. It’s not always a straightforward process, but following these steps, you can effectively migrate your system and enjoy the benefits of a more conversational, story-driven approach to dialogue management. Remember, continuous iteration and testing are key. I've found this approach very fruitful in my own work and trust you will too.
