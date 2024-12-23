---
title: "how can i set v rating to be fixed on vuetify?"
date: "2024-12-13"
id: "how-can-i-set-v-rating-to-be-fixed-on-vuetify"
---

 so you're wrestling with Vuetify's v-rating component and want to nail down that rating so users can't futz with it eh I've been there trust me it's a classic. Spent a good chunk of a weekend debugging a similar issue back in 2018 when I was working on a user feedback system. The user kept clicking on the rating and it was updating without proper server side validation. It was a total mess.

So first thing’s first and I know this is basic but you gotta make sure you’re understanding what’s going on.  Vuetify’s `v-rating` is fundamentally a UI element that reflects and accepts user input. You are correct that it is a two way data binding component which is fine. If you want a "fixed" rating that means you want the component to display a value but not let the user change it. This means you need to use the :value prop instead of v-model to disable user interaction. Think of it like setting a default value in HTML with the difference that in v-model the user can change the value while in :value the value is set and only the value is being displayed.

Let’s get to the point you can use the `:readonly` prop to disable interaction you'll always have to set the :value prop to have a stable read only value and if you want the value to be changed you’ll have to change it from the parent component.

Here’s the most straightforward way to get that behavior and it’s the one I always use now after that nightmare of 2018:

```vue
<template>
  <v-rating
    :value="fixedRating"
    readonly
  ></v-rating>
</template>

<script>
export default {
  data() {
    return {
      fixedRating: 3, // Set the fixed rating value here
    };
  },
};
</script>
```
See how we use `:value` instead of `v-model` That `:value` prop binds the rating to the `fixedRating` data property and the readonly is a Boolean. This ensures the user can't click to change the stars but they'll see the 3 stars highlighted. And the value will always be three no matter what the user does.

Now sometimes you might want a way to trigger a change in the fixed rating from outside the component that's when you have to remember that components in vuejs have props. For instance you're in a more complex parent component and need to pass the value through a variable in that parent component like so:

```vue
// ParentComponent.vue

<template>
  <div>
    <FixedRatingComponent :ratingValue="dynamicRating" />
    <button @click="changeRating">Change Rating</button>
  </div>
</template>
<script>
import FixedRatingComponent from './FixedRatingComponent.vue'
export default {
    components: {
        FixedRatingComponent
    },
  data() {
    return {
      dynamicRating: 3,
    };
  },
  methods: {
    changeRating() {
      this.dynamicRating = Math.floor(Math.random() * 5) + 1; //Just some random number from 1 to 5
    }
  }
};
</script>
```

```vue
// FixedRatingComponent.vue
<template>
  <v-rating
    :value="ratingValue"
    readonly
  ></v-rating>
</template>

<script>
export default {
    props: {
        ratingValue:{
          type: Number,
          required: true
        }
    }
};
</script>
```

This time we're passing the value through the prop `ratingValue` instead of using it locally in the `FixedRatingComponent`. Now the parent component will call the `changeRating` function that changes the `dynamicRating` value and the child `FixedRatingComponent` will read the prop and update the rating. If you remove `:readonly` the component starts being interactive again which is what I was doing back in 2018 and was the reason for the bug. (Don’t make the same mistakes I did).

 so I got one more example for you that might be helpful. This time what if you want to make it reactive not only through changes from the parent but you want to initialize the value from a data that is being changed in the same component. Let’s say you get data from an API and you want to show the rating once you’ve got the info:

```vue
<template>
  <v-rating
    :value="apiRating"
    readonly
  ></v-rating>
</template>

<script>
export default {
  data() {
    return {
      apiRating: 0, // Initial value
      isLoading: true
    };
  },
  async mounted() {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      this.apiRating = Math.floor(Math.random() * 5) + 1;
      this.isLoading = false
    } catch (error) {
      console.error("error fetching the data", error)
    }
  },
};
</script>
```
In this case, we initialize `apiRating` to 0 then we call the API that returns the rating in the `mounted` lifecycle hook. It is important to not use `async` `setup` because in that case the `this` variable will point to `undefined`. We are using a simulated API call by using a `setTimeout` with a promise in this case but that is not important. The important part is that the `v-rating` will update as soon as `this.apiRating` changes.

It's also worth pointing out that with the `readonly` prop the user interaction is disabled completely. The user will not see the hover effect for example if you pass `hover` prop. The user will not be able to interact with the v-rating component at all. And this is what you are looking for.

A crucial concept to grasp here especially when working with a lot of Vuejs or Javascript based components is the understanding of declarative vs imperative programing styles. The difference is really important since the core is understanding how your data changes and how you react to those changes. We are aiming here for a declarative approach.  I mean you’re essentially telling Vuetify: "Here's the `value`, display it and keep it constant and do not allow interaction.". You are declaring the value and letting the component react to it. This is the heart of using Reactivity.

As for resources beyond the Vuetify documentation which you probably already checked I would advise looking into books that focus on component based design like “Component-Based Software Engineering” by  George T. Heineman and William T. Councill. Its a little old but it goes deep into how component interact with each other. Also any book that focuses on reactive programing will be a plus. You can check out books about functional reactive programing too since they go hand in hand with declarative programming like “Functional and Reactive Domain Modeling” by Debasish Ghosh that dives deep into event-driven architectures.

One more thing. It's easy to get caught up in thinking of the UI layer as this separate entity. But really it’s just a reflection of your data model. If your data says a rating should be 3 then it should be 3. You know what they say about opinions though right… they’re like ratings everyones got em. (I had to do it sorry). Keep that in mind and always double check your data source.

Good luck debugging keep at it you’ll get it. I bet it’s just a small detail like a missing :readonly or `:value` prop.
