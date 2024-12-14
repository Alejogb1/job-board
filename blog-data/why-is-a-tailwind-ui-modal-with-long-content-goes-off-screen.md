---
title: "Why is a Tailwind UI modal with long content goes off-screen?"
date: "2024-12-14"
id: "why-is-a-tailwind-ui-modal-with-long-content-goes-off-screen"
---

alright, so you're hitting that classic issue where your tailwind ui modal, packed with more content than a thanksgiving turkey, decides to play hide-and-seek off the edge of the viewport. i've been there, trust me. this isn't some exotic edge case, it's a common pain point, and it usually boils down to how css handles overflow and viewport constraints when you throw lots of stuff into a relatively small container like a modal.

let's break it down like we're debugging a particularly stubborn piece of javascript. first, the default behavior of html containers is to try and fit the content inside, even if that means the content spills over. modals, often being fixed or absolute positioned elements, are especially prone to this since their parent container isn't always the immediate viewport. it could be something further up the dom tree which has fixed sizes that is causing the issue.

think of it this way: you have a box, the modal, and you keep adding more and more lego blocks (the content) inside. at some point, the box is simply not big enough, and the blocks will try to escape the box's confines. css doesn't automatically resize the box to accommodate all the blocks; it needs explicit instructions.

i've spent hours staring at my screen, convinced i was going insane, only to find out that the root of the problem was missing a single `overflow-y: auto;` somewhere in the chain. i remember one particular project where i was building an admin dashboard; the modal for adding new users had about 20 fields, and on smaller screens, it just vanished off the bottom of the page. it looked like a magic trick gone bad.

so how do we fix it? well, there are a few things we can look at. the most straightforward solution is usually related to setting the right overflow behavior on the modal or its containing elements. here's a basic example of how you might structure your modal in tailwind and apply that overflow setting to it:

```html
<div class="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
  <div class="bg-white rounded-lg shadow-xl p-6 w-full max-w-md h-auto max-h-[90vh] overflow-y-auto">
    <!-- modal content here -->
    <h2 class="text-xl font-bold mb-4">this is the title of the modal</h2>
    <p class="mb-4">a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.</p>
    <p class="mb-4">a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.</p>
    <p class="mb-4">a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.</p>
    <p class="mb-4">a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.</p>
    <p class="mb-4">a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.</p>
    <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">close</button>
  </div>
</div>
```

in this snippet, the `max-h-[90vh]` sets a maximum height for the modal, preventing it from growing indefinitely. the `overflow-y-auto` allows the content to scroll within the modal when it exceeds the maximum height set, this is also super important. remember that `h-auto` makes it grow and expand based on content and the `max-h` limits its growing when more text is added, you also have to tell what happens when it exceeds the limit and that's done with `overflow-y-auto`.

sometimes though, it's not quite that simple. you might be encountering issues with parent elements that have explicit heights or that are limiting the modal in unexpected ways. one time, i spent a good portion of my day trying to debug a modal that kept disappearing on mobile only to find out that the main content section of my page had `overflow: hidden;` on its css and that was preventing the modal from popping up.

here is another example where a parent container that is not the modal itself is having its overflow set to `hidden` that might cause you the modal to not scroll properly or go off screen:

```html
<div class="w-full h-screen overflow-hidden">
    <div class="relative p-4 h-full flex flex-col">
      <div class="flex-1 h-full overflow-y-auto">
        <p>main content here which is long long long and could cause modal issues.</p>
        <p>main content here which is long long long and could cause modal issues.</p>
        <p>main content here which is long long long and could cause modal issues.</p>
         <p>main content here which is long long long and could cause modal issues.</p>
         <p>main content here which is long long long and could cause modal issues.</p>
          <p>main content here which is long long long and could cause modal issues.</p>
           <p>main content here which is long long long and could cause modal issues.</p>
      </div>
         <div class="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div class="bg-white rounded-lg shadow-xl p-6 w-full max-w-md h-auto max-h-[90vh] overflow-y-auto">
              <h2 class="text-xl font-bold mb-4">this is the title of the modal</h2>
              <p class="mb-4">a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.</p>
              <p class="mb-4">a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.</p>
              <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">close</button>
            </div>
          </div>
    </div>
</div>

```

here, see how the container of the modal has `overflow-hidden`? that's what can cause issues. so when your modal is not behaving, inspect that dom tree in the browser and search for those kinds of css rules on parent components, and also if the parent container is also absolute positioned, that could also be an issue. the `h-screen` on the main container and the `h-full` on its immediate children is also an important thing to have, especially when you are using flexbox and trying to have the modal take up most of the screen.

another useful thing, is to not make a modal fill the entire screen. using `max-w-md` in the example is a great way to constraint the width so it does not grow more than a certain size in width.

and while we are talking about scrolling, sometimes the issue is that you want to also keep the modal in the center of the screen, and the way to do that with tailwind is to use `flex items-center justify-center` in the immediate parent div of the modal, as you see in the previous examples.

let's look at another common case that i've seen where a modal is embedded into a component with absolute positioning, and the modal container is not correctly inheriting the right size of its viewport:

```html
 <div class="relative">
     <div class="absolute top-0 left-0 right-0 w-full">
        <div class="flex flex-col w-full h-full">
             <p>content here</p>
             <p>content here</p>
             <p>content here</p>
             <p>content here</p>
            <div class="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
                <div class="bg-white rounded-lg shadow-xl p-6 w-full max-w-md h-auto max-h-[90vh] overflow-y-auto">
                  <!-- modal content here -->
                  <h2 class="text-xl font-bold mb-4">this is the title of the modal</h2>
                  <p class="mb-4">a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.</p>
                  <p class="mb-4">a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture. a lot of text here. i am sure you are getting the picture.</p>
                  <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">close</button>
                </div>
            </div>
        </div>
     </div>
 </div>
```

here, `absolute` positioning on the second `div` could also cause some issues, as you can see that the parent of the modal has all that `absolute` and `top-0 left-0 right-0` positioning, which makes the modal more difficult to reason about, this is yet another case where inspecting the dom can help.

in essence, the modal can go off screen because the css properties that you need to make it behave are not applied in the correct order. this often happens with complex components that have several nested divs.

i know that sometimes the solution is not as simple as adding that `overflow-y-auto` but i also feel that the majority of the time it is the root cause of the problem.

when dealing with css layout and scrolling issues, my go to resources are generally books rather than documentation. i recommend checking out "css the definitive guide" by eric meyer. it's a classic that goes really in deep into the different layout mechanisms of css. also "every layout" by andy bell and heyden pickering is amazing for the more modern approaches to layouts and it is an online book. i find it more comprehensive than the standard css documentation because it also explains why things work the way they work, not just how to use them, it makes it easier to debug complex issues.

one last thing, that i should mention is sometimes you also want the modal to scroll the whole page behind it. in that case, it is best to use something called "modal portals", it is beyond the scope of this response, but you should keep it in mind when more complex use cases are needed.

so, yeah, that's pretty much it. keep an eye on your overflow properties, check your parent containers, use max-height, and remember that browsers are not always as smart as we'd like, they need explicit instructions. oh and always try to have a clear structure of your elements, avoiding `absolute` and `relative` positioning in the most inner parts of your dom whenever possible to avoid those types of problems. debugging css is like a box of chocolates, you never know what you are going to get, unless you inspect the dom of course.
