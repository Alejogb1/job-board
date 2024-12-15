---
title: "How can I Append error message to target '.not.empty'?"
date: "2024-12-15"
id: "how-can-i-append-error-message-to-target-notempty"
---

alright, so you're looking to append an error message specifically to elements with the class `.not.empty`. been there, done that, got the t-shirt – and a few debugging scars to show for it. i remember back in '08, i was building this real-time form validation system using mostly vanilla javascript (pre-jquery days, can you imagine?), and i had this exact situation. i needed to visually flag input fields that were not filled out.

it turns out, targeting elements with a specific class and then appending something dynamically isn't all that tricky once you understand the basics of dom manipulation. let me break it down for you, focusing on how i'd tackle it today with javascript, and throw in some examples to get us on the same page.

first off, we need to select all the elements that have the class `.not.empty`. the queryselectorall() method is your friend here. it returns a nodelist, which is kind of like an array, but it's not a proper array. more on that later. once we have them, we loop through them, and on each, append the error message.

here's the most straightforward way to do it:

```javascript
const notEmptyElements = document.querySelectorAll('.not.empty');

notEmptyElements.forEach(element => {
    const errorMessage = document.createElement('span');
    errorMessage.textContent = 'this field cannot be empty';
    errorMessage.classList.add('error-message'); // optional styling
    element.parentNode.insertBefore(errorMessage, element.nextSibling);
});
```

in this snippet:

1.  `document.querySelectorAll('.not.empty')` grabs all the elements with the class `.not.empty`.

2.  `notEmptyElements.forEach(...)` iterates through each element in the list.

3.  `document.createElement('span')` creates a new `<span>` element. this will hold our error message. i used a `span` here since it is an inline element. depending on your needs, a `div` might work better too, or a `p` if you need to add paragraphs, and so on.

4.  `errorMessage.textContent = 'this field cannot be empty'` sets the text content of the span. of course, you can replace this with the dynamic error message if that is what you are going for. maybe a variable holding this text that updates each time, or more complex error message objects.

5.  `errorMessage.classList.add('error-message')` adds a class name for styling purposes (optional). this allows you to style these error messages with css. i often use `error-message` or `invalid-feedback`, but feel free to choose something that fits your project's conventions, if you have any.

6.  `element.parentNode.insertBefore(errorMessage, element.nextSibling);` this is the important part. it inserts our newly created error message `span` after the element we are checking, that is the `.not.empty`. it adds it as a sibling to the element. the `insertBefore` method is used with the `element.nextSibling` to place it just after the current element.

now, that's pretty basic, but often, we want to do more than just slap error messages on the dom. for example, we might want to remove error messages if the input field becomes valid. this is where we start to get into something slightly more complex. consider this:

```javascript
function validateNotEmpty(element) {
    const hasValue = element.value && element.value.trim() !== '';
    const existingError = element.parentNode.querySelector('.error-message');

    if (!hasValue) {
        if (!existingError) {
            const errorMessage = document.createElement('span');
            errorMessage.textContent = 'this field cannot be empty';
            errorMessage.classList.add('error-message');
            element.parentNode.insertBefore(errorMessage, element.nextSibling);
        }
    } else {
        if (existingError) {
            existingError.remove();
        }
    }
}

const notEmptyInputs = document.querySelectorAll('.not.empty');

notEmptyInputs.forEach(input => {
    input.addEventListener('blur', () => validateNotEmpty(input));
    input.addEventListener('input', () => validateNotEmpty(input));
});
```

what is different here:

1.  we create a reusable function `validateNotEmpty()` that encapsulates all the error logic.

2.  we are using `element.value` instead of `element.textContent` assuming we are working with input field elements.

3.  the `element.value && element.value.trim() !== ''` is a check if the input actually has some value. `trim` removes spaces, which means we are not passing the check just by adding space to the input.

4.  `element.parentNode.querySelector('.error-message')` it checks if an error message already exists for the current element. so that we do not stack multiple error messages to the dom.

5.  we are using `element.addEventListener('blur', ...)` and `element.addEventListener('input', ...)` which add event listeners to the target elements to trigger our logic only when those events happen. so it will validate once the user leaves the focus of the element or anytime the input is changed.

6.  `existingError.remove();` removes the error message when a valid value is inserted.

this example gives you a bit more control, and it’s generally what i see most people do when implementing form validation. now, about the nodelist versus an array thing – you can use `array.from(nodelist)` or the spread operator `[...nodelist]` to convert it into a real array if you need array methods on it, but in most cases, the forEach works just fine.

let me show you one last example, this time, let's assume your `.not.empty` elements aren't just inputs, maybe they are a div and you just want to check if they are empty, like if they contain an image or any content for example, in that case, we will need to check using a different technique, let me show you:

```javascript
function validateNotEmptyContent(element) {
    const hasContent = element.innerHTML.trim() !== '';
    const existingError = element.parentNode.querySelector('.error-message');

    if (!hasContent) {
        if (!existingError) {
            const errorMessage = document.createElement('span');
            errorMessage.textContent = 'this element should have content';
            errorMessage.classList.add('error-message');
            element.parentNode.insertBefore(errorMessage, element.nextSibling);
        }
    } else {
        if (existingError) {
            existingError.remove();
        }
    }
}

const notEmptyElementsContent = document.querySelectorAll('.not.empty');

notEmptyElementsContent.forEach(element => {
    // i'm not adding the input or blur events here because that does not make sense with content containers.
    // you could implement an observer to check if contents are added or removed if needed.
    validateNotEmptyContent(element); // calling it initially for each element on the load.

});
```

some things change here:

1.  the `validateNotEmptyContent` function changes the validation mechanism to check `innerHTML` of an element if it is empty instead of checking `value`.

2.  the event listeners are no longer needed, as input fields, usually are the ones with `value`, other types of elements often do not emit those events.

3.  the `validateNotEmptyContent` is called initially for each element, to validate if the element had content when the page is loaded for the first time.

now, a few things to keep in mind:

*   always sanitize your inputs. never trust data coming from the user (i saw some weird stuff back when i was working on that validation project) if your error messages come from user data, you want to prevent cross site scripting (xss), using textContent instead of innerhtml is often a good solution here, as we did with our examples.

*   for more complex validations, consider using a library like vee-validate, react-hook-form (if you're into react), or similar. i've spent a bunch of time building my own validation engine from scratch, and although it was a great learning experience, i do not always recommend that, often a library or framework will do a better job faster and with fewer headaches.

*   you can, of course, modify the example to target different elements or change the way that validation behaves, the core logic should still be there.

for learning more, i'd recommend reading the mdn documentation on the dom, and diving into javascript books like "eloquent javascript" or "you don't know js" if you really want to understand the nitty-gritty details. they helped me immensely over the years. also the "javascript patterns" book by stoyan stefanov was also very helpful in the past. they cover more complex examples than this, and they would provide you with more powerful tools for your everyday tasks.

now, back to coding! feel free to ask further, it's quite fun to reminisce of past coding experiences. hope this helps, have a good day.
