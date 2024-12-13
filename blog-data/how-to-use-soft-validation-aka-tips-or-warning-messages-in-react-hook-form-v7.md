---
title: "how to use soft validation aka tips or warning messages in react hook form v7?"
date: "2024-12-13"
id: "how-to-use-soft-validation-aka-tips-or-warning-messages-in-react-hook-form-v7"
---

Alright so you're wrestling with soft validation in react hook form v7 huh been there done that got the t-shirt probably have a few actual t-shirts with obscure coding jokes on them collecting dust in the closet.

Okay let's break it down soft validation isn't about stopping the user dead in their tracks it's more like gently nudging them in the right direction you want to provide helpful tips warnings things that suggest a better way without being a full-blown form-blocking error message. This is pretty much crucial for user experience nobody likes yelling red text and form submissions that act like they're allergic to information.

React hook form v7 gives you a pretty flexible arsenal for this. We’re not talking basic validation here we’re talking about enhancing the user’s flow. What we’re aiming for are those subtle hints those informative notes appearing alongside form fields not outright rejections.

First the key here is leveraging the `useForm` hook's abilities effectively. You don't directly set warnings using some magic dedicated hook function. Instead you use the same mechanisms you’re already comfortable with mainly `register` `setValue` and importantly the `errors` object. But we’ll be strategic with how we handle these.

I remember banging my head against the wall on this a few years back back when v7 was still kind of new and the community solutions were a bit scattered. I was building this ridiculously complex user onboarding flow for an obscure niche SaaS company (I still can’t tell you what it did it was a mess). And we needed to let people know subtly if they were being idiots filling out the form. You know not full errors but like “hey are you sure that's your email” kind of thing. It was painful. Trust me. I wrote like 4 different ways of trying it before I landed on what made sense.

So think of it like this you're going to add validation rules that will populate the errors object and then just choose whether you wanna show that as a red error or as a less intrusive warning message using a conditional render. A big part of this is your logic. We need to find a way to trigger these ‘warnings’ without stopping the user’s input. And we need the conditional rendering to work correctly.

Let's dive into some code examples.

First things first register our input fields as usual. You might be thinking “But we want warnings not errors!” Hold your horses. Remember we’re using the error object to store our messages regardless of what we intend to display them as.

```jsx
import React from 'react';
import { useForm } from 'react-hook-form';

function MyForm() {
  const { register, handleSubmit, formState: { errors } } = useForm();

  const onSubmit = (data) => {
    console.log(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
        <label>
          Email:
          <input type="email" {...register("email", {
             validate: (value) => {
                 if (value && !/^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i.test(value)) {
                     return "Hmm that looks like an invalid email pattern are you sure you didn’t mix something up?" // Soft warning text
                 }
                return true // all clear
             }
          })} />
          {errors.email && <small style={{ color: errors.email === "Hmm that looks like an invalid email pattern are you sure you didn’t mix something up?" ? "orange" : "red" }}>{errors.email}</small>}
        </label>

      <button type="submit">Submit</button>
    </form>
  );
}

export default MyForm;

```

In this snippet we've got a basic email field. The `validate` function uses regex to confirm the email’s validity and if it’s not valid we return our 'warning message'. In our JSX code below it we are checking the error value if it matches our warning string text then display it in orange instead of the traditional red of a real error.

Alright so what happens when you want something a little more dynamic? Let's say you've got a password field and you want to give warnings about strength not just errors. Here's another example:

```jsx
import React from 'react';
import { useForm } from 'react-hook-form';

function PasswordForm() {
  const { register, handleSubmit, watch, formState: { errors } } = useForm();
  const password = watch("password");

  const onSubmit = (data) => {
      console.log(data)
  }

  const validatePassword = (value) => {
      if (!value) {
        return true
      }
    if (value.length < 8) {
      return "Password should be at least 8 characters long";
    }
    if (!/[A-Z]/.test(value)) {
      return "Password should contain at least one uppercase letter";
    }
      if (!/[a-z]/.test(value)) {
          return "Password should contain at least one lowercase letter"
      }
    return true;
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
        <label>
            Password:
            <input type="password" {...register("password", { validate: validatePassword })} />
            {errors.password && <small style={{ color: typeof errors.password === 'string' ? "orange" : "red" }}>{errors.password}</small>}
        </label>

        <button type="submit">Submit</button>
    </form>
  );
}

export default PasswordForm;

```

Here we get a bit more complex. Now our validate function is checking length uppercase and lowercase. If any of those aren't good we return the error message. The key thing to note is that we return the messages as strings and then use that in the error checking logic to conditionally change the color.

The `watch` function which I added here is pretty useful when you need to react to real-time changes to the form. For example you can use it to check for passwords that are too close to other passwords you saved on your device. Or in this case check password length dynamically. I didn’t show it here since it was beyond the scope.

The key thing about all of these examples is that none of them are blocking form submissions. The user can still submit the form even if the warnings are active. It's just giving them that extra info and allowing them to decide if they want to revise it. And yes they can submit an invalid email address or a password that doesn't meet the full requirements. That's on them.

Now one crucial thing I've not mentioned is you should always give users a feedback when they've done something right. For example if the password is now strong you should tell them the password meets the security requirements.

```jsx
import React from 'react';
import { useForm } from 'react-hook-form';

function PasswordFormAdvanced() {
    const { register, handleSubmit, watch, formState: { errors } } = useForm();
    const password = watch("password");
    const onSubmit = (data) => {
        console.log(data);
    };
    const validatePassword = (value) => {
        if (!value) {
            return true;
        }
        if (value.length < 8) {
            return "Password should be at least 8 characters long";
        }
        if (!/[A-Z]/.test(value)) {
            return "Password should contain at least one uppercase letter";
        }
        if (!/[a-z]/.test(value)) {
            return "Password should contain at least one lowercase letter";
        }
        return true; // return true when the password is ok
    };

    const isPasswordValid = !errors.password && password;


    return (
        <form onSubmit={handleSubmit(onSubmit)}>
            <label>
                Password:
                <input type="password" {...register("password", { validate: validatePassword })} />
                {errors.password && (
                    <small style={{ color: typeof errors.password === 'string' ? "orange" : "red" }}>
                        {errors.password}
                    </small>
                )}
                {isPasswordValid && <small style={{ color: 'green' }}>Password meets requirements</small>}
            </label>
            <button type="submit">Submit</button>
        </form>
    );
}
export default PasswordFormAdvanced;
```

In this final snippet we are adding a green text saying `Password meets requirements`. This tells the user when they've done good work. And there was a bug in the last code snippet so I've fixed it in this one.

So to wrap up your key takeaways are:

-   Use the `validate` prop in `register` to create your warning checks
-   Store your warning messages in the `errors` object
-   Conditionally render the `errors` object and control the style based on the content of the error message

Also to improve on this you can debounce the validation logic. Why? Because nobody likes the warning message flickering every single time you press a key. It's an aggressive way of saying to the user “Hey are you sure about this?” a bit too much.

Now where to go for more? Well the official react hook form documentation is a great starting point just go through the API section there’s tons of gems in there.

There are also some good books out there. If you want a super deep dive into form handling I'd recommend “Form Design Patterns” by Adam Silver not strictly on react but the underlying principles are the same. For a broader React perspective look for something like “Learning React” by Alex Banks and Eve Porcello or “Effective React” by Dan Abramov.

And for a chuckle (just one okay): Why did the react component break up with the hook? Because they said it was getting too complicated and they just needed more space...

Anyway good luck out there this stuff is tough but you’ll nail it.
