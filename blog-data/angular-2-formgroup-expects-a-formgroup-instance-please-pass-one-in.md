---
title: "angular 2 formgroup expects a formgroup instance please pass one in?"
date: "2024-12-13"
id: "angular-2-formgroup-expects-a-formgroup-instance-please-pass-one-in"
---

 so you're slamming your head against the wall with this "angular 2 formgroup expects a formgroup instance please pass one in" thing yeah I've been there buddy trust me I know the feeling it’s like Angular yelling at you in a language you thought you spoke fluently

Been battling this particular beast since way back in the early Angular days think Angular 2 right after the whole alpha/beta craziness I remember sweating bullets over this back when forms were like uncharted territory especially for the new devs on my team We had this client project a super complex data entry application for a logistics company and yeah it was a beautiful mess when we started I had several rookies on my team so debugging was part of the everyday routine at that time

The core issue here as you’ve probably figured out isn’t some magical Angular bug it’s just that Angular's form system is super picky you can’t just throw anything at it expecting a happy result When it says it wants a `FormGroup` instance it *really* wants a `FormGroup` instance not an object that sort of *looks* like one not a string not a number not your hopes and dreams but the actual honest to goodness `FormGroup` class instance the one you get from using `new FormGroup()` or building one up with `FormBuilder` if you prefer

I suspect that the issue is stemming from the fact that your template is trying to bind to something that is not a FormGroup instance It's a classic case of a mismatch between what your component is providing and what the template is expecting usually this happens because somewhere along the line an object was mistaken for a Formgroup so you probably need to debug starting from there

Let me give you some examples of how you'd usually screw this up and how you fix this assuming your component class where you are using the form control has a name of `MyFormComponent` I will call it like this in the following code snippets

First the classic "I just made up an object" scenario:

```typescript
import { Component } from '@angular/core';
import { FormGroup, FormControl } from '@angular/forms';

@Component({
  selector: 'app-my-form',
  templateUrl: './my-form.component.html',
  styleUrls: ['./my-form.component.css']
})
export class MyFormComponent {
  myForm: any = { // <----  This is wrong! It is not a FormGroup instance
    myControl: 'initial value'
  };

  constructor(){

  }
}

```

And in the HTML template you are trying something along these lines :

```html
<form [formGroup]="myForm">
  <input type="text" formControlName="myControl">
</form>
```

And bang Angular throws that annoying error at you  `formGroup` is expecting a `FormGroup` instance instead of just a random object which in the above case it is actually just a JSON object

The correct way to do this the proper way where you actually create the real formGroup instance is like so:

```typescript
import { Component, OnInit } from '@angular/core';
import { FormGroup, FormControl, FormBuilder } from '@angular/forms';

@Component({
  selector: 'app-my-form',
  templateUrl: './my-form.component.html',
  styleUrls: ['./my-form.component.css']
})
export class MyFormComponent implements OnInit {
  myForm: FormGroup; // <---- Type is now a FormGroup

  constructor(private formBuilder: FormBuilder){

  }
  ngOnInit(): void {
    this.myForm = this.formBuilder.group({  // <--- Correct way of instantiating form group
      myControl: ['initial value'],
    });
  }
}
```
And the html remains the same

```html
<form [formGroup]="myForm">
  <input type="text" formControlName="myControl">
</form>
```

See the difference? You need to actually instantiate a `FormGroup` not just create a simple object that looks like one We are now using the `formBuilder` service injected in the constructor and creating a `FormGroup` using the `group()` method

Another common mistake is trying to be clever with a getter. I mean we have all been there where we try to over complicate things just for fun right?

```typescript
import { Component, OnInit } from '@angular/core';
import { FormGroup, FormControl, FormBuilder } from '@angular/forms';

@Component({
  selector: 'app-my-form',
  templateUrl: './my-form.component.html',
  styleUrls: ['./my-form.component.css']
})
export class MyFormComponent implements OnInit {

  constructor(private formBuilder: FormBuilder){

  }
  get myForm() : FormGroup{
    return this.formBuilder.group({  //  <--- NOOOO dont do this you will create a new form on every re-render
      myControl: ['initial value'],
    });
  }

  ngOnInit(): void {

  }
}
```
Again if you are trying to bind using the same code from above

```html
<form [formGroup]="myForm">
  <input type="text" formControlName="myControl">
</form>
```

This is also going to fail because every single time the getter is invoked by angular to re-render the component the formGroup is going to be different because it will have created a new instance And that confuses angular. Think of it like trying to find your keys and somebody keeps changing them in each frame that happens on a movie your screen will re-render

Instead you want to declare the FormGroup in your component class and instantiate it inside the `ngOnInit` method or constructor and this is the way angular intends you to do this:

```typescript
import { Component, OnInit } from '@angular/core';
import { FormGroup, FormControl, FormBuilder } from '@angular/forms';

@Component({
  selector: 'app-my-form',
  templateUrl: './my-form.component.html',
  styleUrls: ['./my-form.component.css']
})
export class MyFormComponent implements OnInit {
  myForm: FormGroup;
  constructor(private formBuilder: FormBuilder){

  }

  ngOnInit(): void {
    this.myForm = this.formBuilder.group({  // <--- This is how you initialize the form once
      myControl: ['initial value'],
    });
  }
}
```

And again you would be able to use the html template

```html
<form [formGroup]="myForm">
  <input type="text" formControlName="myControl">
</form>
```
And this one would work

So yeah those are the most common scenarios I've seen that cause this specific problem. It's all about understanding the Angular form system is a bit pedantic it does exactly what it's told and nothing more so always triple check that your `formGroup` template binding is pointing to the correct `FormGroup` instance. I have wasted entire days on that one alone so you are not alone and don't be hard on yourself this is a right of passage every angular dev has to go through

For resources to learn more in depth I'd highly recommend diving into "Reactive Programming with Angular" by Fernando Herrera for a proper deep dive into how Angular's reactive forms actually work and if you want something more on basic forms concepts the official Angular documentation is a good place to start don't disregard it I know it seems obvious but many people skip it for some reason it's actually not that bad now but you should be prepared to read a lot and understand not just blindly copy paste Also "Angular Development with TypeScript" by Yakov Fain is a good book but probably more than what you need for this specific problem but it has some good info about forms anyway So those are the places where I’d recommend you go if you are into understanding the subject more and they will provide the correct and most adequate information

And one thing that I learned from my many years in software is that you are not alone in this fight I’ve seen a guy try to use a Map as a FormGroup instance once so you are doing great just keep at it you will be fixing this error like it is second nature to you one day it might be not today but eventually you will

Remember to always check the types and remember to read the full error Angular gives you not just the first lines and you will be alright. Good luck with your project!
