---
title: "angular delay function usage?"
date: "2024-12-13"
id: "angular-delay-function-usage"
---

Okay so angular delay function usage right Been there done that Probably more times than I care to admit Lets dive in

The angular delay function which I assume you mean like a mechanism to introduce a pause before an action in your angular app is not something directly built in with a `delay` function like you might see in some other languages like javascript's `setTimeout` or `Promise.then` which by the way can still be used in angular but they are generally not the right choice here in my experience

Here's the thing Angular is all about change detection and reactivity and directly using things like `setTimeout` outside of Angulars change detection context can lead to unpredictable behavior or make it hard to debug Trust me I've debugged enough of those to last a lifetime back when I started out

When you need a delay in angular you should be using RxJS and Observables specifically the `delay` operator If you don't know about RxJS and Observables you should go and familiarize yourself with them like yesterday because they are the foundation of asynchronous operations in angular In my opinion learning RxJS is as important as learning javascript if you are working with angular

My first encounter with this I think was probably about 7 years ago I was building a rather complex form which involved calling several APIs to fetch data based on what the user had selected in previous dropdowns And I was doing this synchronously for each selection without delays so you can imagine the number of API calls that were happening I needed a way to add some artificial delay between API calls in the chain so I wouldn't bombard my backend server I think initially I just used settimeouts because that’s what I knew then and it worked fine until it just didn't So I learned the hard way about change detection and why using settimeouts everywhere is a nightmare

You can introduce delay after an event before an action happens for instance in your template something like this

```typescript
import { Component } from '@angular/core';
import { Subject } from 'rxjs';
import { delay, tap } from 'rxjs/operators';

@Component({
  selector: 'app-delay-example',
  template: `
    <button (click)="buttonClicked.next()">Click Me</button>
    <p *ngIf="message">{{ message }}</p>
  `,
})
export class DelayExampleComponent {
  buttonClicked = new Subject<void>();
  message = '';

  constructor() {
    this.buttonClicked
      .pipe(
        tap(() => {
           this.message = 'processing...';
         }),
        delay(2000),
        tap(() => {
          this.message = 'Button Clicked After Delay';
        })
      )
      .subscribe();
  }
}
```

Here the `buttonClicked` is a Subject its an observable that can manually emit values When the button is clicked it emits an event then the pipe starts and then the `tap` operator sets the message to 'processing' Then it hits the delay the delay operator introduces a 2 second pause after that another tap operator sets the message to a confirmation notice. This ensures that the changes are handled by angulars change detection cycle because it is all within the observable chain

Another use case where `delay` becomes super useful is when you are working with user input for example a search input where you dont want to send search requests with each keystroke you want to wait until the user stops typing for a bit before sending the request. This also reduces api calls and keeps your application performant. This is a debouncing strategy I think I worked on a search input with typeahead in another app and it was extremely slow due to this

```typescript
import { Component } from '@angular/core';
import { Subject } from 'rxjs';
import { delay, switchMap, debounceTime } from 'rxjs/operators';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-search-example',
  template: `
    <input type="text" (input)="searchTextChanged.next($event.target.value)" />
    <ul>
      <li *ngFor="let result of searchResults">{{ result }}</li>
    </ul>
  `,
})
export class SearchExampleComponent {
  searchTextChanged = new Subject<string>();
  searchResults: string[] = [];

  constructor(private http: HttpClient) {
    this.searchTextChanged
      .pipe(
         debounceTime(500),
        switchMap(searchTerm => {
           if(searchTerm){
            return this.http.get<string[]>(`https://api.example.com/search?q=${searchTerm}`);
           }
           return [];
        })
      )
      .subscribe(results => {
         this.searchResults = results || [];
      });
  }
}
```
In this case the `debounceTime` operator waits for 500 milliseconds after the user stops typing before it emits the most recent value from the input field Then the `switchMap` operator is used to cancel the previous request if a new search value is entered and then it makes a new request using the HttpClient of Angular
As a side note using switchMap is safer for http requests as it cancels previous request if a new one starts so it is an important thing to keep in mind if you have multiple requests coming from the same source

Finally another situation I dealt with was in dealing with animations For instance you need to wait a bit before an animation plays out after some event occurred you can use `delay` to make sure that animations look smoother and more natural I remember a time when I wanted a modal dialog to appear and fade in properly the delay before the transition helped a ton or else it was just a modal popping up right away

```typescript
import { Component, HostBinding } from '@angular/core';
import { Subject } from 'rxjs';
import { delay, tap } from 'rxjs/operators';

@Component({
  selector: 'app-animation-example',
  template: `
    <button (click)="showModal.next()">Show Modal</button>
     <div class="modal" [class.visible]="modalVisible">
      <div class="modal-content">
          Modal Content
      </div>
    </div>
  `,
  styleUrls: [`
    .modal{
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0,0,0,0.5);
      display:none;
      opacity: 0;
      transition: opacity 0.3s ease-in-out;
      }
    .modal.visible{
      display: block;
      opacity: 1;
    }
  `
  ]
})
export class AnimationExampleComponent {
  showModal = new Subject<void>();
  modalVisible = false;

  constructor() {
    this.showModal
      .pipe(
         tap(() => this.modalVisible = true),
        delay(100),
        tap(() => {
          console.log("Modal is now visible")
        })
      )
      .subscribe();
  }
}
```
In this example when the button is clicked the `tap` operator changes `modalVisible` to true which shows the modal. Then after a 100ms delay the other tap is called simply to showcase the delay. The css makes the opacity 1 which starts the animation after the delay

The key thing to remember about using RxJS `delay` operator with observables is that you are not freezing the execution of your application you are only delaying the emission of values through the observable stream which fits perfectly well within Angulars change detection cycle

And that’s essentially the gist of it with real-world examples I've dealt with
Now for resources I can tell you that there isn't one single book or article for the delay operator specifically but its crucial to get a solid understanding of RxJS
I would suggest reading books like "Reactive Programming with RxJS" by Sergi Mansilla this should give you the in-depth RxJS knowledge which is highly recommended for Angular development.
Also for a deep dive into all operators including delay the official RxJS documentation is a great resource
and it's always available online. (https://rxjs.dev/)
And yes thats a recommendation for an online resource but mostly a specific documentation page not an article.
And just to add a little humor to it all why do Javascript developers prefer dark mode Because light attracts bugs Haha its true!
I hope this response helped if anything you can ask again anytime
