---
title: "Does dgrid support for aria-sort attribute?"
date: "2024-12-15"
id: "does-dgrid-support-for-aria-sort-attribute"
---

so, you're asking about dgrid and its support for the aria-sort attribute, huh? i've been there, wrestled with dgrid a fair bit myself, so let me share what i've learned.

the short answer is: yes, dgrid *can* work with aria-sort, but it doesn't do it automatically out of the box. it's not like you flip a switch and suddenly you have accessible sorting indicators. you'll need to handle it yourself. which, to be honest, makes sense given dgrid's flexibility, but it also means some extra coding on your end.

back in the day, circa 2014, i was working on this massive internal dashboard application for a logistics company. they had tons of data, all presented in grids, and, of course, accessibility was a major requirement. at the time dgrid was like the "it" grid library. i remember being pretty green when it came to web accessibility specifically aria attributes. i had this naive assumption that frameworks would "just handle it", which as we both know, it's usually not the case.

anyway, the chief compliance officer who was a very stern person was very adamant about meeting accessibility guidelines for compliance purposes and i had a lot of "fun" debugging it all at the time. so when i got to the dgrid part i had to deal with aria-sort specifically. so i started looking for the attribute in the source code for dgrid and i found nothing. then i started digging into the documentation and i only found bits and pieces. i knew i was on my own so i resorted to what i do best: read the code, experiment, and stackoverflow for days.

my approach back then and the one i'd recommend today is to manually add and update the aria-sort attribute on the column headers when the user initiates a sort. you will have to intercept the sorting events, figure out what column is sorted by, what direction and update that specific header element. it might seem cumbersome but it grants you fine-grained control which is useful for custom sorting or styling of the column header.

here’s a conceptual example of how i'd approach this using javascript. i’m assuming you’re using a standard dgrid setup with a simple dom-based grid:

```javascript
require([
    'dgrid/Grid',
    'dgrid/extensions/ColumnHider',
    'dojo/on',
    'dojo/dom-attr',
    'dojo/query',
    'dojo/domReady!'
], function(Grid, ColumnHider, on, domAttr, query){
    var grid = new Grid({
        columns: {
            id: 'ID',
            name: 'Name',
            email: 'Email'
        },
    }, 'myGrid');

    var data = [
        { id: 1, name: 'Alice', email: 'alice@example.com' },
        { id: 2, name: 'Bob', email: 'bob@example.com' },
        { id: 3, name: 'Charlie', email: 'charlie@example.com' }
    ];

    grid.renderArray(data);

    // listen to dgrid sort event
    on(grid, 'dgrid-sort', function(event) {
      var sortedColumn = event.sort[0].attribute;
      var sortDirection = event.sort[0].descending ? 'descending' : 'ascending';

      //reset aria sort for all headers
      query('.dgrid-header .dgrid-cell').forEach(function(node){
        domAttr.remove(node,'aria-sort');
      });
      //get the sorted column header
      var headerNode = query('.dgrid-header .dgrid-cell[data-column-id="'+sortedColumn+'"]')[0];
      if (headerNode) {
          domAttr.set(headerNode, 'aria-sort', sortDirection);
      }
   });
});
```

in the code above i'm attaching an event listener on `dgrid-sort` event, then i reset the `aria-sort` attributes on all headers and set the `aria-sort` attribute to the header that was clicked. keep in mind `data-column-id` depends on how you configure your columns. this simple snippet gets the job done.

a common pitfall that i encountered early on was relying too much on dgrid's internal state, which is not always stable across versions. so the example above avoids trying to guess from internal variables. instead, the `dgrid-sort` event gives the information needed to set the attribute directly.

now for a real-world case, you would probably have a slightly different markup for your headers, possibly with custom classes or elements. that’s totally fine. just adjust the query selectors and the attribute setting to match your particular setup. something like this:

```javascript
require([
    'dgrid/Grid',
    'dgrid/extensions/ColumnHider',
    'dojo/on',
    'dojo/dom-attr',
    'dojo/query',
    'dojo/domReady!'
], function(Grid, ColumnHider, on, domAttr, query){
    var grid = new Grid({
        columns: {
            id: { label: 'ID', sortable:true, renderHeaderCell: function(node){node.innerHTML = '<div>ID</div>'}},
            name: { label: 'Name', sortable:true, renderHeaderCell: function(node){node.innerHTML = '<div>Name</div>'}},
            email: {label: 'Email', sortable: true, renderHeaderCell: function(node){node.innerHTML = '<div>Email</div>'}}
        },
    }, 'myGrid');

    var data = [
        { id: 1, name: 'Alice', email: 'alice@example.com' },
        { id: 2, name: 'Bob', email: 'bob@example.com' },
        { id: 3, name: 'Charlie', email: 'charlie@example.com' }
    ];

    grid.renderArray(data);

    on(grid, 'dgrid-sort', function(event) {
        var sortedColumn = event.sort[0].attribute;
        var sortDirection = event.sort[0].descending ? 'descending' : 'ascending';

      //reset aria sort for all headers
      query('.dgrid-header .dgrid-cell > div').forEach(function(node){
            domAttr.remove(node,'aria-sort');
      });
      //get the sorted column header
      var headerNode = query('.dgrid-header .dgrid-cell[data-column-id="'+sortedColumn+'"] > div')[0];
      if (headerNode) {
        domAttr.set(headerNode, 'aria-sort', sortDirection);
      }
   });
});

```

this example shows the use of a `renderHeaderCell` function to add an extra `div` inside the header to handle markup. the event listener is almost the same but now it's targeting the child `div`. that's how custom markup will affect your code, but with the same pattern the solution remains consistent.

there was also a situation when we decided to have sorting indicator images inside the headers. so we had to update the attribute of those images which of course required some tweaks on how we query the DOM and update the elements. like this:

```javascript
require([
    'dgrid/Grid',
    'dgrid/extensions/ColumnHider',
    'dojo/on',
    'dojo/dom-attr',
    'dojo/query',
    'dojo/domReady!'
], function(Grid, ColumnHider, on, domAttr, query){
    var grid = new Grid({
        columns: {
            id: { label: 'ID', sortable:true, renderHeaderCell: function(node){node.innerHTML = '<div>ID<img src="sort.png" /></div>'}},
            name: { label: 'Name', sortable:true, renderHeaderCell: function(node){node.innerHTML = '<div>Name<img src="sort.png" /></div>'}},
            email: {label: 'Email', sortable: true, renderHeaderCell: function(node){node.innerHTML = '<div>Email<img src="sort.png" /></div>'}}
        },
    }, 'myGrid');

    var data = [
        { id: 1, name: 'Alice', email: 'alice@example.com' },
        { id: 2, name: 'Bob', email: 'bob@example.com' },
        { id: 3, name: 'Charlie', email: 'charlie@example.com' }
    ];

    grid.renderArray(data);

    on(grid, 'dgrid-sort', function(event) {
        var sortedColumn = event.sort[0].attribute;
        var sortDirection = event.sort[0].descending ? 'descending' : 'ascending';

        //reset aria sort for all headers
        query('.dgrid-header .dgrid-cell > div > img').forEach(function(node){
            domAttr.remove(node.parentNode,'aria-sort');
        });

        //get the sorted column header
        var headerNode = query('.dgrid-header .dgrid-cell[data-column-id="'+sortedColumn+'"] > div')[0];

        if (headerNode) {
          domAttr.set(headerNode,'aria-sort', sortDirection);
        }
    });
});
```
now i’m targeting the parent element to update the aria-sort.

in summary, dgrid doesn't magically handle aria-sort. you're going to have to write some code to tie the sort event to attribute updates. and yes, you will need to account for the markup you use, but it's doable and gives you complete control. it's a bit like the old saying, "give a programmer a fish, they'll eat for a day, teach them to query the dom and they will be able to sort forever".

as for resources, i'd suggest looking into the w3c aria authoring practices guide. it's the definitive source for how to implement aria properly. the *accessibility for everyone* book is also an excellent companion for understanding and implementing accessible interfaces in general. and of course the dojo toolkit documentation will be invaluable for diving deep into dgrid internals if you feel the need, although the examples here should get you started without too much struggle.

i hope this helps. let me know if you get stuck.
