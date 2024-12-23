---
title: "Why isn't the Turbo Rails back button reloading the page?"
date: "2024-12-23"
id: "why-isnt-the-turbo-rails-back-button-reloading-the-page"
---

Let’s dive straight into this. I’ve been down this rabbit hole more times than I care to recall, often in projects where I thought I had everything covered. The seemingly simple back button and its interaction with Turbo in Rails can be deceptive. The core issue isn't that the back button doesn't *work*, but rather, that it often doesn't produce the *expected* result—a full page reload. Instead, it might feel like nothing happens or, worse, it might lead to inconsistencies. This arises primarily from how Turbo manages page navigation and caching.

The problem isn’t with the back button itself. Browsers have a robust history mechanism. The crux of the matter is that Turbo, by default, intercepts link clicks and form submissions, turning them into *morphing* updates of the current page rather than traditional full page requests. It accomplishes this through a combination of techniques, such as sending `turbo-stream` responses, morphing changes via Javascript, and crucially, utilizing the browser's history api. When a navigation is made within a Turbo application, Turbo captures that action and manipulates the history api and the dom. This is what gives the user the snappy experience that Turbo is designed for but which can also produce unexpected behaviors with the back button.

Turbo caches page states in its history api entry. These cached pages are typically snapshots of the dom at the time of navigation. When you use the back button, if a previously visited page is found in the history and Turbo recognizes that page as handled by it, Turbo attempts to display the cached content, which prevents a full server request. In many cases, this is great. It’s fast and reduces server load. However, there are circumstances where we *require* the server to be re-contacted, or where cached content might be stale and problematic.

I’ve seen this manifest in a few common scenarios. The most prevalent is when page content is dynamic and tied to backend logic, user authentication or session data that updates between the time a page was initially rendered and the time you press back. The cached representation no longer reflects the current state and can lead to stale data display or unexpected application behaviour.

Let's illustrate this with some examples.

**Example 1: Stale User Data**

Imagine an application that displays a user's profile. The profile has a dynamically updated “last login” timestamp. The sequence is as follows:

1.  User navigates to their profile page. The server renders the page, including the last login timestamp, and turbo caches this state.
2.  The user logs out, and then logs in again. The server now updates their last login time in the database.
3.  The user navigates to their profile page, this time with the updated timestamp, which is then cached in turbo.
4.  The user clicks the back button. Because Turbo uses its cached page, the user will see the *old*, incorrect timestamp.

This is a scenario where the cached snapshot of the dom is incorrect.

```javascript
// Example showing how a user timestamp could be rendered. This would typically be embedded in server-rendered HTML
// but this gives us an idea of the content that is cached by turbo and becomes stale.
function renderUserProfile(user){
    const lastLoginTime = new Date(user.lastLogin).toLocaleString();
    return `
        <div>
        <h1>User Profile</h1>
        <p>Last Login: ${lastLoginTime}</p>
        </div>
    `;
}

// First render
let user1 = { lastLogin: new Date().getTime() - (60*60*1000) } //an hour ago
document.getElementById('user-profile').innerHTML = renderUserProfile(user1)

// Some time later ... user re-authenticates and we render again
setTimeout(() => {
    let user2 = { lastLogin: new Date().getTime() } //now
    document.getElementById('user-profile').innerHTML = renderUserProfile(user2)
}, 5000) // Simulate delay as if server call
```

In this simplified example, the user sees outdated info due to Turbo's cache.

**Example 2: Form Submission Issues**

Consider a page where a user edits a form. If the back button is pressed after editing but *before* submitting, the cached version will reflect the page with the unsaved edits if a turbo cache hit occurs. When the user resubmits the form it could trigger unexpected server-side errors because the form state might not accurately reflect what's in the browser.

```html
<form id="edit-form">
    <label for="name">Name:</label>
    <input type="text" id="name" name="name" value="Initial Value">
    <button type="submit">Submit</button>
</form>

<script>
    document.getElementById('edit-form').addEventListener('submit', (event) => {
        event.preventDefault();
        const nameInput = document.getElementById('name');
        console.log("Form Submitted:", nameInput.value);

        //Simulating a server call but no server communication is actually happening here
       setTimeout(() => {
            nameInput.value = "updated value";
           console.log('Form Updated Successfully on server')
            }, 1000)
     });
 </script>

```

In the above, the user changes the name field in the form. If they press back and then re-submit, depending on if the turbo cache is engaged, the form they submit to the server could contain a cached value and not what they had intended to submit

**Example 3: State-Dependent Content**

I worked on an internal dashboard project where the state of a data table had filtering and sorting applied. The filters and sorting were all done on the server. Navigating away and then back meant that the cached dom would often show the table *without* the filters or sorting applied. This was incredibly confusing and frustrating for our users. We expected that the page would reload the state of the filters and sorting but instead the cached dom would be displayed.

```javascript
// Example using a server-returned data list
function renderDataList(data) {
    let listItems = data.map(item => `<li>${item}</li>`).join('');
    return `<ul>${listItems}</ul>`;
}

let initialData = ["Item 1", "Item 2", "Item 3"];
document.getElementById('data-list').innerHTML = renderDataList(initialData);

// Simulating server-side filtering with new data
function applyFilter() {
  setTimeout(() => {
    const filteredData = ["Filtered Item 1", "Filtered Item 2"]
     document.getElementById('data-list').innerHTML = renderDataList(filteredData);

  }, 2000)

}
applyFilter()
```

In this case, we would like the back button to display the initial data set (Item 1, 2, 3) and not the cached, filtered data set.

So, what can we do? Several effective strategies can help to ensure proper back-button behavior in Turbo applications:

1.  **Explicitly Disable Caching:** The easiest solution in cases like those shown above is to disable Turbo caching for specific pages using the `<meta name="turbo-cache-control" content="no-cache">` meta tag in the head of your html. I've had situations where doing this for specific routes or views fixes the problem, but it should be used sparingly due to the performance penalty. This is a good approach if you have a view or route that depends on a highly variable state. It’s a quick fix, but less efficient as it forces a server request on every visit.

2.  **Server-Side Redirects:** In some cases, like after form submissions, redirecting the user to a new URL instead of just morphing the page can prevent stale data. This ensures a full reload and can clear the cache related to the previous state.

3.  **Forcing Full Loads:** You can use javascript to intercept back button events and force a full page reload. Be cautious doing this as it can conflict with Turbo and also removes the user benefit of fast page loads but it is effective. A common approach is to intercept the `turbo:before-visit` event.

```javascript
document.addEventListener('turbo:before-visit', (event) => {
    if (event.detail.action === 'restore') {
      event.preventDefault()
      window.location.reload();
    }
  });
```

4.  **Leverage the `turbo:visit` Event:** Use the `turbo:visit` event with the `restore` action. Within this event listener you can conditionally decide if to force a full reload. This gives you more control and flexibility over when to invalidate the turbo cache.

5.  **`data-turbo-permanent` elements:** Sometimes, we want to preserve certain parts of our application. In this case, we can mark html elements with the attribute `data-turbo-permanent` and they will be persisted over turbo navigation. This can be important to save transient state, but keep in mind this isn’t a replacement for caching and it doesn’t solve the problem of stale data if that state is tied to the backend.

For a deeper understanding, I'd recommend reviewing the official Turbo documentation, which covers navigation and caching behavior extensively. "Hotwire Components" by Joe Masilotti offers a practical, hands-on explanation of these concepts as well. "Programming Phoenix LiveView" by Bruce A. Tate and Sophie DeBenedetto, although focused on LiveView, provides useful background on component state management that relates to these issues. Additionally, articles on the browser’s history API from the Mozilla Developer Network (MDN) will enhance your understanding of this aspect of web development and how it relates to Turbo.
In short, the back button problem isn't really a back button problem; it's a result of Turbo's efficient caching strategy interacting with dynamic application state. The best solution involves a careful balancing of performance considerations and the specifics of the application logic. With thoughtful planning and a good understanding of how Turbo works, these issues are manageable.
