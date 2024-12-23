---
title: "Why is Rails treating POST requests as GET requests when using a React frontend?"
date: "2024-12-23"
id: "why-is-rails-treating-post-requests-as-get-requests-when-using-a-react-frontend"
---

,  I've seen this particular head-scratcher surface quite a few times, usually with a developer new to integrating a React frontend with a Rails backend. The scenario you’ve described – Rails interpreting POST requests from a React application as GET requests – is almost always indicative of a problem with how the data is being submitted, particularly concerning the content-type header. Let me walk you through what’s likely happening, based on my experience fixing this exact issue on a previous project where we were migrating a legacy system to a more modern stack.

It boils down to this: Rails, by default, expects POST data to be submitted with a specific encoding— typically `application/x-www-form-urlencoded`. If React (or more accurately, the browser, instructed by your React code) is sending data with a different content type, such as `application/json`, and without the correct Rails CSRF token, Rails might, and often does, treat it as a GET request because it doesn't recognize the incoming request format properly and fails CSRF checks. This happens behind the scenes due to Rails not being able to parse the POST data using its usual methods for url-encoded data, causing it to default to GET.

The crucial aspect here is the content-type header in your HTTP request. When your React application makes a POST request, it's imperative that this header is set correctly to `application/json` if your request body is json, or `application/x-www-form-urlencoded` if you’re using url encoded params. And, if you are sending data from a form, you must include the CSRF token that rails expects. Let me illustrate this with three code examples covering different but related scenarios.

**Example 1: Incorrect Content-Type and Missing CSRF Token**

Suppose you have the following React component intending to create a user:

```jsx
import React, { useState } from 'react';

function CreateUserForm() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await fetch('/users', {
        method: 'POST',
        body: JSON.stringify({ name, email }),
      });
      if (response.ok) {
        console.log('User created successfully');
      } else {
        console.error('Error creating user');
      }
    } catch (error) {
      console.error('Network error', error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="text" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
      <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
      <button type="submit">Create User</button>
    </form>
  );
}
export default CreateUserForm;
```

In this example, there's no content-type header explicitly set, so the browser might default to sending the data in `text/plain` , and the CSRF token is missing too. Rails will likely not recognize this POST request, will fail the CSRF check, and potentially treat it as a GET. Consequently, your controller will likely raise an error or, in some cases, perform operations as if it had received a GET request, resulting in unexpected behavior.

**Example 2: Correct Content-Type Header (application/json) with CSRF Token**

Here's how you'd fix the previous example to ensure Rails handles the POST request correctly:

```jsx
import React, { useState } from 'react';
import { getCSRFToken } from '../utils/csrf'; // assuming csrf token extraction utility.

function CreateUserForm() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const csrfToken = getCSRFToken();
      const response = await fetch('/users', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': csrfToken,
        },
        body: JSON.stringify({ name, email }),
      });
       if (response.ok) {
        console.log('User created successfully');
      } else {
        console.error('Error creating user');
      }
    } catch (error) {
      console.error('Network error', error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="text" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
      <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
      <button type="submit">Create User</button>
    </form>
  );
}

export default CreateUserForm;
```

```javascript
// example of a utils/csrf.js
export function getCSRFToken(){
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;
  return csrfToken || '';
}
```

Here, we explicitly set the `Content-Type` to `application/json`, matching the format of the data we’re sending, and included the csrf token obtained from the meta tag. The `X-CSRF-Token` is the header that rails uses. Note, you must embed the csrf tag on your rails application layouts or on a rails template rendered within react application. Rails will now correctly interpret the data as a POST request. Note, depending on your API configuration, you may also need to configure your rails app to handle json data.

**Example 3: Using Url Encoded form data**

If for some reason you need to submit form data encoded as url parameters rather than json, you can do that as follows:

```jsx
import React, { useState } from 'react';
import { getCSRFToken } from '../utils/csrf'; // assuming csrf token extraction utility.
import { URLSearchParams } from 'url';

function CreateUserForm() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const csrfToken = getCSRFToken();
      const params = new URLSearchParams();
      params.append('name', name);
      params.append('email', email);

      const response = await fetch('/users', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRF-Token': csrfToken,
        },
        body: params.toString(),
      });
       if (response.ok) {
        console.log('User created successfully');
      } else {
        console.error('Error creating user');
      }
    } catch (error) {
      console.error('Network error', error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="text" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
      <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
      <button type="submit">Create User</button>
    </form>
  );
}

export default CreateUserForm;
```

Here, we use `URLSearchParams` to convert the data to url encoded parameters. Note that because the `Content-Type` is set to `application/x-www-form-urlencoded`, Rails will be able to parse the request correctly, and, crucially, the CSRF token is passed in a header.

To further solidify your understanding, I’d recommend delving into a few key resources: the official Rails Guides documentation, specifically the sections on Action Controller and form processing, and the Fetch API documentation on MDN, to understand how the browser sends requests. The "HTTP: The Definitive Guide" by David Gourley and Brian Totty is another excellent resource for in-depth knowledge on HTTP requests and headers. Furthermore, "Web Application Architecture" by Leon Shklar and Richard Rosen is a useful reference for architecting web apps. You'll find that while initially troublesome, this issue is usually rooted in a fundamental misunderstanding of how HTTP requests are processed by the browser and expected by the backend, and once you’ve grasped the basic mechanics, you’ll be better equipped to handle similar challenges. It is a fairly common issue, and you aren't alone in having faced it!
