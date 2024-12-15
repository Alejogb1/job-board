---
title: "How to Refresh Authentication Cookie when updating claims in a database?"
date: "2024-12-15"
id: "how-to-refresh-authentication-cookie-when-updating-claims-in-a-database"
---

so, you're hitting that classic problem: you’ve got user claims stored in a database, and you're updating them, but the authentication cookie the user has is still based on the old info. this is a common pitfall, and i've definitely tripped over it more than once back in my early days fiddling with web apps. let me walk you through how i usually handle it, focusing on the stuff that’s worked for me over the years, rather than some theoretical ideal.

first off, let’s be clear. authentication cookies are basically a snapshot in time. when a user logs in, you’re typically creating this cookie with a set of claims, like their user id, roles, permissions, whatever your app needs. this cookie is then sent with every request, so the server doesn't have to hit the db for every single interaction. now, when you update something in the database that impacts those claims, the cookie is out of sync. the user is essentially still running around with an old identity card.

the challenge here isn't usually about the database update itself, it’s about invalidating the old cookie and forcing the user to refresh it, ideally without making them log in from scratch. we aim for a seamless experience, not to completely kick them out randomly. there is more than one way to skin this cat (oops, sorry, old habit) lets focus on the good ones.

here's what i’ve found works well. when a user's claims change – let's say, their role gets updated or a permission is granted – you've got a few options, and the route you take mostly depend on how important the changes are in the given context. the first approach is to simply invalidate the current cookie and force the user to re-authenticate on their next request. i’ve done this in the past, and it's quite straightforward to implement.

in a typical web framework, you can clear the authentication cookie, typically by setting its expiration date to the past. when the client makes the next request, it will no longer have the cookie and that triggers an authentication flow.

```python
# Example with Flask
from flask import Flask, session, redirect, url_for

app = Flask(__name__)
app.secret_key = "your_super_secret_key"  # should come from env var in prod

@app.route('/update_user_claims', methods=['POST'])
def update_user_claims():
   # Imagine the claims are updated in the db here
    
   # clear session to invalidate the authentication cookie
   session.clear() 

   # redirect to somewhere safe maybe index
   return redirect(url_for('index'))


@app.route('/')
def index():
  if 'user_id' in session:
      return "You are logged in"
  else:
      return "Please login"


@app.route('/login')
def login():
  # Imagine the login logic is here, and it sets the cookie
    session['user_id'] = 123 # Example user id

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
```

in this example the `/update_user_claims` is the endpoint where your claim update logic happens. after the update, the session is cleared which in turn invalidates the cookie and the user is sent to `/` endpoint which in the example triggers a logout as it does not have `user_id`. this is not a smooth experience and is the reason we should explore other options.

now, a better approach involves actively refreshing the cookie after the update but before the user sees the consequences of not having the changes present. here you'll need a mechanism to re-issue the authentication cookie with the new claims. this usually means loading the updated user claims from the database after any relevant update and using them to generate a new cookie.

```python
# Example with Django
from django.shortcuts import redirect
from django.contrib.auth import login, get_user_model
from django.contrib.auth import authenticate
from django.http import HttpResponse
User = get_user_model()

def update_claims_and_refresh(request):
    # pretend that user has role update here in the database
    user = request.user

    # Fetch the latest user info from the database
    updated_user = User.objects.get(pk=user.pk)

    # Re-authenticate user to update auth cookie
    user_backend = authenticate(request, username=updated_user.username, password=user.password)

    if user_backend:
      login(request, user_backend)
      # redirect the user to the page they came from, or to a default page
      return redirect('/')
    else:
      return HttpResponse("User not found")
```

the `update_claims_and_refresh` view shows how you would re-authenticate the user. in a django or similar framework, the user is re-authenticated and a new authentication cookie is created by the framework itself. this process is seamless, they would likely never notice the re-authentication.

finally, there are cases when you don’t want to re-authenticate the user on every change, sometimes, just a specific change you don't mind the old cookie, but you want the system to refresh it once the user does a specific action. a common strategy here is to include a version number in your claims payload. when a change is made, you increment a user claims version in the database, while on your authentication system you check the claims version in the cookie, if it is older than what is in the database, you redirect them to a re-authentication endpoint which refreshes the cookie while the change is being applied. this is what i call “lazy refresh” and works when the user doesn’t necessarily need the change immediately.

```python
# Example using a lazy refresh in nodejs

const express = require('express');
const cookieParser = require('cookie-parser');
const jwt = require('jsonwebtoken');

const app = express();
app.use(cookieParser());
app.use(express.json());

const SECRET_KEY = 'your_super_secret_key'; // should come from env

const users = {
    'user1': { id: 1, name: 'User One', version: 1, roles: ['user']},
    'user2': { id: 2, name: 'User Two', version: 1, roles: ['user']},
}

// Middleware to simulate a db fetch of user and update it if needed
async function fetchUser(userId) {
  let user =  Object.values(users).find(u => u.id === Number(userId))
  if (!user) {
     return null
  }

  //simulate update user
  if (user.name === 'User One') {
      user.version = user.version + 1
      user.roles = ['admin']
  }

  return user
}

const authenticateUser = (req, res, next) => {
  const token = req.cookies.token;

  if (!token) {
    return res.status(401).send('Unauthorized. Please Login');
  }

  try {
      const decodedToken = jwt.verify(token, SECRET_KEY)
      req.user = decodedToken.user
    next()
  } catch(error) {
    res.status(401).send('Invalid token');
  }
};

const checkUserClaimsVersion = async (req, res, next) => {
  const userId = req.user.id;
  const currentClaimVersion = req.user.version
  const freshUser = await fetchUser(userId)

    if (freshUser && freshUser.version > currentClaimVersion) {
      return res.status(401).send('token outdated, refresh your token by going to /refresh');
   }

  next()
}

app.post('/login', (req, res) => {
    const user = Object.values(users).find(u => u.name === req.body.username);

    if (!user) {
        return res.status(401).send('user not found');
    }

    const token = jwt.sign({ user: { id: user.id, version: user.version, name: user.name, roles: user.roles } }, SECRET_KEY);

    res.cookie('token', token, { httpOnly: true });
    res.status(200).send('logged in');
});

app.get('/refresh', authenticateUser, async (req, res) => {
    const freshUser = await fetchUser(req.user.id)
    if (!freshUser) {
      return res.status(401).send('user not found')
    }
     const token = jwt.sign({ user: { id: freshUser.id, version: freshUser.version, name: freshUser.name, roles: freshUser.roles } }, SECRET_KEY);
    res.cookie('token', token, { httpOnly: true });
    res.status(200).send('token refreshed');
});


app.get('/me', authenticateUser,checkUserClaimsVersion ,(req, res) => {
    res.status(200).json(req.user);
});

app.get('/logout', (req, res) => {
    res.clearCookie('token');
    res.status(200).send('logged out');
});


app.listen(3000, () => console.log('Server started on port 3000'));
```
in this example the `/me` endpoint is protected by two middleware. first it is protected by the authentication middleware that checks the token and if the token has a valid signature it attaches the user object to the request. the second middleware uses the attach user from the previous middleware to check the version of the user in the token, if the version is outdated it would redirect the user to the refresh endpoint, which then triggers the user re-authentication, the `/me` endpoint then can see the new version and send a correct response. this process happens seamlessly on the background but adds extra complexity to your app so you need to evaluate if this is worth doing.

now, which approach is best really depends on your specific needs. if your claims changes are super critical and the app should not work with the old cookie, then simply invalidating the old cookie and forcing re-authentication is your safest bet, although it may affect the user experience slightly. if you want to be seamless, re-issuing the cookie after each change will do the job, but will have some performance hit, finally lazy refresh makes sense when you have lots of claim changes but it isn't important that those changes be propagated immediately.

as for diving deeper into this, i can highly recommend reading about how authentication works for the most used frameworks out there, there is an excellent section in the security chapter of the 'web application hacker's handbook' that explains the attack surface for different auth techniques. and if you are curious about how cookie management works under the hood, the mozilla developer docs have an excellent explanation. i've found those sources have been invaluable when i was going through the same issues myself.

in summary, refreshing the authentication cookie when you update claims is a pretty standard problem, it is less of a how and more of a what, think about what is more important to you, consistency or user experience, then pick your approach and code. if you feel you need more specific advice on your case, remember to add details about the specific tech stack you are working on.
