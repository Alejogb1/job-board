---
title: "branch pricing branch io api usage?"
date: "2024-12-13"
id: "branch-pricing-branch-io-api-usage"
---

Okay so you're asking about branch pricing and using the Branch IO API right Been there done that got the t-shirt or actually more like the stackoverflow badge ha

Lets dive in I've wrestled with Branch IO's stuff for a good while now and the pricing model can be a bit of a… puzzle at first glance I remember back in like 2018 I was working on this mobile app for a local bookstore you know typical small business stuff We were using Branch for deep linking mostly trying to get users from social media ads right into specific book pages in our app It worked great initially but man the billing was a head scratcher for a while

Here's the gist Branch IO pricing usually revolves around a few key things

**Monthly Active Users MAU** this is a big one They track how many unique users interact with your Branch links each month The more people clicking your deep links the higher your MAU count goes and this is a main factor in how they price their plans Its like the more you use the more you pay pretty standard SaaS model honestly

**Link Clicks** This is sort of tied to MAU but it tracks every single click on your branch link Its not just unique users but each click they might make so if a user clicks five branch links one month it will count as 5 clicks

**Data Retention** They also charge depending on how long they need to store your historical data This depends on your needs and how much you want to keep a record of but it is something that will increase the price

**Specific Features** Branch also has tiered features so the more advanced stuff like deep linking with custom routing or advanced reporting costs extra

Now for the API yeah that's where things get interesting and where I spent the most time banging my head on my desk getting it to work

Let's say you're using the Branch SDK you've already done the basic setup you know initialization and linking stuff right? ok so lets get to the examples I have here some snippets that I actually used once for our small bookstore app

**Example 1: Creating a basic Branch Link**

```javascript
// Assuming you have the branch SDK initialized

const branch = require('branch-sdk');

const linkProperties = {
  channel: 'facebook',
  feature: 'share',
  campaign: 'summer_sale_2024',
  tags: ['book_promotion', 'sale']
};


const dataProperties = {
  "$desktop_url": "https://mybookstore.com/book/123",
    "$ios_url": "mybookstore://book/123",
    "$android_url": "mybookstore://book/123",
  book_id: '123',
  book_title: 'The Hitchhiker\'s Guide to the Galaxy'
};
branch.link({
  data: dataProperties,
  onSuccess: (url) => {
    console.log('Branch link created:', url);
    //Now this url can be used for facebook adds or anything else
  },
  onError: (err) => {
    console.error('Error creating branch link:', err);
  }
})
```

This is the very basic stuff you know creating a link programmatically This example shows how we can create a shareable link and then how to use it We are providing extra data which we will use to access the correct page in the app

**Example 2: Handling Deep Linking**

```javascript
// Assuming you have the Branch SDK initialized

branch.initSession((err, data) => {
  if (err) {
    console.error('Error init Branch session:', err);
    return;
  }


  if (data && data['+clicked_branch_link']) {
    const bookId = data.book_id;
    console.log('User came from a branch link the book id is', bookId);
   // Here we could trigger an action to load the book page
    // Example:
    //  if(bookId){ loadBookPage(bookId); }

  } else {
      //User didn't come from branch link
        console.log('User opened the app directly');
  }


});
```

Here we are listening for the deep link data after the user opens the application This shows how to use that information from the previous example and navigate the user to the correct page inside the app This will help you understand how to deep link and understand if a user has arrived from a branch link or not

**Example 3: Using Link Identity**

```javascript
const branch = require('branch-sdk');

branch.setIdentity("user123", (err, data) => {
    if(err){
        console.error('Error setting identity',err);
    }
    console.log('identity saved', data);
// At this point we can track the activity of this user, like the pages they view
})
```

Here we set a user identification this is useful if you want to understand the behaviour of a specific user and its also useful for attribution This will allow Branch to track the same user activity throughout different sessions

Ok back to the pricing I know you asked a specific question about pricing and i am not going to lie it can be a bit tricky I’ve seen lots of teams struggle to keep their costs under control with Branch This happens because it is easy to generate a lot of clicks if your marketing team is performing really well and sometimes it is hard to keep up with the amount of users specially if you are growing really fast.

Here's the thing about Branch's pricing it is not static and it can change they have different plans and tiers and sometimes it's not entirely transparent I swear once I thought I had it figured out and then BAM new billing cycle hit and the numbers were different It was like trying to debug code with inconsistent error messages really frustrating sometimes but eventually you learn how it works.

The best thing to do is to really understand their documentation and to keep track of your metrics in Branch and compare them with your usage in your analytics to understand what is really going on. Their dashboard is usually pretty good but sometimes you might need to dig deeper and use the API to gather more insights.

Here’s a tip that saved my team a lot of money in that bookstore project We used to create Branch links for everything even for internal app navigation which is not a great idea We fixed this by creating Branch links only for external sources and the problem was solved. We could then keep the cost down and we didn't need that many credits.

It is also very important to correctly tag all your links and campaigns if you dont do this you can get bad metrics and this will lead to wrong conclusions

Oh and a little inside joke i found during the process when you are dealing with api calls you really have to double check your parameters or you might end up paying a lot for clicks that you didnt mean to have. It is like if you accidentally put a negative sign in an accounting software and suddenly you own the bank. Well not really you just pay a lot of useless link clicks.

Ok enough joking around.

So where can you learn more about this stuff you might be asking? instead of recommending you random stackoverflow questions let me tell you some books or papers I found very useful when dealing with Branch

For a good high level understanding of deep linking and attribution check out "Mobile Deep Linking: Strategies and Techniques" by someone named Dave Teare if I recall correctly I know I read it sometime in 2020. It is a good introduction to the main concepts and some common issues

Also "Understanding Deep Linking: A Technical Overview" a report published by the W3C that talks in a more technical way about deep linking and its implications

And for Branch specific stuff well Branch themselves have a pretty good documentation But i would say that "Branch Documentation" is a good start it is a bit long but it is a must read if you are going to implement branch

And last if you really want to be a pro in deep linking then read the following paper its a research paper that I found super interesting "The Evolution of URL Schemes for Mobile Deep Linking" by  Ehsan Hashemi and David J. DeWitt it is a deep dive into all things URL schemes and deep linking a bit technical but a really good read for a real expert.

So yeah that's basically all I have to say now that was a long text eh? hope it was useful If you need more help just ask I'm always willing to share my experience and help fellow developers I have been in this game long enough and now I just want to help
