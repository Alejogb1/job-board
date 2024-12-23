---
title: "How can I fetch Instagram posts by hashtag and user, using the Graph API and Rails, without app review?"
date: "2024-12-23"
id: "how-can-i-fetch-instagram-posts-by-hashtag-and-user-using-the-graph-api-and-rails-without-app-review"
---

Let's talk about fetching Instagram posts by hashtag and user, specifically within the constraints of the Graph API and a Rails application, while avoiding the dreaded app review process. It's a challenge I've navigated firsthand multiple times, and I'm happy to share the specifics, gleaned from those experiences.

The core issue stems from the Graph API's inherent design. Full access to user data, including posts, generally requires a formal app review process. This review focuses on ensuring your application adheres to Facebook's platform policies and user privacy. However, there are pathways to obtain *limited* data without triggering this rigorous review process. These pathways largely revolve around using what’s sometimes referred to as ‘basic access’ permissions.

My initial foray into this area involved building a simple content aggregator for a small marketing team. We wanted to pull in public posts related to specific campaigns, and we certainly didn't want to deal with the months-long review ordeal. I quickly realized that direct access to a user's complete post history or all posts under a hashtag, without explicit user permission, was completely off the table under basic access. But there was a way, a somewhat constrained but usable approach.

First, we must accept the limitations. Basic access permissions, specifically for `instagram_basic` and `instagram_content_publish`, allow us to fetch *publicly available* data. This means:

1.  **User Data:** We can only access data related to the Instagram Business account or Instagram Creator account associated with the access token we're using. We *cannot* use this token to query data for *other* users.

2.  **Hashtag Data:** We can only access media tagged with specific hashtags where those hashtags are *associated with posts from the connected Instagram Business or Creator account*. In other words, we cannot arbitrarily grab all public posts for a given hashtag.

3.  **Limited Metadata:** The specific fields returned with basic access are reduced compared to full access. For instance, you might not have access to precise engagement metrics or all the comments.

Now, let's get practical with some Rails-centric examples. You'll need a few gems; `koala` is an excellent choice for interacting with the Graph API, and you’ll likely need a way to manage your access tokens (which I’m not demonstrating here, since it depends on your setup). I’m assuming, for the purpose of this explanation, that you already have a functioning Facebook App with an Instagram account linked to it and that you can obtain a valid, short-lived or long-lived access token. (For a deeper dive into access token handling, the official Facebook Graph API documentation and *Programming the Facebook API* by Michael L. Nutter and Ian C. Fette provide excellent guidance).

**Example 1: Fetching a User’s Recent Posts**

Here’s how we could retrieve recent posts by a particular user associated with the given access token (remember, only the user associated with the access token).

```ruby
require 'koala'

def fetch_recent_user_posts(access_token, fields = 'id,caption,media_type,media_url,permalink,timestamp')
  graph = Koala::Facebook::API.new(access_token)
  instagram_account_id = graph.get_connections('me', 'accounts').first['id']
  recent_media = graph.get_connections(instagram_account_id, 'media', fields: fields)
  return recent_media['data']
rescue Koala::Facebook::ClientError => e
  puts "Error fetching user posts: #{e.message}"
  return nil
end

# Example Usage:
access_token = 'YOUR_ACCESS_TOKEN'
posts = fetch_recent_user_posts(access_token)

if posts
  posts.each do |post|
    puts "Post ID: #{post['id']}"
    puts "Caption: #{post['caption'].nil? ? 'No caption' : post['caption']}"
    puts "Type: #{post['media_type']}"
    puts "URL: #{post['media_url']}"
    puts "Link: #{post['permalink']}"
    puts "Timestamp: #{post['timestamp']}"
    puts "---"
  end
end
```

In this snippet, we first initialize a `Koala` object using the given access token. Then, we retrieve the instagram account id that is associated with this token, and then use that to fetch the 'media' connection, which provides recent posts associated with this account. Notice how we are using a specific `fields` param to retrieve specific details, keeping only what we need to enhance efficiency. The `rescue` block is crucial for robust error handling, something I always advocate after a few early blunders where I neglected this aspect.

**Example 2: Fetching Posts With Specific Hashtags (Posted by the Connected Account)**

Now, let's tackle hashtag retrieval, bearing in mind the previously mentioned limitations:

```ruby
require 'koala'

def fetch_user_posts_with_hashtag(access_token, hashtag, fields = 'id,caption,media_type,media_url,permalink,timestamp')
  graph = Koala::Facebook::API.new(access_token)
    instagram_account_id = graph.get_connections('me', 'accounts').first['id']

  # We first need to get the recent posts from the user.
  recent_media = graph.get_connections(instagram_account_id, 'media', fields: fields)

  # Filter those posts to return only ones that have the desired hashtag.
  posts_with_hashtag = recent_media['data'].select do |post|
    post['caption']&.downcase&.include?("##{hashtag.downcase}")
  end

  return posts_with_hashtag
rescue Koala::Facebook::ClientError => e
  puts "Error fetching user posts with hashtag: #{e.message}"
  return nil
end

# Example Usage:
access_token = 'YOUR_ACCESS_TOKEN'
hashtag = 'mycampaignhashtag'

posts = fetch_user_posts_with_hashtag(access_token, hashtag)

if posts
  posts.each do |post|
     puts "Post ID: #{post['id']}"
    puts "Caption: #{post['caption'].nil? ? 'No caption' : post['caption']}"
    puts "Type: #{post['media_type']}"
    puts "URL: #{post['media_url']}"
    puts "Link: #{post['permalink']}"
    puts "Timestamp: #{post['timestamp']}"
     puts "---"
  end
end
```

The core of this example is using the previous method to fetch the users posts, and *then* performing in-code filtering based on the inclusion of a particular hashtag within the post's caption. This is the only reliable method I've found to achieve this within the constraints of basic permissions. It's essential to understand that this method will *not* fetch public posts from *other* users that include the same hashtag. Again, error handling is key and should always be included in your API interactions.

**Example 3: Fetching the Business Account Details**

This example demonstrates how to fetch basic information about the Business or Creator account linked to the access token:

```ruby
require 'koala'

def fetch_business_account_details(access_token)
    graph = Koala::Facebook::API.new(access_token)
    instagram_account_id = graph.get_connections('me', 'accounts').first['id']
    business_details = graph.get_object(instagram_account_id, fields: 'username,profile_picture_url,biography,followers_count,following_count')
    return business_details
    rescue Koala::Facebook::ClientError => e
        puts "Error fetching business account details: #{e.message}"
      return nil
end

# Example Usage:
access_token = 'YOUR_ACCESS_TOKEN'
account_details = fetch_business_account_details(access_token)

if account_details
  puts "Username: #{account_details['username']}"
  puts "Profile Picture: #{account_details['profile_picture_url']}"
    puts "Biography: #{account_details['biography']}"
  puts "Followers Count: #{account_details['followers_count']}"
    puts "Following Count: #{account_details['following_count']}"
end
```
This demonstrates fetching basic details, providing an overview of the connected business account.

In summary, while bypassing app review does limit what you can achieve with the Graph API, using `basic access` permissions still allows for gathering some data. Crucially, keep the limitations in mind: you’re primarily working with the content and metadata associated with the authenticated account, and your access to hashtagged posts will be specific to those published from that account. For more details on permissions and how they apply, I suggest consulting the official Instagram Graph API documentation and perhaps exploring some third-party educational platforms offering in-depth API training. These have been invaluable tools in my own journey. Finally, never forget error handling – it’s not just a good idea, it’s an absolute necessity when interacting with third-party APIs.
