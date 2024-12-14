---
title: "Why am I getting an Error using searchkick (ElasticSearch) on Rails (macOS 12.4) ETHON: Libcurl initialized objc[36397]: +[NSNumber initialize]?"
date: "2024-12-14"
id: "why-am-i-getting-an-error-using-searchkick-elasticsearch-on-rails-macos-124-ethon-libcurl-initialized-objc36397-nsnumber-initialize"
---

alright, let’s tackle this. it looks like you’re bumping into a classic combo of searchkick and elasticsearch playing a bit rough on your mac, and that objc message is a definite clue. i’ve seen this rodeo before, so let's break it down.

the core problem stems from a conflict in how ruby is interacting with the underlying libraries that searchkick relies on – specifically, libcurl, which is often the go-to for making network requests, and elasticsearch which is basically a fancy search database. the `objc[36397]: +[number initialize]` thing is the mac os x objc runtime letting you know that something is happening under the hood with the foundation framework's number class. it's usually a side-effect of threading or library initialization issues, not necessarily a core part of your problem.

here’s my take based on what i’ve seen before, and what i suspect is going on:

**the usual suspects:**

1.  **library version mismatches:** this is a biggie. searchkick, elasticsearch, and libcurl have specific version dependencies. if your ruby gem versions don’t quite align with what the operating system expects, things can go sideways fast. for instance, you might have a version of libcurl lurking in a different spot that your ruby gems are trying to talk to, and that version might be clashing. i remember once, back in the rails 4.2 days, i spent hours chasing a similar error. turns out, i'd accidentally installed a dev version of `elasticsearch-ruby` which was causing utter chaos. it was a good learning experience, i suppose.

2.  **threading issues:** searchkick often does its indexing and searching work asynchronously. this can sometimes introduce unexpected interaction with system libraries, especially if multiple things are happening at the same time with threads. this may be amplified because you’re on macOS, as it has particular ways to do threading.

3.  **macOS-specific quirks:** macOS, while awesome, isn’t always the friendliest for setting up development environments. sometimes, the way it manages dynamic libraries can conflict with what ruby gems expect. things like older openssl versions being bundled by your system and affecting libraries is a common problem.

**how to approach this beast:**

here’s the usual process i follow when this kind of error pops up, and what’s worked for me in the past:

*   **gem versions and locking:** first, check your gem versions. run `bundle list | grep elasticsearch` and `bundle list | grep searchkick`. make sure they align with what is expected. it might also be good to check the `searchkick` repo to see if there is a version compatibility table. once you figure out the expected versions it's also a good idea to lock the versions down in your gemfile with a `~>` or `==` and run `bundle update`. make sure you also lock down dependencies for those gems.

*   **isolating the problem:** try running the minimum possible code to cause the error. a simple elasticsearch connection test. if even that triggers it you have the core problem isolated. something like this in a rails console:
```ruby
    require 'elasticsearch'

    client = Elasticsearch::Client.new(host: 'localhost:9200')

    begin
        response = client.ping
        puts "Elasticsearch connection successful: #{response}"
    rescue Elasticsearch::Transport::Transport::Errors::HostResolutionError => e
        puts "Elasticsearch connection failed: #{e}"
    end
```

if this fails, the problem is with the client or connectivity, not with `searchkick` itself.

*   **reinstalling libcurl:** sometimes, rebuilding the libraries that ruby uses can help. try reinstalling the libcurl gem by doing something like:
```bash
    gem uninstall curb
    gem install curb
```

this can help shake things loose sometimes. also, double check that openssl libraries are properly installed.

*   **threading concerns:**  if the problem seems tied to background jobs or async work, try a simplified test where you perform the operations synchronously first, without any threading. something like this in a rails model:
```ruby
    class Article < ApplicationRecord
      searchkick

      def sync_index
       self.reindex
      end
    end

    #in the rails console
    article = Article.first
    article.sync_index
```

if the problem goes away when you do this, then you know threading is the cause.

*   **macOS dynamic libraries:** in certain cases, the issue might stem from the system-wide libcurl on macOS clashing with the ones the ruby gems use. you might have to consider installing libcurl using a package manager like `brew` or re-linking it, though this should be a last resort. i'm not gonna post that here because it is very case specific and if it's needed you'd know it.

**diving deeper:**

if none of that fixes the issue, you might have to delve deeper:

*   **ruby version:** make sure your ruby version is compatible with all gems you're using. an old ruby version could very well be incompatible. check for incompatibilities on the gems repos. i once struggled with a memory issue in an old rails 3 app. it turned out, i needed to patch ruby itself!
*   **system logs:** check system logs for clues. sometimes the error messages in the console are not super descriptive, and other libraries or the system itself may be throwing errors that are being hidden by the ruby process.

**resources i recommend:**

*   **“effective c++” by scott meyers**: despite the name, it is an excellent deep dive into how c++ works that many of the libraries are based on.
*   **"understanding the linux kernel" by daniel p. bovet and marco cesati**: it might seem like overkill but understanding the operating system is very valuable.

i hope this helps point you in the right direction. debugging this kind of issue is usually a process of elimination, so take it step by step. and remember, if all else fails, a fresh `bundle install` and a quick "have you tried turning it off and on again?" is always a good idea. sometimes computers can get stubborn and just need a little nudge.

and now for the only joke i am allowing myself: why did the programmer quit his job? because he didn't get arrays (a raise).
