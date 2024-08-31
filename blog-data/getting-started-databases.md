---
title: "Meta's Engineer guide to getting started with Databases"
date: '2024-08-29'
id: 'getting-started-databases'
---
	•too many keywords, not enough basics.
	•	break it down.
	•	focus on the fundamentals. 
	(did you know the word “fundamentals” comes from the latin “fundamentum” 
	meaning foundation? yeah, latin’s still lurking in our tech lingo.)
	•	learn by doing.
	•	start small, grow from there.
	•	stick with simple stuff. 
	(speaking of simple, did you know the first computer bug 
	was literally a moth found in a computer? crazy, right?)
	•	understand the trade-offs.
	•	optimize when it makes sense.
	•	pick the right tools.
	•	keep the user in mind.
	•	use the right data types. 
	•	choose the right indexes.
	•	write efficient queries.
---
hey, it’s jiang wei from meta,

so, where do we start? look, cramming all the cool buzzwords in isn’t gonna help much. you gotta get what’s going on at the core of database design and how it all fits together.

```java
import java.util.HashMap;
import java.util.Map;

public class KeyValueStore {
    private Map<String, String> store = new HashMap<>();

    public void put(String key, String value) {
        store.put(key, value);
    }

    public String get(String key) {
        return store.getOrDefault(key, null);
    }

    public void delete(String key) {
        store.remove(key);
    }
}
```

first things first, let’s not get too fancy too fast. keep it modular, write everything down (trust me, your future self will thank you), and for the love of code, don’t reinvent the wheel. and yeah, keep learning and optimizing, but don’t overdo it. also, have a plan b for when things go sideways, 'cause they will.

```python
class SimpleCache:
    def __init__(self):
        self.cache = {}

    def set(self, key, value):
        if len(self.cache) >= 5:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key, None)

cache = SimpleCache()
cache.set('a', 1)
cache.set('b', 2)
```

it's great you're comfy with java and python. java's your go-to for handling complex stuff and memory, while python’s more of a quick and dirty, get-things-done kinda tool. they’re both solid choices for building something like a database engine.

```java
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class OperationQueue {
    private BlockingQueue<Runnable> queue = new LinkedBlockingQueue<>();

    public void addOperation(Runnable operation) throws InterruptedException {
        queue.put(operation);
    }

    public void execute() throws InterruptedException {
        while (!queue.isEmpty()) {
            Runnable operation = queue.take();
            operation.run();
        }
    }
}
```

alright, what’s the game plan? think about what you’re really aiming for here. are you building a key-value store, a document store, or something else entirely? keep it simple to start, like maybe a basic key-value store, and then level up as you go. that’s kinda how we kicked things off at meta with our early database projects—focused on the core stuff first, then piled on the rest later.

oh, and don’t forget about the boring stuff like operating systems. knowing how your database plays with the os is key. memory management, disk i/o, multitasking—all that stuff matters more than you think. if you get it right, your database will run smoother than a freshly waxed surfboard. check out [working with file systems](https://engineering.fb.com/2022/06/20/data-infrastructure/transparent-memory-offloading-more-memory-at-a-fraction-of-the-cost-and-power/) and [optimizing for SSDs](https://engineering.fb.com/2021/06/21/data-infrastructure/magma/) for some useful insights.

```python
def write_to_disk(filename, data):
    with open(filename, 'w') as file:
        file.write(data)

write_to_disk('example.txt', 'Hello, Disk I/O!')
```

ever thought about how file systems work? or how to really get the most out of ssds? well, maybe you should. understanding this stuff can make a world of difference. check out [introducing bryce canyon: our next generation storage platform](https://engineering.fb.com/2017/03/08/data-center-engineering/introducing-bryce-canyon-our-next-generation-storage-platform/) and [tao: facebook's distributed data store for the social graph](https://research.facebook.com/publications/tao-facebooks-distributed-data-store-for-the-social-graph/) for more info.

one more thing—take a peek at what’s already out there. look at [redis](https://engineering.fb.com/2022/06/08/core-infra/cache-made-consistent/), [bigtable](https://cloud.google.com/bigtable), and [dynamodb](https://aws.amazon.com/dynamodb/)... you know, the big dogs. they’ve been built for specific reasons, and studying them can give you a solid footing for your project.

now, let’s talk about making sure your stuff actually works:

- unit testing: test each piece on its own, like we do at meta. catch the bugs before they bug you.

```java
public class UnitTest {
    public static boolean testPut() {
        KeyValueStore store = new KeyValueStore();
        store.put("test", "value");
        return "value".equals(store.get("test"));
    }

    public static void main(String[] args) {
        System.out.println("testPut: " + (testPut() ? "Passed" : "Failed"));
    }
}
```

- integration testing: once you’re happy with the individual parts, see how they play together. throw in some real-world scenarios, like network fails or crazy traffic, and see if your system holds up.

```python
def integration_test():
    cache = SimpleCache()
    cache.set('a', 1)
    cache.set('b', 2)
    assert cache.get('a') == 1
    assert cache.get('b') == 2
    cache.set('c', 3)
    assert cache.get('c') == 3

integration_test()
```

- continuous integration (ci): automate your tests. seriously. it’s a lifesaver. at meta, we rely on ci to keep our systems from imploding.

as for that cool paper on autonomous testing, it’s interesting but maybe skip it for now unless you’re really into that stuff. you can dive in when you’re ready. check out [autonomous testing](https://engineering.fb.com/2021/10/20/developer-tools/autonomous-testing/) for some extra reading if you’re curious.

once you’ve got the basics down, you might wanna get into some advanced stuff, like distributed databases. that’s where things get really wild. check out [scaling services with shard manager](https://engineering.fb.com/2020/08/24/production-engineering/scaling-services-with-shard-manager/), [akkio: a distributed database for real-time analytics](https://engineering.fb.com/2018/10/08/core-infra/akkio/), and [fsdp: a distributed database for real-time analytics](https://engineering.fb.com/2021/07/15/open-source/fsdp/) for some deep dives.

and, while we’re at it, a quick rundown on some best practices:

- keep it modular. makes life easier for testing, maintaining, and expanding.

```java
class User {
    private String username;

    public User(String username) {
        this.username = username;
    }

    public String getUsername() {
        return username;
    }
}

class UserManager {
    private Map<String, User> users = new HashMap<>();

    public void addUser(String username) {
        users.put(username, new User(username));
    }

    public User getUser(String username) {
        return users.get(username);
    }
}
```

- write stuff down. yeah, i know, documentation is boring, but it’ll save you a headache later. trust me.
- don’t build everything from scratch. use libraries and tools that are already out there, like [rocksdb](https://engineering.fb.com/2013/11/06/core-data/under-the-hood-rocksdb-a-high-performance-key-value-store/) or [leveldb](https://github.com/google/leveldb).

```python
import json

def load_data(filename):
    with open(filename) as file:
        return json.load(file)

data = load_data('data.json')
```

- keep learning. tech moves fast. keep up with the latest, hit up conferences, and chat with the community.

hope that helps you get your bearings as you dive into this project. building a database engine is a beast, but it’s totally worth it. take it slow, iterate, and don’t be afraid to make mistakes. if you need more tips, just holler.

good luck, and remember—it’s a marathon, not a sprint.

cheers,

jiang.wei@jobseekr.ai

---