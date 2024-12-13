---
title: "gsutil sync vs cp n?"
date: "2024-12-13"
id: "gsutil-sync-vs-cp-n"
---

Alright so gsutil sync versus gsutil cp -r I've been down this rabbit hole before believe me its a classic and it always sparks a good debate I've seen junior devs completely fumble this one on numerous occasions so lets get into it

Ok first off lets ditch the hype and look at the actual nuts and bolts The basic problem you're facing is moving data from a source to a destination right we all do this everyday its like the bread and butter of any cloud based interaction the real kicker is *how* you do it

`gsutil cp -r` is your classic recursive copy its dumb in a way it does exactly what it says its the brute force method copy everything from point A to point B It recursively goes down directory structures and copies all files and directories no magic no bells just pure unadulterated file copying Its like a dump truck just pouring files no questions asked you asked for a copy you get it all.

Now lets imagine this scenario i had back in my early days working with some huge image datasets for a training model the data was around 20TB or so it was absolutely massive The project required syncing the latest version of that data from our staging bucket to the training bucket on a nightly basis We were using `gsutil cp -r` initially thinking it was the straightforward approach The first night it took almost the entire night to complete and when we tried checking the status the log files had literally terabytes of duplicate log entries about already copied files it was clear that this wasnt ideal that was the first sign we had to do something better The next morning we saw a massive spike in our bills i swear the billing system almost imploded. I mean it was so bad it was laughable in a sad way you know the feeling.

Here's an example of how we were doing it initially nothing fancy just plain dumb copying

```bash
gsutil cp -r gs://source-bucket/data/ gs://destination-bucket/data/
```

This does work yes it copies everything but it doesnt care if those files are already there if a file exists already it gets copied again wasting time and bandwidth Its like trying to move your entire house every time you get a new pair of socks you just dont do it you only move what you need

Now `gsutil sync` on the other hand it is the sophisticated option Its smarter it only moves what has changed or what doesn't exist in the destination It does a comparison between the source and the destination before copying it checks modification times file sizes it's not just copying everything blindly. Think of it like a good librarian they know what books are already in the library and only bring in the new ones or the updated versions. It's efficient it saves time and it saves a ton of money when you're working with large datasets that change frequently.

Back to my image dataset saga after that initial billing scare we started looking for alternatives we dived into the `gsutil` documentation and found the glory that is `gsutil sync` we were hesitant at first we had used cp our entire life it was a tough change but we had to do it. The difference in performance and cost was mind blowing. Our nightly data syncs went from 8+ hours to just a few minutes. The data was still huge but now only the latest images and their associated files were getting transferred It was an absolute game changer.

Here's the `gsutil sync` command we ended up using it was way cleaner:

```bash
gsutil sync gs://source-bucket/data/ gs://destination-bucket/data/
```

See? looks similar but its a totally different beast. Now in our case we also started adding specific options to even further improve the behavior we used options like `-d` which would remove files in the destination if they do not exist in the source which is very important in some scenarios like when files are removed in the upstream and should also be removed on the downstream target

```bash
gsutil sync -d gs://source-bucket/data/ gs://destination-bucket/data/
```

The -d option means that if a file is in the destination and not the source it will be deleted This is useful if you want to maintain an exact replica or if the source bucket is being used as a central source of truth. Just be very careful when using this option especially in production to avoid accidental data loss that could lead to a bad day.

Now what are the downsides or caveats. Well `gsutil sync` does require some additional overhead because of the comparison step it needs to do that initial check to figure out whats changed or what doesn't exist. However for most use cases the benefits outweigh the added overhead. For the most part when working with any respectable amount of data I would absolutely recommend using `gsutil sync`.

There are some nuances that we found later when it comes to how sync works with modified timestamps. It relies on the last modification time to determine if the file has changed If the modification time is the same but the file content has changed sync will miss it that's something you have to keep in mind. Now if that was to happen you can add a checksum validation to it which will force a re-copy in cases of mismatching content with same mod times.

Now if you want to dive deeper and actually understand what's going on under the hood I'd recommend that you take a look at distributed file system papers look for research on efficient data synchronization and parallel transfers Its good to understand the core of this not just use it blindly.

To summarise `gsutil cp -r` is the dumb but simple way perfect for small quick copies or when you absolutely need to copy everything no matter what. On the other hand `gsutil sync` is your workhorse for large frequent transfers it is the better choice for most use cases especially when dealing with large datasets where efficiency is crucial. Its one of those things that seems simple on the surface but has a lot of under-the-hood magic going on. It's another tool that shows the importance of understanding the trade-offs between convenience and efficiency.
