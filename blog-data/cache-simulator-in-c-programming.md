---
title: "cache simulator in c programming?"
date: "2024-12-13"
id: "cache-simulator-in-c-programming"
---

Alright so a cache simulator in C eh Been there done that got the t-shirt several times I mean this is like a rite of passage for any self-respecting systems programmer right? I remember my undergrad days fondly or maybe not so fondly trying to get these things to work without segfaulting every other line of code It wasn't pretty let me tell you but it was educational I think

Okay so you're looking at building a cache simulator in C I'm guessing you need something that can model different cache configurations probably with LRU or FIFO replacement policies maybe some write-through write-back stuff and perhaps different block sizes right I mean that's the usual drill with this kind of thing You're not gonna get the full speed benefits of a real cache without building an actual chip which is another story completely I've dabbled in FPGAs and that is a wild west not to derail this too much

First things first lets talk about the core data structures you'll need to handle this You're gonna be storing cache lines in some format each line probably containing a valid bit maybe a tag and a data block I typically go for a struct for this it makes it all nice and tidy

```c
typedef struct {
  unsigned int valid;
  unsigned int tag;
  unsigned char *data;
} CacheLine;
```

See nothing too complicated there valid tag and your data pointer Now you'll want to group these cache lines into sets and then group these sets into the cache You could use multi-dimensional arrays but dynamically allocating it can be more flexible especially when experimenting with different cache sizes and associativity Also I'm a big fan of using malloc to practice my dynamic memory management skills I've had some bad habits in my past where I thought you could just use any pointer anywhere and everything would just work haha that was a joke my professor sure didn't think so

So here is how you might initialize a cache with dynamic memory allocation notice the `calloc` which is a nice way to get your data initialized to zeros which is what we need

```c
CacheLine **createCache(int numSets, int associativity, int blockSize) {
  CacheLine **cache = (CacheLine **)malloc(sizeof(CacheLine *) * numSets);
  if(cache==NULL){
    return NULL;
  }

  for (int i = 0; i < numSets; i++) {
    cache[i] = (CacheLine *)calloc(associativity,sizeof(CacheLine));
    if(cache[i]==NULL){
        // handle error freeing other allocations etc
        return NULL;
    }
    for(int j=0; j<associativity; j++){
        cache[i][j].data = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
         if(cache[i][j].data==NULL){
           // handle error freeing other allocations etc
           return NULL;
         }
    }
  }
  return cache;
}
```

This gives you an array of pointers to arrays which represents your sets and within each set you've got arrays of cache lines Pretty basic stuff really Now the real fun starts when you need to implement those cache access functions I've spent way too many hours debugging this kind of code I once made the mistake of not recomputing the set index correctly between lines of code and it lead to many sleepless nights I think the prof even made it into a case study

To access the cache you need to extract the set index and the tag from your memory address This is where bit manipulation comes in handy I'm assuming you already know the basics or you wouldn't be here If not check out some materials on bitwise operators It's literally the bread and butter of low-level stuff like this

```c
typedef struct {
  unsigned int hit;
  unsigned int miss;
} CacheStats;

CacheStats accessCache(CacheLine **cache, unsigned int address, int numSets, int associativity, int blockSize, CacheStats stats) {
  unsigned int offsetBits = 0; // assuming power of 2 block size
  unsigned int temp = blockSize;
  while (temp >>= 1) {
    offsetBits++;
  }
  unsigned int setBits = 0;
  temp = numSets;
  while (temp >>= 1) {
    setBits++;
  }
  unsigned int tagBits = 32 - offsetBits - setBits;

  unsigned int offsetMask = (1 << offsetBits) - 1;
  unsigned int setMask = (1 << setBits) - 1;
  unsigned int tagMask = (1 << tagBits) - 1;


  unsigned int offset = address & offsetMask;
  unsigned int setIndex = (address >> offsetBits) & setMask;
  unsigned int tag = (address >> (offsetBits + setBits)) & tagMask;


  for (int i = 0; i < associativity; i++) {
    if (cache[setIndex][i].valid && cache[setIndex][i].tag == tag) {
      stats.hit++;
      // Move the hit cache line to the end of the set for LRU implementation. This will require
      // shifting of cache lines and may be slow for very large values of associativity.
      // Note that there are more efficient implementations. But for simplicity this one
      // works well enough.
      if (i<associativity-1){
        CacheLine tmp = cache[setIndex][i];
        for(int j = i; j < associativity -1 ; j++){
          cache[setIndex][j] = cache[setIndex][j+1];
        }
        cache[setIndex][associativity -1] = tmp;
      }

      return stats; // Hit
    }
  }

  stats.miss++;
  // Cache miss implement LRU by replacing the last element in the set
  int lastElement = associativity-1;

  cache[setIndex][lastElement].valid = 1;
  cache[setIndex][lastElement].tag = tag;
  //simulate loading data from memory into this cache line I'm assuming you are
  // using a simple zero initialized char array as data
  for(int i = 0 ; i < blockSize ; i++){
     cache[setIndex][lastElement].data[i] = 0;
  }
  
  return stats; // Miss
}

```

This function is doing the heavy lifting It calculates the set index and tag from the memory address iterates through the cache lines in the specified set and checks if the tag matches It also has a basic LRU implementation which moves hit cache lines to the end of the set This is a naive implementation but it's enough to get you started

Now this is just the basic stuff there are a lot of areas you can expand on You can add support for write policies you can add an access timing model you could simulate a multi-level cache hierarchy oh and don't forget to free the dynamically allocated memory when you are done or you'll get those pesky memory leaks

As for resources I'd recommend looking into "Computer Architecture A Quantitative Approach" by Hennessy and Patterson This book is the bible for computer architecture stuff It's a hefty book but it covers everything you could ever need to know about caches including more advanced topics such as inclusive and exclusive cache strategies And then there is "Modern Processor Design Fundamentals of Superscalar Processors" by Shen and Lipasti this is a bit more into the weeds but its good if you want to explore details. And a good source for practical implementation issues is Patterson and Hennessy's "Computer Organization and Design The Hardware/Software Interface" It covers the practical implications of architectural concepts which are useful for building simulators

Look a lot of this is gonna be about understanding the algorithms and data structures I've given you the basic ingredients but it's up to you to bake the cake You'll be debugging this stuff for hours I am almost certain of it You should get very familiar with your debugger and memory analysis tools This kind of stuff can get very tricky when you are dealing with pointers and memory and it's a great opportunity to hone your low level C skills

So yeah that's my take on cache simulation in C Hope that helps and good luck with your coding adventures
