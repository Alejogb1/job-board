---
title: "Is CoreData's merge policy necessary when using external storage?"
date: "2024-12-23"
id: "is-coredatas-merge-policy-necessary-when-using-external-storage"
---

Let’s tackle this head-on; it’s a good question that often sparks debate among those of us who’ve spent some time in the trenches with CoreData, particularly when external storage gets thrown into the mix. The straightforward answer? It’s nuanced. It’s not a binary “yes” or “no.” In my experience, having managed applications dealing with substantial datasets, especially those involving large images and video files, the interplay between CoreData’s merge policies and external storage is crucial to understand to avoid data corruption and performance pitfalls.

The short explanation is that while external storage moves the large data blobs out of the main SQLite database, it doesn't magically make merge policies irrelevant. They work at the level of your `NSManagedObjectContext` and how it handles changes, and external storage is just one aspect of how your attributes are persisted. Think of it this way: your merge policy is the gatekeeper for changes to your persistent store, regardless of whether the actual data is in the main database or pointed to by an external file. It's about managing the *changes* themselves, not where the content ultimately rests.

To truly grasp the need, we need to consider *how* conflicts occur. Core Data uses `NSManagedObjectContext` objects to represent a scratchpad for changes. You modify properties of your managed objects within the context, and when you save that context, Core Data needs to figure out how to reconcile those changes with the persistent store – which could include data changed by other contexts or different threads. This reconciliation process is exactly what the merge policy governs. If you have concurrent modifications to the same entity within different contexts or even different threads and you’re not careful with your merge policy, especially during save operations, you will encounter issues, external storage or not.

Let’s imagine a simple scenario with a `Photo` entity, which has an attribute `thumbnail` that can be quite large. Initially, we might store the thumbnail within the main database. As our data grows, we decide to shift the `thumbnail` attribute to external storage.

Here’s a quick code snippet illustrating the initial database setup (prior to moving to external storage):

```swift
import CoreData

class Photo: NSManagedObject {
    @NSManaged public var id: UUID
    @NSManaged public var thumbnail: Data? // Initial setup, stored inline in the database
}
```
Now, let’s introduce external storage:
```swift
    func moveThumbnailToExternalStorage(photo: Photo) {
         photo.willChangeValue(forKey: "thumbnail")
         photo.allowsExternalBinaryDataStorage = true
         photo.didChangeValue(forKey: "thumbnail")
    }
```
With that change, CoreData is now managing that `thumbnail` attribute to external files. However, the merge conflict problem does not go away. We now have a pointer to external storage, but the act of modifying that pointer or any other attribute for that entity is still subject to conflict rules as defined by your merge policy.

Now, consider this: you have two contexts, `contextA` and `contextB`. Both have fetched the same `Photo` object. `contextA` modifies the thumbnail while context B modifies the same entity, just a different attribute. If they save concurrently, you have a conflict. The external storage doesn’t really enter into the merge itself. Instead, the merge process occurs at the level of your `NSManagedObjectContext` to check whether there were conflicting changes. If your policy is something naive like "overwrite," you could lose changes from either `contextA` or `contextB`.

To illustrate with a practical example:

```swift
import CoreData

func exampleMergePolicy(contextA: NSManagedObjectContext, contextB: NSManagedObjectContext) throws {

    // Assume both contexts have fetched the same 'photo' with ID 'myPhotoID'
    let photoA = try fetchPhoto(id: "myPhotoID", context: contextA)
    let photoB = try fetchPhoto(id: "myPhotoID", context: contextB)


    //Simulate modification in both contexts
    photoA?.thumbnail = Data("new thumbnail data for context A".utf8)
    photoB?.caption = "New caption for context B"

    do {
       try contextA.save()
       try contextB.save()
    }
    catch {
        // Conflict resolution happens based on the context's merge policy
        // Without a proper policy, the last save wins resulting in loss of data
       print ("Error saving contexts: \(error)")
    }
}


func fetchPhoto(id: String, context: NSManagedObjectContext) throws -> Photo? {
    let fetchRequest = NSFetchRequest<Photo>(entityName: "Photo")
    fetchRequest.predicate = NSPredicate(format: "id == %@", id)
    return try context.fetch(fetchRequest).first
}
```

In this example, without a careful merge policy set up on those contexts, either `contextA`’s thumbnail data or `contextB`'s caption will be lost. The external storage will successfully store data, but if you have data modification conflicts, those changes won't make it to the persistance store.

Here’s the critical piece of information: Your merge policy needs to be appropriate for the type of data and how you’re working with it. If you're using a child-parent context setup, with main thread contexts and background processing, the parent context may have to take care of these merges.

So, what should you do? First, understand your data access pattern. If you have write conflicts, decide what you want to do. Should one win over the other? Should you attempt to merge the changes? For situations with potentially conflicting data changes like this, especially in a multithreaded or concurrent environment, you should look into merge policies that preserve both sets of changes where possible and notify you when there is an actual conflict. Policies like `NSMergeByPropertyObjectTrumpMergePolicy` and implementing a custom merge policy using `NSMergePolicy` might be suitable. The first one favors objects with the latest modification timestamps, and with the latter you have complete control over how merges occur.

For a solid foundation on Core Data concurrency and policies, I would recommend starting with Apple’s official Core Data documentation. While often dry, it's the ultimate source of truth. For more conceptual clarity, I’ve often found Marcus Zarra’s books helpful. Specifically, his works on Core Data patterns and advanced techniques helped me manage complex concurrency issues and custom merge conflicts. Also, I recommend reading Apple’s WWDC sessions that have been given on Core Data, as they often present practical use cases that are directly relatable and highly educational.

In summary, while external storage helps with the sheer size of your data, the need for a sound merge policy is not diminished. In fact, with more data being potentially written concurrently, a robust policy becomes even *more* vital. Don’t treat external storage as an end-all fix to data merging challenges. It's a separate optimization that needs to work in concert with well-defined merge strategies. You might not always need sophisticated merge policies in simple applications, but once concurrency becomes a factor, a solid merge policy is not just advisable; it’s often necessary to avoid subtle and difficult-to-debug data inconsistencies.
