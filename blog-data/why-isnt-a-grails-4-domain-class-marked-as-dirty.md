---
title: "Why isn't a Grails 4 domain class marked as dirty?"
date: "2024-12-23"
id: "why-isnt-a-grails-4-domain-class-marked-as-dirty"
---

, let’s tackle this. It’s a common head-scratcher, and something I actually spent a good chunk of time investigating back when I was working on a fairly complex inventory management system built on Grails 4. We had a similar problem where we'd modify domain objects, seemingly in a straightforward fashion, and they just wouldn’t register as dirty with Hibernate, meaning no automatic updates to the database. It became clear that understanding how Grails and Hibernate track changes was paramount.

The core issue isn’t a simple matter of Grails intentionally ignoring changes. Instead, it's a nuanced interaction between how Grails handles its domain classes, how Hibernate tracks modifications, and, crucially, the way we sometimes unintentionally circumvent those tracking mechanisms. In essence, it boils down to two primary reasons: object equality and detached entities.

First, consider object equality. Hibernate uses object identity (reference equality) for change detection within a persistent session. This is typically good enough for most ORM use cases, but we often mistakenly assume any modification to properties within an object will trigger dirtiness. This assumption falls apart when you're working with detached entities. Detached entities are domain objects that have been loaded from the database but are no longer associated with an active Hibernate session. If you load a domain object, modify some of its properties outside the scope of an active session and then try to save, the changes might be lost. Hibernate compares the object loaded with what's currently in the persistence context. If the object being saved is a new object (not found in the persistence context, and not equal to an object that is), then it inserts a record rather than updates the record. If the object being saved is an object loaded in the persistence context (i.e. it's the same object reference), then hibernate will detect changes. If the object being saved is *not* equal to one that is in the persistence context, then hibernate has no way of knowing which data it should update. It is important to understand the difference between `==` (referential equality), `equals()` (logical equality), and that a *persisted object* has a Hibernate representation (an associated object within the persistence context).

Let’s illustrate this with some conceptual code. Remember, I'm deliberately simplifying the database interaction to highlight the core problem:

```groovy
// Conceptual example - Detached entity problem

import grails.gorm.transactions.Transactional

class Product {
    Long id
    String name
    BigDecimal price
    static constraints = {
        name blank: false
        price min: 0
    }
}

class ProductService {

    @Transactional
    Product updateProductPrice(Long productId, BigDecimal newPrice) {
        // Load product object from database (attached to session)
        def product = Product.get(productId)
        // Simulate performing work outside the session
        def detachedProduct = product.clone()
        detachedProduct.price = newPrice
        //  Attempt to update the database
        detachedProduct.save(flush:true) // Will not update if detached and not equals(), will create if not found
        return detachedProduct
    }

    @Transactional
    Product updateProductPriceAttached(Long productId, BigDecimal newPrice) {
         // Load product object from database (attached to session)
         def product = Product.get(productId)
         // Modify the property on the attached entity
         product.price = newPrice
         product.save(flush:true) // Correctly updates since it is in the session.
         return product
    }

    @Transactional
    Product mergeProductPrice(Long productId, BigDecimal newPrice) {
        // Load product object from database (attached to session)
         def product = Product.get(productId)
        // Simulate performing work outside the session
        def detachedProduct = product.clone()
        detachedProduct.price = newPrice
        // Merge the detached entity back into the session
        def mergedProduct = Product.merge(detachedProduct)
        mergedProduct.save(flush:true)
        return mergedProduct
    }
}
```

In `updateProductPrice`, the product loaded from the database is cloned, modified, and then an attempt is made to save the modified clone. Because this is a new object, Hibernate does not recognize that the object needs to update an existing record. However, in the `updateProductPriceAttached` the change is made to the object that is currently associated with an active Hibernate session and therefore changes are correctly updated. Finally in `mergeProductPrice`, the detached object is merged back into the session, which results in Hibernate detecting the change and updates the database.

The second significant reason involves situations where we're not explicitly setting properties but instead modifying objects within collections. Let’s say, for example, we have a Product that has a list of `Variant` objects. If we load a `Product` and then modify one of the existing `Variant` objects in the list, *Hibernate won’t see this as a change to the product object itself.* This is because the list itself hasn't changed. Hibernate doesn't perform deep object analysis every time you make a change within a collection. We have to trigger a change on the collection itself. You can either change the collection (i.e. `myObject.variants.add(newVariant)`) which would trigger a dirtiness, or, you can use the specific `addTo` and `removeFrom` methods generated by grails which update relationships correctly.

Consider the following example:

```groovy
// Conceptual example - Collection Modification issue

import grails.gorm.transactions.Transactional

class Product {
    Long id
    String name
    static hasMany = [variants: Variant]
    static constraints = {
        name blank: false
    }
}

class Variant {
    Long id
    String color
    BigDecimal price
    static belongsTo = [product: Product]
    static constraints = {
        color blank: false
        price min: 0
    }
}

class ProductService {

    @Transactional
    Product updateVariantPrice(Long productId, Long variantId, BigDecimal newPrice) {
        def product = Product.get(productId)
        def variant = product.variants.find { it.id == variantId }
        if(variant) {
            variant.price = newPrice // This does not update the product, or mark as dirty.
            product.save(flush: true)
        }
        return product
    }
    @Transactional
    Product updateVariantPriceCorrectly(Long productId, Long variantId, BigDecimal newPrice) {
        def product = Product.get(productId)
        def variant = product.variants.find { it.id == variantId }
        if (variant) {
            def index = product.variants.indexOf(variant)
            product.variants.get(index).price = newPrice
            // Manually invoke product.save(flush: true), this will mark product as dirty because a change was detected
           product.save(flush: true)

        }
        return product
    }

     @Transactional
     Product updateVariantPriceCorrectly2(Long productId, Long variantId, BigDecimal newPrice) {
         def product = Product.get(productId)
         def variant = Variant.get(variantId)
         if (variant) {
             variant.price = newPrice
             variant.save(flush: true) // This is correct as the variant is attached
          }
        return product
     }

}
```

In `updateVariantPrice` the change to the variant is not detected by Hibernate on the `Product` object. However, `updateVariantPriceCorrectly` makes changes on the `Product` object list itself, which correctly results in an update. `updateVariantPriceCorrectly2` is also correct because it loads a single `Variant` from the database which is then modified and saved to the database.

Now, how do we avoid these pitfalls? Here’s a breakdown of practical solutions:

1.  **Keep entities attached:** The most reliable approach is to keep your entities attached to the Hibernate session as long as possible. Perform your operations within transactional service methods, using `Product.get(id)` to load entities, and make your modifications directly on those objects. Avoid passing detached entities around.
2. **Use the GORM specific relationship methods**: Grails provides relationship methods, such as `addToVariants()` and `removeFromVariants()`, use these methods instead of directly modifying the list. This ensures that Hibernate’s change tracking is properly triggered and relationships are correctly updated.
3.  **Manually trigger dirty checking when necessary**: If you *must* work with detached entities, use the `merge()` method to re-attach them to a session. This will tell Hibernate to compare the modified entity with the one already in the session (or if the entity isn't in the session, create a new record). Remember that after merging, subsequent changes will be tracked. However, note that using `merge` correctly is not trivial as a developer must ensure they are updating the correct fields and not overwriting them with potentially outdated information from the persistence context.

For further reading, I'd highly recommend delving into the Hibernate documentation, specifically the chapters on entity states (transient, persistent, detached), sessions, and change detection. "Java Persistence with Hibernate" by Christian Bauer, Gavin King, and Gary Gregory is also an invaluable resource. Additionally, the official Grails documentation provides thorough explanations of how GORM interacts with Hibernate.

In conclusion, the 'dirty' problem isn't a bug in Grails or Hibernate, but rather a consequence of their design and implementation. By understanding the nuances of object identity, persistence context management, and collection tracking, we can write more robust and predictable applications. It's a case of working with the framework, not against it. And as a developer who has experienced the frustration firsthand, I can confidently say it's worth investing the time to understand.
