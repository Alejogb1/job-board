---
title: "How can a repository be implemented for a child aggregate entity in Domain-Driven Design?"
date: "2024-12-23"
id: "how-can-a-repository-be-implemented-for-a-child-aggregate-entity-in-domain-driven-design"
---

Right, let's tackle this. The topic of repositories for child aggregates within a Domain-Driven Design (ddd) context is something I've personally seen trip up quite a few teams, myself included back in the early days. It's a nuanced area, not least because it directly challenges the notion of an aggregate root as the sole point of access for data modification. It's easy to fall into traps that undermine the core principles of ddd if you're not careful. So, how *do* we handle this practically? The short answer: generally, you *don’t* create a repository directly for a child aggregate. It might seem counterintuitive at first, but bear with me as we explore the rationale.

My past experience has taught me that directly querying or modifying child entities outside the context of their parent aggregate often leads to inconsistencies. Imagine, for example, a scenario where you’re modeling an `Order` aggregate. It naturally includes child entities such as `LineItem`. If we allow a repository to fetch individual `LineItem` entities independently of their parent `Order`, we're essentially breaking the aggregate’s consistency boundaries. You could, for instance, modify a `LineItem` and not have it reflected properly in the overall `Order` aggregate state, or introduce logic to handle this that becomes cumbersome. We quickly end up with distributed updates and inconsistencies.

The cardinal rule of aggregates in ddd is that the aggregate root—in this case, the `Order`—is the gatekeeper to all state changes within that aggregate. All operations that affect child entities must go through the root. Therefore, the repository should primarily deal with aggregate roots. Now, let's look at this in practice and how we can practically manage fetching and modification of child aggregates.

The first principle is that access to child aggregates should always be through operations on the parent aggregate root. This means you'll be using methods on the aggregate root to fetch or modify the child entities. For example, instead of having a `LineItemRepository` with `findById`, we’d retrieve the parent `Order` via `OrderRepository`, and then operate on the line items through methods on the `Order` aggregate.

Here’s a simplified code snippet (using a pseudo-language since specific implementation details vary depending on context) to illustrate fetching data:

```pseudocode
// Assuming an Order aggregate with a collection of LineItem entities.

class Order {
    orderId;
    customer;
    lineItems;

   // method to retrieve an individual lineItem
    getLineItemById(lineItemId) {
       return lineItems.find(item => item.itemId == lineItemId);
    }

    // method to add a lineItem
    addLineItem(item){
      lineItems.add(item);
    }

    // method to remove a lineItem
    removeLineItem(itemId){
        lineItems.filter(item => item.itemId != itemId);
    }
}

class OrderRepository {
   findById(orderId) {
     // logic to retrieve the order from data store
   }

    save(order){
        //logic to persist the order in data store
    }
}

class ApplicationService{
    orderRepository;

    getLineItem(orderId, lineItemId){
        order = orderRepository.findById(orderId);
        return order.getLineItemById(lineItemId);
    }

    addLineItemToOrder(orderId, item){
        order = orderRepository.findById(orderId);
        order.addLineItem(item);
        orderRepository.save(order);
    }

    removeLineItemFromOrder(orderId, itemId){
        order = orderRepository.findById(orderId);
        order.removeLineItem(itemId);
        orderRepository.save(order);
    }
}

```

In this example, we don’t have a separate `LineItemRepository`. Instead, all operations relating to `LineItem` occur via methods within the `Order` aggregate, which is retrieved and persisted through the `OrderRepository`. The application service then coordinates the operations using the provided methods.

Now, let’s address scenarios where you *might* think you need a repository for a child. Consider read models or projection layers. In some cases, you may need to optimize queries for specific read-only use cases that target child entities, but these should be projections of the aggregate’s data, not direct access to the persisted child entities. These read models are often handled separately and do not typically fall under the domain model. This is essential to separate concerns: the domain model focuses on maintaining consistency and behavior, while projection models facilitate efficient querying for UI or reporting.

Here's an example of projecting data for read purposes, potentially into a view model:

```pseudocode
//example projection layer for optimized read operations

class LineItemViewModel{
    itemId;
    orderId;
    itemName;
    quantity;
    price;
}

class ReadOnlyLineItemRepository{
  findById(itemId){
     //logic to query a projection table for line items
     //and return a LineItemViewModel
  }

   findForOrder(orderId){
     //logic to query a projection table for all line items
     //related to the given orderId
     //and return a list of LineItemViewModels
   }
}

class ApplicationService{
  readOnlyLineItemRepository;

   getLineItemDetails(lineItemId){
     return readOnlyLineItemRepository.findById(lineItemId);
   }

   getLineItemsForOrder(orderId){
     return readOnlyLineItemRepository.findForOrder(orderId);
   }
}
```

Here, our `ReadOnlyLineItemRepository` doesn't interact with the domain's `LineItem` entity directly. Instead, it fetches data from a pre-built projection, often stored separately and optimized for read performance. These projections would be populated in an eventual consistency manner, for example, by listening to domain events triggered when the aggregate is modified.

Finally, consider scenarios where you have very complex child aggregate structures and are having difficulty maintaining transactional consistency. In these rare cases, if your child aggregate is truly behaving like a small aggregate in its own right, you might consider refactoring to make it an independent aggregate. However, this should only be considered after very careful analysis and not as a way to circumvent core ddd principles. Be aware that this shift can drastically change your domain boundaries and require a deeper understanding of your requirements.

```pseudocode
//hypothetical scenario where we promote LineItem to an aggregate.
//This should be done very rarely and with great care
class LineItem{
   itemId;
   orderId; // link to the order
   itemName;
   price;
   quantity;

   updateQuantity(newQuantity){
        quantity = newQuantity;
   }
}

class LineItemRepository{
  findById(itemId){
      // Logic to retrieve LineItem
  }

  save(lineItem){
      // Logic to persist lineItem
  }
}

class Order{
    orderId;
    customer;
    lineItemIds;

    addLineItem(itemId){
       lineItemIds.add(itemId);
    }
}

class ApplicationService{
    lineItemRepository;
    orderRepository;

    updateLineItemQuantity(itemId, quantity){
         lineItem = lineItemRepository.findById(itemId);
         lineItem.updateQuantity(quantity);
         lineItemRepository.save(lineItem);
    }

    addLineItemToOrder(orderId, itemId){
      order = orderRepository.findById(orderId);
      order.addLineItem(itemId);
      orderRepository.save(order);
    }
}

```

In this hypothetical example, `LineItem` is now an aggregate root with its own repository. We've significantly altered the relationships and data access. Notice how we're now dealing with a list of ids in `Order`, and we’re managing the `LineItem` aggregate independently. Note that this is a design decision made for very specific situations and not a common approach.

In summary, repositories in ddd are intended for aggregate roots. Access to child aggregates should be mediated via methods on the root itself. Read models or projections provide performance enhancements for read-only scenarios. In very rare cases, and only after careful consideration, you might consider promoting a child aggregate to an aggregate root itself, but such cases should be exceptional. For deeper dives, I recommend exploring *Domain-Driven Design: Tackling Complexity in the Heart of Software* by Eric Evans, and *Implementing Domain-Driven Design* by Vaughn Vernon. These are foundational texts that will solidify your understanding of the topic far better than any casual internet resource. These principles are key to building robust, maintainable, and consistent systems using ddd.
