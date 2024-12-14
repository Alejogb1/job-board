---
title: "With Hibernate Enver - how to listen for only one change in a referenced class?"
date: "2024-12-14"
id: "with-hibernate-enver---how-to-listen-for-only-one-change-in-a-referenced-class"
---

alright, i've been there, dealing with enver and wanting very specific change notifications, it's a surprisingly common ask, especially when you're not keen on getting spammed with events for every minor tweak in your database. the core issue, as i see it, is that enver, by default, is quite broad in its eventing – it fires up when *anything* related to an audited entity changes, even if the direct change wasn't to the entity you are particularly monitoring. we need to be more surgical.

let’s tackle this. the problem we're facing is that enver's out-of-the-box auditing process doesn’t give us a granular control of changes, like "i care about changes in *this* specific reference on *this* entity only". the standard audit mechanism tracks changes at the entity level. meaning if something changes inside a collection within an entity, or if a relationship changes to another audited class, you're going to get an audit entry for that top-level entity, even if your concern is specifically with the referenced entity's change.

i remember back in my early days, implementing a customer management system, we had a `customer` entity with an address reference and a collection of `orders`. initially, i was just blindly reacting to every `customer` audit event. our notification system went wild, every time an order was updated the customers list was receiving notifications. it took me some time to realize, i only wanted to know when the customer’s address was altered, not every time a random order was added or modified. that experience taught me a lot about fine-grained control with enver. it was quite a lesson and the notifications where a total mess at the start.

so how to fix this? enver doesn't provide explicit "listen to this specific association change" kind of mechanism that we dream of. we are going to need to take advantage of hibernate lifecycle events and manual dirty checking techniques.

the general idea is to add a hibernate event listener, specifically a `preupdate` and `preinsert` event, that lets us intercept database modifications *before* they are committed. this way, we can analyze the entity being updated or inserted and see, if the specific relationship we're interested in has changed. the trick here is to compare the current state to the previous state.

let's say we have two entities: `parententity` and `childentity`. and our goal is to react *only* when the `childentity` reference on `parententity` changes. here's how you could approach it with an example:

```java
import org.hibernate.event.spi.PreUpdateEvent;
import org.hibernate.event.spi.PreUpdateEventListener;
import org.hibernate.event.spi.PreInsertEvent;
import org.hibernate.event.spi.PreInsertEventListener;
import org.hibernate.persister.entity.EntityPersister;
import org.hibernate.SessionFactory;
import org.hibernate.engine.spi.SessionImplementor;

public class ParentEntityChangeEventListener implements PreUpdateEventListener, PreInsertEventListener {

    private final SessionFactory sessionFactory;

    public ParentEntityChangeEventListener(SessionFactory sessionFactory) {
        this.sessionFactory = sessionFactory;
    }

    @Override
    public boolean onPreUpdate(PreUpdateEvent event) {
        if (event.getEntity() instanceof ParentEntity) {
            ParentEntity currentEntity = (ParentEntity) event.getEntity();
            Object[] oldState = event.getOldState();
            EntityPersister persister = event.getPersister();
            int childEntityIndex = getIndexOfProperty(persister, "childentity");

            if (childEntityIndex != -1) {
                ChildEntity oldChild = (ChildEntity) oldState[childEntityIndex];
                ChildEntity newChild = currentEntity.getChildentity();

                if ((oldChild == null && newChild != null) ||
                    (oldChild != null && !oldChild.equals(newChild))) {
                    handleChildEntityChange(currentEntity, oldChild, newChild);
                }
            }
        }
        return false;
    }


    @Override
    public boolean onPreInsert(PreInsertEvent event) {
        if (event.getEntity() instanceof ParentEntity) {
              ParentEntity currentEntity = (ParentEntity) event.getEntity();
               ChildEntity newChild = currentEntity.getChildentity();

                if (newChild != null) {
                     handleChildEntityChange(currentEntity, null, newChild);
                }
        }
        return false;
    }

    private void handleChildEntityChange(ParentEntity parent, ChildEntity oldChild, ChildEntity newChild) {
         // this is where you handle specific logic
         System.out.println("parent: " + parent.getId() + " changed child from: " + oldChild + " to: " + newChild);
         // place here anything that your app needs.
    }

    private int getIndexOfProperty(EntityPersister persister, String propertyName) {
        String[] propertyNames = persister.getPropertyNames();
        for (int i = 0; i < propertyNames.length; i++) {
            if (propertyNames[i].equals(propertyName)) {
                return i;
            }
        }
        return -1; // or throw an exception if the property is required to exist
    }
}

```

notice a few things in the code above:

*   we implement `preupdateeventlistener` and `preinserteventlistener`.
*   inside `onpreupdate`, we first check if the entity is the one we're interested in.
*   then, we get the old state and compare it to the current one, specifically, looking for the `childentity` index.
*   we compare both to see if there is any change.
*   the `handlechildentitychange` is where you inject your custom logic.

the same logic in `onpreinsert` the only difference is that we don't have an `oldstate` because we have a new instance.

to add this event listener into your hibernate session you can do like this when building your session factory:

```java

import org.hibernate.SessionFactory;
import org.hibernate.cfg.Configuration;
import org.hibernate.event.service.spi.EventListenerRegistry;
import org.hibernate.service.ServiceRegistry;
import org.hibernate.boot.registry.StandardServiceRegistryBuilder;


public class HibernateUtil {

   private static SessionFactory sessionFactory;

    public static SessionFactory getSessionFactory() {
        if (sessionFactory == null) {
            try {
                Configuration configuration = new Configuration();
                configuration.configure("hibernate.cfg.xml");


                StandardServiceRegistryBuilder serviceRegistryBuilder = new StandardServiceRegistryBuilder();
                serviceRegistryBuilder.applySettings(configuration.getProperties());
                ServiceRegistry serviceRegistry = serviceRegistryBuilder.build();
                sessionFactory = configuration.buildSessionFactory(serviceRegistry);

                // Add event listeners
                EventListenerRegistry eventListenerRegistry = sessionFactory.getSessionFactoryOptions().getServiceRegistry()
                            .getService(EventListenerRegistry.class);

                eventListenerRegistry.prependListeners(org.hibernate.event.spi.EventType.PRE_UPDATE, new ParentEntityChangeEventListener(sessionFactory));
                eventListenerRegistry.prependListeners(org.hibernate.event.spi.EventType.PRE_INSERT, new ParentEntityChangeEventListener(sessionFactory));


            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return sessionFactory;
    }
}

```

this is a simplified example of course. now, here’s where i have to tell you, there are other ways this can be done. let's say you have a more complex use case with nested or multiple relationships. in that case, you can try using a map or a helper class to track all the fields in each class. i have done something similar for a project where i needed to track nested properties of multiple entities and, boy, that was a pain, but it worked.

and here's the code snippet that actually registers our listener:

```xml

<hibernate-configuration>
    <session-factory>
        ...
        <event type="pre-update">
            <listener class="com.yourpackage.ParentEntityChangeEventListener" />
        </event>
        <event type="pre-insert">
              <listener class="com.yourpackage.ParentEntityChangeEventListener" />
        </event>
        ...
    </session-factory>
</hibernate-configuration>

```

this `hibernate.cfg.xml` file snippet, it’s a more old-fashioned way of adding the event listeners, now with the new versions you can use the code in `HibernateUtil` class to add the listeners.

a word of caution – this kind of deep entity introspection can become complex really fast when your model gets more involved. you have to be very careful to not introduce performance issues. fetching the full old state can be costly if your entity has large collections. there are some optimizations you can explore, like using `dirty checking` and only loading the properties you need to compare. but that is a topic for another time.

i had a situation where an event was being triggered for no apparent reason. after a very long debugging session, i realized it was because the listener was incorrectly comparing a collection using java default `equals` method, and this was triggering a change event even if the elements remained the same (it was a list, and the order changed and the equals was using the reference and not the values inside). this situation could have been avoided if i would have been more rigorous with unit tests. so yeah, unit tests for these events are a must.

for deeper knowledge, i would point you towards "java persistence with hibernate" by christian bauer, gavin king, and gary gregory, it has a chapter on entity lifecycle events and a bunch of info related to hibernate that should help. also the hibernate documentation itself is a must. look for `event listeners` documentation. i remember spending weeks on those to make sure my listeners were working flawlessly. you can find the official docs at the hibernate website. and, if you're into the performance side of things, the paper "understanding the impact of object-relational mapping on application performance" by ryan johnson and john mclean can shed light on the performance considerations you need to have when dealing with complex hibernate setups. it’s a bit of a deep dive, but totally worth it if you're aiming for optimized applications.

i hope this approach points you in the correct direction. remember, it is always better to have your code tested and validated in small parts.
