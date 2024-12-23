---
title: "Should a new Solr core be created for indexing nested JSON files and separate fields not linked to the unique key?"
date: "2024-12-23"
id: "should-a-new-solr-core-be-created-for-indexing-nested-json-files-and-separate-fields-not-linked-to-the-unique-key"
---

, let's delve into this. I've encountered similar architectural decisions multiple times, most memorably while working on a large-scale document management system for a financial institution. We were grappling with deeply nested JSON structures representing complex transaction data, which sounds akin to what you're dealing with. The crux of your question—whether to create a new Solr core for indexing nested json files and separate fields not linked to the unique key—isn't a simple yes or no; it depends heavily on specific needs and expected use cases.

My immediate inclination isn't automatically towards a new core, simply because core proliferation often introduces management overhead. Think about backup procedures, schema evolution, and query optimization across multiple cores – all of this adds to complexity. However, a single core isn't always the optimal choice either. Let's examine both sides before we reach a considered answer.

One of the biggest drivers pushing towards a separate core is the issue of 'schema bloat' within the same core. If your nested json structures and detached fields significantly expand the number of fields in the core schema, you're potentially increasing index size, which translates to higher storage and, potentially, query latency. Also, performance might suffer if the majority of your queries target just the primary, non-nested data or the separated, non-unique fields, and the core becomes overwhelmed by processing the extra fields. If the primary field set only accounts for 20% of your index and queries, this is when I would become seriously concerned.

Another key issue is schema complexity and maintainability. Nested JSON can sometimes be represented in Solr using the `flattened` field type, but it can become quite intricate with deeper nests and varying schemas across documents. Maintaining a single schema to encompass such varied data could become challenging. Separate cores allow you to implement schemas optimized for their respective use cases. If we introduce the separated, non-unique fields, you now have potentially several distinct datasets that might benefit from tailored analysis.

On the other hand, sticking to a single core does offer certain advantages. First, query time can be simplified when data is in one place, making joins less complex. Second, and perhaps more significantly, when dealing with data that is related but represented differently, co-location in one core opens possibilities for sophisticated queries combining them using boosting and functions. This is a serious advantage if this type of combination is part of your expected user interaction.

Now, for practical examples. Let's assume, for the sake of demonstration, that you have documents representing customer orders. The core document has a unique id, customer id, order date, etc., and a nested json field called 'order_items' containing details of the individual items in the order. You might also have a separate field, 'store_location,' which is not directly tied to the main document's unique id.

**Scenario 1: Single core with flattened json**

```xml
<!-- Solr schema.xml fragment -->
<field name="id" type="string" indexed="true" stored="true" required="true"/>
<field name="customer_id" type="string" indexed="true" stored="true"/>
<field name="order_date" type="date" indexed="true" stored="true"/>
<field name="order_items" type="json" indexed="true" stored="true"/>
<field name="store_location" type="string" indexed="true" stored="true"/>

<!-- Dynamic field for flatten json -->
<dynamicField name="order_items_*" type="string" indexed="true" stored="true"/>
```

Here, I’m assuming you configure `order_items` as a `json` type and then use a dynamic field to enable the use of `order_items.item_name` etc. You'll need to prepare the json documents during index time, flattening them using the `FlatteningTransformer`. You would do something like this in your solrconfig:

```xml
<updateRequestProcessorChain name="flatten">
    <processor class="solr.FlatteningTransformerFactory"/>
    <processor class="solr.LogUpdateProcessorFactory"/>
    <processor class="solr.RunUpdateProcessorFactory"/>
</updateRequestProcessorChain>

<requestHandler name="/update/flatten" class="solr.UpdateRequestHandler">
    <lst name="defaults">
      <str name="update.chain">flatten</str>
    </lst>
  </requestHandler>
```

**Scenario 2: Single Core with `copyField`**

```xml
<!-- Solr schema.xml fragment -->
<field name="id" type="string" indexed="true" stored="true" required="true"/>
<field name="customer_id" type="string" indexed="true" stored="true"/>
<field name="order_date" type="date" indexed="true" stored="true"/>
<field name="order_items" type="string" indexed="false" stored="true"/>
<field name="store_location" type="string" indexed="true" stored="true"/>

<!-- copy field example -->
<copyField source="order_items" dest="order_items_text"/>

<field name="order_items_text" type="text_en" indexed="true" stored="false" multiValued="true"/>

```
In this scenario, the `order_items` is stored as a `string` and then copied to `order_items_text`, for the text search. I am using a multi-valued text field as a simplified example. You would then have to do some processing at the data ingest level to convert your `order_items` json into a string of text that can be added to the `order_items_text` field.

**Scenario 3: Multiple cores**

*   **Core 1 (orders):** `id`, `customer_id`, `order_date`.
*   **Core 2 (order_items):** `order_id` (reference to the order), `item_id`, `item_name`, `item_price`, etc.
*   **Core 3 (locations):** `location_id`, `location_name`, `location_description`.

Here, each core is specifically structured and would need an associated schema. You'd use joins or cross-core queries to retrieve aggregated information. This method also involves a more complex update and data synchronization.

Considering these examples, I would typically avoid creating a separate core unless:
1. The nested json structure is very deep and the schema changes often.
2. The separate fields, like 'store_location', are not commonly used in core queries, or are used under very different circumstances than the `order` data.
3. The primary dataset (order details) and secondary datasets (items, locations) grow significantly, causing query performance issues.
4. You anticipate different indexing strategies or different update rates for the different data sets.

For more in-depth understanding, you could check *Solr in Action* by Matt Weber and Peter Lubbers. It provides comprehensive coverage of Solr's features and best practices. *Lucene in Action, Second Edition* by Erik Hatcher and Otis Gospodnetic is a helpful resource if you want to get more into the lower-level search engine workings. If you anticipate using graph-like queries, exploring the work done at the Apache TinkerPop project, while not directly Solr, can be helpful in shaping ideas.

In conclusion, while a separate core might seem appealing to solve immediate problems, always consider the long-term implications and try to keep the overall system complexity at a minimum by adopting a single core as your default position. Carefully analyze your query patterns and data volume. If you can reasonably accommodate your data within a single core using approaches like dynamic fields, flattening, or carefully constructed `copyField` statements, it's usually the less disruptive path. You want to avoid introducing unnecessary complexity, and the addition of a new core should be a decision based on robust evidence and necessity.
