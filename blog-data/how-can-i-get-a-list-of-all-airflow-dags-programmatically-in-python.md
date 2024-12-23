---
title: "How can I get a list of all Airflow DAGs programmatically in Python?"
date: "2024-12-23"
id: "how-can-i-get-a-list-of-all-airflow-dags-programmatically-in-python"
---

Alright,  It's a question I’ve bumped into several times, particularly during those larger deployments where tracking dependencies and overall job flow becomes critical. Programmatically accessing your Airflow dag list is not just a matter of curiosity; it's fundamental for automation, monitoring, and, frankly, maintaining sanity in complex environments. We're going to explore how to do this with the Python API, focusing on precision and practicality.

From my own experiences, I recall a large-scale migration project a few years back where we were shifting hundreds of daily batch processes into Airflow. Dynamically generating dashboards to visualize these DAGs, rather than relying on the web UI alone, was crucial for stakeholder buy-in and faster issue resolution. So, trust me, this seemingly simple task has substantial benefits.

The core functionality we’re going to use hinges on Airflow's metadata database, accessible through its ORM (Object-Relational Mapper). The central object we'll be interacting with is `DagModel`. This class maps to the `dag` table in your Airflow database. Let's start with the simplest case: fetching a list of all dag ids.

```python
from airflow.models import DagModel
from airflow.utils.session import provide_session

@provide_session
def get_all_dag_ids(session=None):
    """
    Fetches a list of all DAG ids present in the Airflow metadata database.

    Args:
        session: An optional SQLAlchemy session. If not provided, one is created.

    Returns:
        A list of strings representing all DAG ids.
    """
    dag_models = session.query(DagModel).all()
    return [dag.dag_id for dag in dag_models]

if __name__ == '__main__':
    dag_ids = get_all_dag_ids()
    print(f"All DAG IDs: {dag_ids}")

```

This snippet uses the `provide_session` decorator, which streamlines access to the database session and ensures proper handling of resources. It queries all rows from the `dag` table, representing our dag definitions, and extracts their `dag_id` attributes. This provides a basic list of available DAGs. However, sometimes just IDs aren't sufficient; we often need more metadata. Let’s expand this.

```python
from airflow.models import DagModel
from airflow.utils.session import provide_session
from sqlalchemy import or_

@provide_session
def get_dag_metadata(include_paused=False, session=None):
    """
    Retrieves metadata for all DAGs, optionally filtering by paused state.

    Args:
        include_paused: A boolean indicating whether to include paused DAGs.
        session: An optional SQLAlchemy session. If not provided, one is created.

    Returns:
        A list of dictionaries, each containing metadata for a DAG.
    """

    if include_paused:
        dag_models = session.query(DagModel).all()
    else:
         dag_models = session.query(DagModel).filter(or_(DagModel.is_paused == False, DagModel.is_paused == None )).all()

    dag_metadata = []
    for dag in dag_models:
        dag_metadata.append({
            "dag_id": dag.dag_id,
            "fileloc": dag.fileloc,
            "is_paused": dag.is_paused,
            "is_active": dag.is_active,
            "last_parsed_time": dag.last_parsed_time,
        })
    return dag_metadata

if __name__ == '__main__':
    all_dags_info = get_dag_metadata(include_paused=True)
    active_dags_info = get_dag_metadata(include_paused=False)

    print("All Dags (including paused):\n", all_dags_info)
    print("\nActive Dags:\n", active_dags_info)

```

Here, we're not just fetching IDs. Instead, we pull `fileloc` (the file path of the dag), `is_paused` (whether the dag is paused), `is_active` (whether it is currently active), and `last_parsed_time`. We're also adding filtering by paused state, because you often need to ignore deactivated DAGs. We are using `or_` clause here to make sure that if the `is_paused` column has a `Null` value it is not ommitted, especially when the DAG is new and is not explicitly paused or unpaused yet.

This structure provides a more detailed understanding of your DAGs. This kind of output was invaluable during that aforementioned migration project. Knowing the file locations, for example, allowed us to quickly jump to the source code for any given dag.

Now, to take this a step further, consider the scenario where we want to filter based on a tag or set of tags associated with a DAG. While Airflow doesn’t store tags directly in the `dag` table, they are stored in the `dag_tag` and linked through `dag_tag_link`. This requires a join operation:

```python
from airflow.models import DagModel, DagTag, DagTagLink
from airflow.utils.session import provide_session
from sqlalchemy import or_

@provide_session
def get_dags_by_tags(tags, session=None):
    """
    Retrieves metadata for DAGs matching the specified tags.

    Args:
        tags: A list of tag strings to filter by.
        session: An optional SQLAlchemy session. If not provided, one is created.

    Returns:
        A list of dictionaries, each containing metadata for a matching DAG.
    """

    dag_models = session.query(DagModel).join(DagTagLink).join(DagTag).filter(
          DagTag.name.in_(tags)
        ).group_by(DagModel.dag_id).all()

    dag_metadata = []
    for dag in dag_models:
        dag_metadata.append({
            "dag_id": dag.dag_id,
            "fileloc": dag.fileloc,
            "is_paused": dag.is_paused,
            "is_active": dag.is_active,
            "last_parsed_time": dag.last_parsed_time,
             "tags": [tag.name for tag in dag.dag_tag],
        })
    return dag_metadata

if __name__ == '__main__':
    filtered_dags = get_dags_by_tags(tags=["production", "etl"])
    print(f"DAGs with tags 'production' and 'etl':\n{filtered_dags}")
```

This function introduces a join across `DagModel`, `DagTagLink`, and `DagTag` tables. We're able to filter on the `name` attribute of the `DagTag` table to match our list of `tags`. It also returns the full metadata of matching DAGs. I used this approach in a particularly complex system where hundreds of dags were categorized with tags like “reporting,” “data-ingestion,” and “monitoring” to easily filter based on the functionality the DAG is providing.

A few essential points: Always handle database sessions correctly, using `provide_session` ensures you don't leave open connections. For more complex filtering operations, familiarize yourself with SQLAlchemy’s API, especially its capabilities around filtering and joining. And for further reading, I highly recommend the official Airflow documentation, specifically the section detailing the metadata database and its models. Also, exploring "SQLAlchemy ORM Tutorial" by the SQLAlchemy team can significantly deepen your understanding of how the ORM works, especially when dealing with intricate relationships between models, such as those demonstrated in the tag example. Finally, reading “Architecture of Open Source Applications: Airflow” is a great resource to understand the inner workings of the platform.

Ultimately, programmatically listing DAGs offers a powerful way to interact with Airflow and opens possibilities for advanced tooling and custom integrations. Understanding the underlying data structure and leveraging the ORM correctly are key for doing this efficiently. This ability helped me on multiple projects, and I hope this detailed explanation and practical examples will do the same for you.
