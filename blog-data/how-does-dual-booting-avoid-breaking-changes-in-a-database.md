---
title: "How does dual booting avoid breaking changes in a database?"
date: "2024-12-23"
id: "how-does-dual-booting-avoid-breaking-changes-in-a-database"
---

Okay, let's talk about dual booting and how it can act as a safety net when working with databases, especially concerning breaking changes. It's a topic that’s definitely come up a few times over my career, most notably when I was migrating a legacy application to a new stack – that particular project had me sweating more than a few times. The database, as you can imagine, was the heart of the issue.

The core idea behind using dual booting for this purpose isn't about directly altering the database’s structure within the boot process itself. Instead, it's about providing an *isolated* operating system environment where you can test database changes without the risk of impacting the production system. Think of it as a dedicated laboratory. When we talk about "breaking changes," we typically mean modifications to the database schema or data that could render existing applications incompatible or corrupt data. This can include things like altering column types, renaming tables, or removing fields that an application relies on. Testing these changes directly on a production database without a safety net is, frankly, a recipe for disaster.

Dual booting, in this context, provides a powerful mechanism for mitigating these risks. With a dual-boot configuration, you typically have two separate operating system installations on the same machine, each booting from its own disk partition or virtual disk. Crucially, they can be configured to use independent database instances. Let's say we're using postgresql as an example; we’d set up two separate postgresql installations, each with its own data directory, port and configuration. The first one might house your production database which your legacy app uses and the second database will be used for development, staging or testing with your new app/schema changes.

Here’s the process in action as we applied it in my previous job:

1.  **Environment Setup:** We had a machine, let's call it `dev-station`, with two separate SSDs. On one drive, we kept the production-like environment; essentially, an exact copy of the OS and application code running live, including a copy of the production database (cloned, not actively mirrored!). The second drive had the development environment. Both environments were set up with independent copies of the database, and we utilized different ports to ensure there was no conflict between the two instances.

2.  **Schema Changes and Application Modifications:** When we needed to introduce database changes for the new stack, we worked exclusively on the development environment. This included any SQL schema modifications, data migrations and any necessary changes to the application that interfaced with the database, keeping our production environment completely untouched.

3.  **Testing and Validation:** We ran all of our tests within the development OS. If anything broke in that environment, it would be contained there, without any disruption to production. This allowed us to catch any compatibility issues or data migration problems before they could affect the live system.

4.  **Deployment (Controlled Rollout):** After thorough testing within the dual-booted environment, we were confident in the changes and could confidently move them to production. Usually, this involved a careful rollout, testing on a smaller group of users to start and incrementally increasing the load.

Now, let's get into some code examples. Note that these are conceptual examples to demonstrate database interaction; full configuration would depend heavily on your specific OS and database setup.

**Example 1: Starting Separate PostgreSQL Instances**

This example demonstrates starting two PostgreSQL instances on different ports. While this wouldn't be run from inside an application, understanding how instances are managed at the system level helps conceptualize the isolation provided by the dual-boot environment.

```bash
# on the production environment (booted from first drive)
sudo systemctl start postgresql

# assume postgresql listens on default port 5432
# and uses a production database named "prod_db"

# on the development environment (booted from second drive)
sudo systemctl start postgresql@1111

# this command will start another postgresql instance listening on a different port 1111
# it uses different configuration and data directories. This would be manually configured
# using systemctl configuration
# and would utilize a development database called "dev_db"
```

**Example 2: Data Migrations with Different Databases**

This shows a conceptual example of how you might use separate database connections for a migration process on your dev and prod databases. We use python for this example but the principle is the same. Assume you are migrating from one schema to another. We use SQLAlchemy for this example but its similar for most ORMs.

```python
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select
from sqlalchemy.orm import declarative_base, Session

# Base for SQLAlchemy ORM
Base = declarative_base()

# Define our Models
class OldUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String)


class NewUser(Base):
    __tablename__ = "new_users"
    id = Column(Integer, primary_key=True)
    name = Column(String)

# Function to perform the migration
def perform_migration(source_db_url, dest_db_url):
    prod_engine = create_engine(source_db_url)
    dev_engine = create_engine(dest_db_url)

    Base.metadata.create_all(dev_engine)

    with Session(prod_engine) as prod_session:
        with Session(dev_engine) as dev_session:

            # Simple data copy for migration purposes
            for old_user in prod_session.query(OldUser).all():
                new_user = NewUser(id=old_user.id, name=old_user.username)
                dev_session.add(new_user)
            dev_session.commit()
            print("Migration Complete")

if __name__ == "__main__":
    # prod database url
    prod_db_url = "postgresql://user:pass@localhost:5432/prod_db"
    # dev database url
    dev_db_url = "postgresql://user:pass@localhost:1111/dev_db"
    perform_migration(prod_db_url, dev_db_url)
```

In this example, you can see that the `perform_migration` function explicitly connects to two distinct databases, identified by their URLs. This would be running within the development environment. This level of isolation is only possible within a setup like the one I have described where we have two isolated operating systems and databases on the same hardware.

**Example 3: Configuration File Differences**

Here is how we might change our application to use different databases based on the system.

```python
import os

def get_database_url():
    if os.path.exists("/mnt/second_drive"):
        # Development Environment
        return "postgresql://user:pass@localhost:1111/dev_db"
    else:
        # Production Environment
        return "postgresql://user:pass@localhost:5432/prod_db"


# Example usage
database_url = get_database_url()
print(f"connecting to database at: {database_url}")
# Create database engine.
# engine = create_engine(database_url)
```

In this case we can detect if our second drive is mounted, meaning it is the development environment and change our database connection based on it. It is a basic way of doing it but the same effect can be achieved using environment variables.

For those who are new to this kind of setup, I'd highly recommend diving into some relevant reading. “Database Internals: A Deep Dive into How Distributed Data Systems Work” by Alex Petrov is an excellent resource for understanding the inner workings of databases and why you would need to approach schema changes with caution. For those interested in learning more about system administration and environment configuration, “Linux System Administration” by Vicki Stanfield and Roderick W. Smith is a good start. Understanding the underlying operating system is important to understand how to correctly implement this methodology. Also you may want to study up on modern database migrations using alembic or liquibase for practical considerations.

Ultimately, the dual boot strategy allows for a "practice run" of the database changes and ensures that when deployment day comes, the migration goes off without a hitch. This is why it becomes very useful in complex migrations or where there are changes to critical tables and fields. It is a time investment for setting up but well worth the cost if it is necessary in your scenario. We ended up doing the same thing for major framework upgrades too. The overall time investment is outweighed by the peace of mind and confidence it gives you and your team.
