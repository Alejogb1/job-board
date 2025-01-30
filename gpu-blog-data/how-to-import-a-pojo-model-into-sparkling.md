---
title: "How to import a POJO model into Sparkling Water (Scala)?"
date: "2025-01-30"
id: "how-to-import-a-pojo-model-into-sparkling"
---
Sparkling Water, specifically in the Scala context, requires careful handling when dealing with Plain Old Java Objects (POJOs). Direct import isn't supported out of the box because Sparkling Water, built on top of H2O, operates primarily on H2O frames, which are specialized, distributed data structures. The challenge lies in bridging the gap between the typical structure of a POJO and the columnar, compressed representation that H2O uses.  From my experience transitioning a legacy banking application onto a Spark infrastructure incorporating machine learning, I encountered this hurdle early on.

To clarify, you can't directly pass a collection of POJOs, like a `List<Account>` where `Account` contains fields like `accountId`, `balance`, and `transactionHistory` to H2O/Sparkling Water functions expecting an H2O frame. Instead, I've found that converting the POJOs into a suitable format, typically a Spark RDD or Dataset, and then subsequently into an H2O frame, is the most efficient path. This process involves serialization, schema definition, and potentially data type conversion, all of which I'll elaborate on.

The first critical step is to ensure the POJO is serializable. This often involves implementing the `java.io.Serializable` interface (or, in certain Scala cases, a corresponding mechanism). This makes your POJOs eligible for distribution across the Spark cluster. Once your POJOs can be serialized and shipped, the subsequent transformation hinges on your starting point within the Spark ecosystem.

**Code Example 1: Importing from a Spark RDD of POJOs**

Let's consider a scenario where you have an RDD already containing your POJOs, for example an `RDD[Account]`. I'll assume the Account POJO looks something like this:

```java
// Java Account POJO (must implement Serializable)
import java.io.Serializable;

public class Account implements Serializable {
    private String accountId;
    private double balance;
    private String lastTransactionDate;

    // Constructor, getters, setters...
    public Account(String accountId, double balance, String lastTransactionDate) {
        this.accountId = accountId;
        this.balance = balance;
        this.lastTransactionDate = lastTransactionDate;
    }

    public String getAccountId() {
        return accountId;
    }

    public void setAccountId(String accountId) {
        this.accountId = accountId;
    }

     public double getBalance() {
        return balance;
    }

    public void setBalance(double balance) {
       this.balance = balance;
    }
    public String getLastTransactionDate(){return lastTransactionDate;}
    public void setLastTransactionDate(String lastTransactionDate){this.lastTransactionDate = lastTransactionDate;}
}
```

Now, on the Scala side, let’s import some essentials:

```scala
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.h2o._
import java.util.Arrays

object PojoImportExample {

  def main(args: Array[String]): Unit = {

      val sparkConf = new org.apache.spark.SparkConf().setAppName("PojoImport").setMaster("local[*]")
      val sc = new SparkContext(sparkConf)
      val h2oContext = H2OContext.getOrCreate(sc)
    import h2oContext._

    // Example of creating Java POJOs
     val account1 = new Account("ACC123", 1000.0, "2024-02-29")
     val account2 = new Account("ACC456", 5000.0, "2024-02-20")
     val accountsList = Arrays.asList(account1, account2)

      // Create an RDD of POJOs
    val accountsRDD: RDD[Account] = sc.parallelize(accountsList)

    // Convert the RDD of POJOs to an H2O Frame
    val accountsHF: H2OFrame = accountsRDD
    // accountsHF is now ready to be used by Sparkling Water models and algorithms.
    accountsHF.show()

  }
}
```
**Commentary:** The crucial line is `val accountsHF: H2OFrame = accountsRDD`. Sparkling Water provides an implicit conversion from an RDD to an H2OFrame *if* the RDD's elements can be converted. This conversion process internally relies on Spark serialization and subsequently transforms each POJO into a suitable columnar representation within the H2O frame. The H2O frame `accountsHF` is now ready to be used for H2O specific algorithms. The `show()` command helps verify the successful creation of the H2O Frame.

**Code Example 2: Importing from a Spark Dataset of POJOs**

The approach changes slightly when working with Spark Datasets, a structure that benefits from more strongly-typed schemas. I tend to prefer datasets for structured data. Here's how to handle that scenario, still assuming the same POJO class:

```scala
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.h2o._
import java.util.Arrays

object DatasetPojoImportExample {

  def main(args: Array[String]): Unit = {

      val spark = SparkSession
        .builder()
        .appName("DatasetPojoImport")
        .master("local[*]")
        .getOrCreate()
      val h2oContext = H2OContext.getOrCreate(spark)

      import h2oContext._
      import spark.implicits._

        // Example of creating Java POJOs
        val account1 = new Account("ACC123", 1000.0, "2024-02-29")
        val account2 = new Account("ACC456", 5000.0, "2024-02-20")
      val accountsList = Arrays.asList(account1, account2)

      // Create an RDD of POJOs
        val accountsRDD = spark.sparkContext.parallelize(accountsList)

      // Create a Dataset of POJOs
      val accountsDS: Dataset[Account] = accountsRDD.toDS()

    // Convert the Dataset of POJOs to an H2O Frame
      val accountsHF: H2OFrame = accountsDS

      accountsHF.show()
  }
}
```

**Commentary:** In this example, `accountsRDD.toDS()` converts our RDD into a Spark Dataset. This step automatically infers the schema from the `Account` class. The conversion to an `H2OFrame` is again seamless using implicit conversions:  `val accountsHF: H2OFrame = accountsDS`. H2O can interpret the schema defined by Spark in the dataset and create the H2OFrame representation accordingly.

**Code Example 3: Explicit H2OFrame Construction with Data Types**

While implicit conversions work in the majority of cases, sometimes we need more granular control, particularly regarding data types. Here is an example that uses `H2OFrame` constructor directly:

```scala
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.h2o._
import water.Key
import water.fvec.{H2OFrame, Vec}
import java.util.Arrays
import scala.collection.JavaConverters._


object ExplicitPojoImportExample {

  def main(args: Array[String]): Unit = {

      val sparkConf = new org.apache.spark.SparkConf().setAppName("ExplicitPojoImport").setMaster("local[*]")
      val sc = new SparkContext(sparkConf)
      val h2oContext = H2OContext.getOrCreate(sc)

      import h2oContext._

    // Example of creating Java POJOs
      val account1 = new Account("ACC123", 1000.0, "2024-02-29")
      val account2 = new Account("ACC456", 5000.0, "2024-02-20")
      val accountsList = Arrays.asList(account1, account2)

      // Create an RDD of POJOs
     val accountsRDD: RDD[Account] = sc.parallelize(accountsList)

     // Transform POJOs to tuples for H2OFrame
      val tuplesRDD: RDD[(String, Double, String)] = accountsRDD.map(account => (account.getAccountId, account.getBalance, account.getLastTransactionDate))

      // Extract the data into lists, to create H2O Vectors
     val idCol = tuplesRDD.map(_._1).collect().toList.asJava
     val balCol = tuplesRDD.map(_._2).collect().toList.asJava
     val dateCol = tuplesRDD.map(_._3).collect().toList.asJava

     // Construct the H2O frame from vectors
     val accountHF = new H2OFrame(
      Key.make(),
      Array[String]("accountId","balance","lastTransactionDate"),
      Array[Vec](
       Vec.makeVec(idCol.toArray(Array[String]()) , null),
       Vec.makeVec(balCol.toArray(Array[java.lang.Double]()), null),
       Vec.makeVec(dateCol.toArray(Array[String]()), null)
      )
     )

     accountHF.show()
  }
}
```

**Commentary:** Here, I'm constructing the H2O frame directly by creating individual H2O `Vec` objects and defining their associated column names using the `H2OFrame` constructor. This process requires more effort but provides explicit control over the data types used in the resulting H2OFrame and allows for types that do not implicitly map using the automatic conversions described earlier. You'll notice usage of `Vec.makeVec` methods for creation, as H2O’s `Vec` objects are foundational. This demonstrates that it is possible to move away from the implicit transformations if a high level of customisation is needed. It involves extracting columnar data from the POJOs and converting that data into H2O `Vec` objects. It also demonstrates explicit naming of columns using a string array.

In summary, while you cannot directly import POJOs, converting them to Spark RDDs or Datasets forms a bridge that Sparkling Water uses to bring them into the H2O ecosystem as H2O Frames. Implicit conversions are often adequate, but manual frame creation offers granular control. When dealing with POJOs, serialization should always be checked, and a strategy should be chosen based on the trade-off between ease of use and control.

For further investigation, I suggest delving deeper into the following:

*   **Spark documentation:** Thoroughly understand how RDDs and Datasets work including implicit conversions.
*   **H2O and Sparkling Water documentation:**  Specifically review the details on H2O frames, data types, and the process of data import using implicit transformations and constructor-based techniques.
*   **Serialization mechanisms:** Study the best practices when dealing with Java serialization in distributed environment. Focus on how the POJOs are serialized and deserialized between Spark and H2O.
*   **Data type considerations:** Pay close attention to how Java types map to those used by H2O during transformation to `H2OFrame`.
