---
title: "How can I log queries from multiple connections outside Laravel's Eloquent?"
date: "2024-12-23"
id: "how-can-i-log-queries-from-multiple-connections-outside-laravels-eloquent"
---

,  I've faced this exact challenge a few times over the years, particularly when dealing with legacy systems or when needing to integrate with databases that aren't directly managed by Eloquent. The standard Laravel query log won't catch these external connections, so you need to employ a bit of a different approach. The key here is to understand that we're essentially hooking into the underlying database driver layer, irrespective of Eloquent.

My first major encounter with this was back when we were migrating a complex reporting application. It had its own custom data access layer, using a mix of direct PDO connections and some ODBC magic for connecting to an ancient mainframe. Laravel was being integrated piece by piece, and we needed visibility into *all* queries, not just the Eloquent-generated ones. The solution involved a combination of custom loggers and some clever wrapping.

The core principle is to intercept the raw SQL execution before it hits the database. We can achieve this at different levels, but focusing on the PDO driver itself gives us the most comprehensive capture. The specific approach varies a little depending on how your external connections are implemented, but the underlying concept remains the same: we need a way to consistently log SQL statements and their associated binding parameters.

Here's a breakdown of how I usually approach this, along with illustrative code snippets.

**Approach 1: Wrapping PDO Statements**

If you're using PDO directly (or a wrapper that allows access to the underlying PDO instance), you can implement a wrapper class. This class will intercept the `execute` method and log the SQL statement and parameters *before* passing it to the real PDO connection.

```php
<?php

class LoggingPDOStatement
{
    private $statement;
    private $logger;

    public function __construct(PDOStatement $statement, \Psr\Log\LoggerInterface $logger)
    {
        $this->statement = $statement;
        $this->logger = $logger;
    }

    public function execute(array $params = null)
    {
        $sql = $this->statement->queryString; // Get the raw SQL
        $logMessage = "Executing Query: " . $sql;
        if ($params) {
            $logMessage .= " with params: " . json_encode($params);
        }
        $this->logger->debug($logMessage);

        return $this->statement->execute($params);
    }


    // Pass through any other methods required.
    public function bindParam($parameter, &$variable, $data_type = PDO::PARAM_STR, $length = null, $driver_options = null)
    {
        return $this->statement->bindParam($parameter, $variable, $data_type, $length, $driver_options);
    }
   // Similar pass throughs for other PDOStatement methods.
    public function fetch($fetch_style = PDO::FETCH_BOTH, $cursor_orientation = PDO::FETCH_ORI_NEXT, $cursor_offset = 0) {
         return $this->statement->fetch($fetch_style, $cursor_orientation, $cursor_offset);
    }

    public function fetchAll($fetch_style = PDO::FETCH_BOTH, $fetch_argument = null, $ctor_args = null)
     {
        return $this->statement->fetchAll($fetch_style, $fetch_argument, $ctor_args);
    }

    public function bindValue($parameter, $value, $data_type = PDO::PARAM_STR){
        return $this->statement->bindValue($parameter, $value, $data_type);
    }

}

class LoggingPDO extends PDO {
    private $logger;

    public function __construct($dsn, $username = null, $password = null, array $options = null, \Psr\Log\LoggerInterface $logger)
    {
       parent::__construct($dsn, $username, $password, $options);
       $this->logger = $logger;
    }
     public function prepare($statement, array $driver_options = array()){
         $stmt = parent::prepare($statement, $driver_options);
         return new LoggingPDOStatement($stmt, $this->logger);

     }
}

// Example Usage:
// Assuming you have a PSR logger instance: $logger
//$pdo = new PDO("mysql:host=localhost;dbname=test", 'user', 'password');
//$stmt = $pdo->prepare("SELECT * FROM users WHERE id = :id");
//$stmt->execute(['id' => 1]);

$logger = new class implements \Psr\Log\LoggerInterface {
    public function emergency($message, array $context = []) { $this->log('emergency', $message, $context); }
    public function alert($message, array $context = []) { $this->log('alert', $message, $context); }
    public function critical($message, array $context = []) { $this->log('critical', $message, $context); }
    public function error($message, array $context = []) { $this->log('error', $message, $context); }
    public function warning($message, array $context = []) { $this->log('warning', $message, $context); }
    public function notice($message, array $context = []) { $this->log('notice', $message, $context); }
    public function info($message, array $context = []) { $this->log('info', $message, $context); }
    public function debug($message, array $context = []) { $this->log('debug', $message, $context); }
    public function log($level, $message, array $context = []) { error_log('['.strtoupper($level).'] ' . $message . json_encode($context) . "\n", 3, __DIR__ . "/app.log"); }
};

$pdo = new LoggingPDO("mysql:host=localhost;dbname=test", 'user', 'password',[], $logger);
$stmt = $pdo->prepare("SELECT * FROM users WHERE id = :id");
$stmt->execute(['id' => 1]);


?>
```

This example shows a wrapper around both `PDOStatement` and `PDO`. In this case, when you instantiate `LoggingPDO` and prepare a statement, it will intercept the execute call and log the details. It uses a simple error log output, but you'd plug in your own PSR logger here (such as a Monolog instance configured in Laravel).

**Approach 2: Using a Database Abstraction Layer with Logging**

If you're working with an abstraction layer on top of PDO, you will need to adjust the wrapping to suit that specific layer. For example, some libraries may expose their own 'execute' method or similar.

```php
<?php
// This example uses a hypothetical DatabaseAbstraction layer.
class LoggingDatabaseAbstraction {
  private $db;
  private $logger;
  public function __construct($db, \Psr\Log\LoggerInterface $logger) {
    $this->db = $db;
    $this->logger = $logger;
  }

  public function executeQuery($sql, array $params = null) {
    $logMessage = "Executing Query: " . $sql;
        if ($params) {
            $logMessage .= " with params: " . json_encode($params);
        }
    $this->logger->debug($logMessage);
    return $this->db->executeQuery($sql, $params);
  }

   public function fetchAll($sql, array $params = null, $fetchType = null) {
      $logMessage = "Executing Query: " . $sql;
        if ($params) {
            $logMessage .= " with params: " . json_encode($params);
        }
      $this->logger->debug($logMessage);
       return $this->db->fetchAll($sql, $params, $fetchType);
  }
  // other methods.
}

// Example Usage. Assume $database is instance of our DatabaseAbstraction

$logger = new class implements \Psr\Log\LoggerInterface {
    public function emergency($message, array $context = []) { $this->log('emergency', $message, $context); }
    public function alert($message, array $context = []) { $this->log('alert', $message, $context); }
    public function critical($message, array $context = []) { $this->log('critical', $message, $context); }
    public function error($message, array $context = []) { $this->log('error', $message, $context); }
    public function warning($message, array $context = []) { $this->log('warning', $message, $context); }
    public function notice($message, array $context = []) { $this->log('notice', $message, $context); }
    public function info($message, array $context = []) { $this->log('info', $message, $context); }
    public function debug($message, array $context = []) { $this->log('debug', $message, $context); }
    public function log($level, $message, array $context = []) { error_log('['.strtoupper($level).'] ' . $message . json_encode($context) . "\n", 3, __DIR__ . "/app.log"); }
};


class DatabaseAbstraction {
  private $pdo;
  public function __construct($pdo) {
    $this->pdo = $pdo;
  }

  public function executeQuery($sql, array $params = null) {
    $stmt = $this->pdo->prepare($sql);
    return $stmt->execute($params);
  }

  public function fetchAll($sql, array $params = null, $fetchType = \PDO::FETCH_ASSOC){
      $stmt = $this->pdo->prepare($sql);
      $stmt->execute($params);
      return $stmt->fetchAll($fetchType);
  }
}

$pdo = new PDO("mysql:host=localhost;dbname=test", 'user', 'password');
$database = new DatabaseAbstraction($pdo);
$loggingDb = new LoggingDatabaseAbstraction($database, $logger);
$loggingDb->executeQuery("SELECT * FROM users WHERE id = :id", ['id'=> 2]);

?>
```

**Approach 3: Monkey Patching Driver Functions (Advanced, Use with Caution)**

This approach is considerably more complex and potentially more fragile, but it's useful if you can't readily access the PDO connection directly. This involves overriding the underlying functions of your database driver, for example if using the standard php mysql client extension. This is often considered undesirable, but in very restrictive situations it could be an option.

```php
<?php
// A very simplified monkey-patch example for illustration only.
// Note: This is NOT recommended for production in most cases,
// but included for completeness.
$logger = new class implements \Psr\Log\LoggerInterface {
    public function emergency($message, array $context = []) { $this->log('emergency', $message, $context); }
    public function alert($message, array $context = []) { $this->log('alert', $message, $context); }
    public function critical($message, array $context = []) { $this->log('critical', $message, $context); }
    public function error($message, array $context = []) { $this->log('error', $message, $context); }
    public function warning($message, array $context = []) { $this->log('warning', $message, $context); }
    public function notice($message, array $context = []) { $this->log('notice', $message, $context); }
    public function info($message, array $context = []) { $this->log('info', $message, $context); }
    public function debug($message, array $context = []) { $this->log('debug', $message, $context); }
    public function log($level, $message, array $context = []) { error_log('['.strtoupper($level).'] ' . $message . json_encode($context) . "\n", 3, __DIR__ . "/app.log"); }
};
$original_mysql_query = 'mysql_query';
if (!function_exists('mysql_query')) {
 function mysql_query($query, $link_identifier = null)
 {
   global $original_mysql_query;
   global $logger;
    $logMessage = "Executing Query (monkey patched): " . $query;
    $logger->debug($logMessage);
   return $original_mysql_query($query, $link_identifier);
 }
}
$conn = mysql_connect('localhost', 'user', 'password');
mysql_select_db('test', $conn);
mysql_query("SELECT * FROM users", $conn);

?>
```
This example shows how to override a very old function, which in most cases is not recommended as the `mysql` extension has been deprecated for years. It can serve as an example to how it could be done with other drivers if no other means of accessing the query exists. *However, this method should be approached with considerable care due to its potential for side effects and fragility.*

**Important Notes & Recommendations**

* **PSR Logger:** Ensure you use a PSR-3 compatible logger. This allows you to easily switch between different logging mechanisms (e.g., Monolog, Syslog) without rewriting the logging logic.
* **Performance Considerations:** Log only what’s necessary. Excessive logging, especially of large datasets, can impact performance. Consider using different logging levels to control the verbosity.
* **Security:** Be careful not to log sensitive data (passwords, personal information, etc.). Use parameter binding, and take care in how you implement the logger.
* **Error Handling:** Properly handle errors that might occur during logging. You don't want your logging mechanism to cause issues with the actual application.
* **Documentation:** Make sure your wrapping or monkey patching code is well-documented and easy to understand.

**Further Reading and Resources**

For a deep dive into related topics, I'd recommend the following:

1.  **"Patterns of Enterprise Application Architecture" by Martin Fowler:** This book has excellent sections on data access layers and how to abstract them properly.
2.  **PSR-3: Logger Interface:** Familiarizing yourself with the PSR-3 standard is essential for standardizing logging within PHP. You can find the official specification on the PHP-FIG website.
3.  **PDO Documentation:** The official PHP documentation for PDO will help you understand how to hook into the query execution process.

The approaches outlined here have served me well in various situations where I've needed to monitor queries from outside of Eloquent. Choosing the right technique depends heavily on your environment and the specific constraints you’re working with. Remember to test your changes thoroughly and always be mindful of potential side effects.
