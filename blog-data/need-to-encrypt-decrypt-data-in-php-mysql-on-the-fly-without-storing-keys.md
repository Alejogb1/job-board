---
title: "Need to Encrypt & Decrypt Data in PHP/MySQL on the Fly (Without Storing Keys)"
date: '2024-11-08'
id: 'need-to-encrypt-decrypt-data-in-php-mysql-on-the-fly-without-storing-keys'
---

```php
<?php

// Encryption Key (Manually entered by user)
$encryptionKey = $_POST['encryption_key'];

// Database Connection
$db = new mysqli("localhost", "username", "password", "database_name");

// Select customer data
$sql = "SELECT * FROM customers WHERE customer_id = " . $_POST['customer_id'];
$result = $db->query($sql);
$customer = $result->fetch_assoc();

// Encrypt data
$encryptedData = array();
foreach ($customer as $key => $value) {
  $encryptedData[$key] =  openssl_encrypt($value, 'aes-256-cbc', $encryptionKey);
}

// Update customer record
$sql = "UPDATE customers SET ";
$updateFields = array();
foreach ($encryptedData as $key => $value) {
  $updateFields[] = "$key = '$value'";
}
$sql .= implode(", ", $updateFields);
$sql .= " WHERE customer_id = " . $_POST['customer_id'];

if ($db->query($sql) === TRUE) {
  echo "Customer data encrypted successfully.";
} else {
  echo "Error: " . $db->error;
}

$db->close();

?>
```
