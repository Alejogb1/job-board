---
title: "One Website, Two IPs: How Do I Do This?"
date: '2024-11-08'
id: 'one-website-two-ips-how-do-i-do-this'
---

```php
<?php

if ($_SERVER['HTTP_HOST'] === 'my_first_domain_name.fr') {
    // Use French site
    // ...
} elseif ($_SERVER['HTTP_HOST'] === 'my_second_domain_name.fr') {
    // Use English site
    // ...
}

?>
```
