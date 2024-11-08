---
title: 'Java NIO: "Broken pipe" Error - Quick Fix?'
date: '2024-11-08'
id: 'java-nio-broken-pipe-error-quick-fix'
---

```java
try {
  // Your SocketChannel write() operation
} catch (IOException e) {
  if (e.getMessage().equals("Broken pipe")) {
    // Connection is broken, close the socket
    socketChannel.close();
  } else {
    // Handle other IOExceptions
  }
}
```
