---
title: "Securing My Web Service: Best Practices for Public/Private Key Authentication"
date: '2024-11-08'
id: 'securing-my-web-service-best-practices-for-public-private-key-authentication'
---

```javascript
// Server-side (Node.js example)
const express = require('express');
const jwt = require('jsonwebtoken');
const { generateKeyPairSync } = require('crypto');

const app = express();

// Generate RSA key pair
const { privateKey, publicKey } = generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: {
    type: 'spki',
    format: 'pem',
  },
  privateKeyEncoding: {
    type: 'pkcs8',
    format: 'pem',
  },
});

// Example: Endpoint for user authentication (simplified)
app.post('/login', async (req, res) => {
  try {
    const { username, password } = req.body;

    // Replace with your actual user authentication logic
    const user = await authenticateUser(username, password); // Example

    if (!user) {
      return res.status(401).send('Unauthorized');
    }

    // Create JWT
    const payload = { user }; // Include user information in payload
    const token = jwt.sign(payload, privateKey, { algorithm: 'RS256' });

    // Send JWT to client
    res.json({ token });

  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Example: Protected route
app.get('/protected', (req, res) => {
  // Verify JWT
  try {
    const token = req.headers.authorization.split(' ')[1];
    const decoded = jwt.verify(token, publicKey);
    res.json({ message: `Hello, ${decoded.user.username}` });
  } catch (error) {
    res.status(401).send('Unauthorized');
  }
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});

// Client-side (JavaScript example)
// ...

// Store JWT securely (e.g., in local storage)
localStorage.setItem('jwt', token);

// When making requests, include JWT in Authorization header
fetch('/protected', {
  headers: {
    Authorization: `Bearer ${localStorage.getItem('jwt')}`,
  },
})
  .then((response) => {
    // ...
  })
  .catch((error) => {
    // ...
  });
```
