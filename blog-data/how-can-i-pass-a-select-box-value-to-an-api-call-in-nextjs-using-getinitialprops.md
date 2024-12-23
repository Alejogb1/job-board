---
title: "How can I pass a select box value to an API call in Next.js using `getInitialProps`?"
date: "2024-12-23"
id: "how-can-i-pass-a-select-box-value-to-an-api-call-in-nextjs-using-getinitialprops"
---

Alright,  I've definitely been down this road more times than I care to count, especially back when `getInitialProps` was the workhorse for data fetching in Next.js. So, the scenario is this: you have a select dropdown, a user picks an option, and you need to use that selected value to make an API request to populate your page with dynamic data. It’s a very common use case. We'll focus on how to achieve this specifically with `getInitialProps`, keeping in mind that while it's considered a legacy approach now, understanding it is crucial for dealing with older Next.js projects and even understanding the evolution of data fetching.

Essentially, `getInitialProps` runs server-side on initial page load and then also client-side when navigating between pages within the application. Crucially for your problem, it gives us access to the incoming context, which includes query parameters derived from your URL. Our goal here is to capture the select box value, serialize it (usually within the URL), and then make use of this value within `getInitialProps`. Let me walk you through a typical strategy, based on how I've handled this in the past, along with three concrete examples.

**The Basic Flow**

First, you'd have your select box on the page. When the user makes a selection, you'll need to programmatically trigger a navigation to the same page but with the selected value appended to the URL as a query parameter. Let’s say, for simplicity, our select box is for choosing categories of products, and we'll use `category` as the query parameter. This is handled in the client-side. Then, on the server side, `getInitialProps` will parse this parameter and use it in our API call.

**Example 1: Simple Product Listing**

Imagine you are building a rudimentary e-commerce site. The primary view is a product listing page, and the category filter is a dropdown menu.

```javascript
// pages/products.js
import { useState } from 'react';
import { useRouter } from 'next/router';

function Products({ products, selectedCategory }) {
  const [category, setCategory] = useState(selectedCategory);
  const router = useRouter();

  const handleCategoryChange = (event) => {
    const newCategory = event.target.value;
    setCategory(newCategory);
    router.push(`/products?category=${newCategory}`);
  };

  return (
    <div>
      <select value={category} onChange={handleCategoryChange}>
        <option value="electronics">Electronics</option>
        <option value="books">Books</option>
        <option value="clothing">Clothing</option>
      </select>

      {products && products.length > 0 ? (
        <ul>
          {products.map(product => (
            <li key={product.id}>{product.name}</li>
          ))}
        </ul>
      ) : (
        <p>No products found for this category.</p>
      )}
    </div>
  );
}


Products.getInitialProps = async ({ query }) => {
  const selectedCategory = query.category || 'electronics'; //default to electronics
  try {
      const res = await fetch(`https://api.example.com/products?category=${selectedCategory}`);
      const products = await res.json();
      return { products, selectedCategory };
  } catch (error) {
      console.error("Error fetching products:", error);
      return { products: [], selectedCategory };
  }
};


export default Products;

```

In this example, `handleCategoryChange` updates the local state and, more importantly, programmatically navigates to `/products?category=selectedvalue`. Then `getInitialProps` on the server, receives that value through the `query` object. It fetches the correct data based on the selected category and passes it down as props. Notice the default assignment in case the parameter is not present on initial load.

**Example 2: Handling Multiple Filters**

Let's say you need to handle multiple filters, not just category, but perhaps price range as well.

```javascript
// pages/search.js
import { useState } from 'react';
import { useRouter } from 'next/router';

function SearchResults({ results, filters }) {
  const [category, setCategory] = useState(filters.category || 'all');
  const [priceRange, setPriceRange] = useState(filters.priceRange || 'any');
  const router = useRouter();


  const handleFilterChange = (event) => {
      const {name, value} = event.target;
      if (name === "category") {
          setCategory(value);
      } else {
          setPriceRange(value)
      }

    const newQuery = {
        ...router.query,
        category: name === "category" ? value : category,
        priceRange: name === "priceRange" ? value: priceRange
    };

    router.push({
        pathname: '/search',
        query: newQuery,
    });

};


  return (
    <div>
        <select name="category" value={category} onChange={handleFilterChange}>
            <option value="all">All</option>
            <option value="tech">Tech</option>
            <option value="home">Home</option>
        </select>
        <select name="priceRange" value={priceRange} onChange={handleFilterChange}>
            <option value="any">Any</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
        </select>

      {results && results.length > 0 ? (
        <ul>
          {results.map(result => (
            <li key={result.id}>{result.title}</li>
          ))}
        </ul>
      ) : (
        <p>No results found.</p>
      )}
    </div>
  );
}


SearchResults.getInitialProps = async ({ query }) => {
  const category = query.category || 'all';
  const priceRange = query.priceRange || 'any';

  try {
      const res = await fetch(`https://api.example.com/search?category=${category}&priceRange=${priceRange}`);
    const results = await res.json();
    return { results, filters: { category, priceRange } };
  } catch (error) {
    console.error("Error fetching search results:", error);
    return { results: [], filters: { category, priceRange } };
  }
};

export default SearchResults;
```

This code demonstrates how to handle multiple filters. We serialize them in the URL as query parameters, update them on each change and also retain the values in the state, and then in `getInitialProps`, we extract them and use them for the API call.

**Example 3: Handling Errors and No Data**

Lastly, it’s essential to handle situations where either an API call fails or no data matches a particular query. Here, we introduce some basic error handling.

```javascript
// pages/articles.js
import { useState } from 'react';
import { useRouter } from 'next/router';


function Articles({ articles, error, selectedTag }) {
  const [tag, setTag] = useState(selectedTag);
  const router = useRouter();

  const handleTagChange = (event) => {
    const newTag = event.target.value;
    setTag(newTag);
    router.push(`/articles?tag=${newTag}`);
  };


  if (error) {
      return <p>An error occurred: {error}</p>;
  }


  return (
    <div>
      <select value={tag} onChange={handleTagChange}>
        <option value="technology">Technology</option>
        <option value="science">Science</option>
        <option value="art">Art</option>
      </select>
        {articles && articles.length > 0 ? (
           <ul>
             {articles.map(article => (
                 <li key={article.id}>{article.title}</li>
             ))}
          </ul>
         ) : (
           <p>No articles found for this tag.</p>
        )}
    </div>
  );
}



Articles.getInitialProps = async ({ query }) => {
    const selectedTag = query.tag || 'technology';
  try {
        const res = await fetch(`https://api.example.com/articles?tag=${selectedTag}`);

      if (!res.ok) {
        throw new Error(`API call failed with status: ${res.status}`);
      }


    const articles = await res.json();
    return { articles, selectedTag, error: null };
  } catch (error) {
    console.error("Failed to fetch articles:", error);
    return { articles: [], selectedTag, error: error.message };
  }
};

export default Articles;
```

In this version, we add a basic check for the HTTP status code in `getInitialProps`. If the API call fails, we throw an error and send it back to the page as a prop. This prevents the page from showing a blank screen and gives the user feedback about the error.

**Further Reading**

For more comprehensive understanding of data fetching strategies in React and specifically Next.js, I would recommend these resources:

*   **“Server-Side Rendering with React” by Marc L. Klemp.** This is a classic deep dive into SSR concepts in React, essential for comprehending how `getInitialProps` and similar server-side mechanisms function.
*   **The official Next.js Documentation** (especially older versions if you are working on a legacy codebase.) The Next.js documentation is the gold standard for up-to-date information and detailed guides on data fetching strategies. Check both `getInitialProps` documentation and more modern approaches as well. It provides excellent explanations and practical examples.
*   **"Full Stack Web Development with React" by Robin Wieruch.** This book is a good overview of React with more emphasis on the practical aspects. The data fetching strategies are very well laid out for the server rendered environment.

In conclusion, while `getInitialProps` might not be the recommended approach for new Next.js projects, understanding how it works is invaluable for working with legacy codebases and grasping the evolution of data fetching. The key lies in effectively serializing your select box value into the URL as a query parameter, and then utilizing this parameter within `getInitialProps` to tailor your API requests. These three examples should provide a solid foundation to start with, and with some additional research, and these resources you'll be very well equipped to handle most similar challenges.
