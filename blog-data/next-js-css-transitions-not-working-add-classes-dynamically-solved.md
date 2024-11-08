---
title: "Next.js CSS Transitions Not Working?  Add Classes Dynamically, Solved!"
date: '2024-11-08'
id: 'next-js-css-transitions-not-working-add-classes-dynamically-solved'
---

```javascript
const { useState } = React;

const Modal = () => {
  const [activeLocation, setActiveLocation] = useState("");

  const handleLocationClick = () => {
    setActiveLocation("Example");
  };

  return (
    <div className="flex flex-col w-[600px]">
      <div className="flex flex-col text-hsr-dark font-medium grow">
        <ul className="list-disc list-inside">
          <li
            onClick={handleLocationClick}
            className={`${
              activeLocation === "Example" ? "active-location-item" : ""
            } location-item cursor-pointer text-2xl p-10 outline outline-2 outline-transparent  mb-1`}
          >
            Example
          </li>
        </ul>
      </div>
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('app')).render(<Modal/>);

tailwind.config = {
  theme: {
    extend: {
      colors: {
        "hsr-dark": "#404040",
        "hsr-grey": "#BFBFBF",
      },
    },
  },
};
```

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js" integrity="sha512-8Q6Y9XnTbOE+JNvjBQwJ2H8S+UV4uA6hiRykhdtIyDYZ2TprdNmWOUaKdGzOhyr4dCyk287OejbPvwl7lrfqrQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js" integrity="sha512-MOCpqoRoisCTwJ8vQQiciZv0qcpROCidek3GTFS6KTk2+y7munJIlKCVkFCYY+p3ErYFXCjmFjnfTTRSC1OHWQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script src="https://cdn.tailwindcss.com/3.3.2"></script>

<div id="app"></div>

<style type="text/tailwindcss">
:root {
  --foreground-rgb: 0, 0, 0;
  --background-start-rgb: 214, 219, 220;
  --background-end-rgb: 255, 255, 255;
}

.location-item {
  background-image: linear-gradient(
    to right,
    theme("colors.hsr-dark") 50%,
    theme("colors.hsr-grey") 50%
  );
  background-size: 200% 100%;
  background-position: right bottom;
  transition: all 0.75s;
}

.active-location-item {
  background-position: left bottom;
}
</style>
``` 

