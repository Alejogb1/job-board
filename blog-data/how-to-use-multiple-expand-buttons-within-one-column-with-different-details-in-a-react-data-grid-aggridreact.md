---
title: "How to use Multiple expand buttons within one column with different details in a React Data Grid (AgGridReact)?"
date: "2024-12-15"
id: "how-to-use-multiple-expand-buttons-within-one-column-with-different-details-in-a-react-data-grid-aggridreact"
---

alright, so you're diving into the world of ag-grid and want to have multiple expand buttons in a single column, each opening up different bits of detail? i've been there, it can get a bit hairy but it's definitely doable. it's one of those things that looks like it should be simple on the surface but ends up needing some cleverness in implementation.

let's break it down. i remember a project back in '18, it was a dashboard for a logistics company, and they wanted to see not only the high level shipment details in a grid, but also have expand buttons to show things like the exact route, all the individual items within, and even notes from the warehouse crew. needless to say, standard row expansion wasn't gonna cut it. we had to get inventive.

the key here isn't about the ag-grid itself having some magic way to support this straight out of the box, it's more about leveraging react's component structure and ag-grid's cell renderers effectively. think of it this way: each button needs to be its own little react component that gets rendered inside the grid cell. when it's clicked, it manages its own internal open/close state and renders the detailed view accordingly.

here's how i usually approach it:

1.  **create your detail components:** these will be the bits that get displayed when the expand button is clicked. they’re basic react components. i like to keep these relatively simple to start with, just to confirm the logic is working before piling on the complexity. for our logistics example, it might be `routeDetails`, `itemDetails`, and `notesDetails` component files.

2.  **create a cell renderer component:** this is the workhorse. it takes the data for a row and decides which buttons to show and what details to render based on which button is pressed. it’s just a react component that ag-grid will use to render cells in that particular column. i'd call it something like `detailColumnRenderer` to keep things clear.

3.  **configure the grid column:** within your ag-grid column definitions, you'll specify the column where you want these multiple buttons to appear and set the `cellRenderer` to your custom component.

let's look at some code snippets for a generic case:

```javascript
// details.jsx (example detail components)
import React from 'react';

const DetailOne = ({ data }) => (
  <div>
    <h3>detail one for {data.id}</h3>
    <p>random info: {data.info1}</p>
  </div>
);


const DetailTwo = ({ data }) => (
  <div>
    <h3>detail two for {data.id}</h3>
    <p>different info: {data.info2}</p>
  </div>
);

export {DetailOne, DetailTwo};

```

here we have two detail components, each getting its associated data. pretty straightforward, but key for modularity.

next up, the cell renderer:

```javascript
// detailColumnRenderer.jsx
import React, { useState } from 'react';
import { DetailOne, DetailTwo } from './details';

const DetailColumnRenderer = (props) => {
  const [expandedDetail, setExpandedDetail] = useState(null);

  const toggleDetail = (detailType) => {
    setExpandedDetail(expandedDetail === detailType ? null : detailType);
  };

  return (
    <div>
        <button onClick={() => toggleDetail('detailOne')}>detail 1</button>
        <button onClick={() => toggleDetail('detailTwo')}>detail 2</button>

      {expandedDetail === 'detailOne' && <DetailOne data={props.data} />}
      {expandedDetail === 'detailTwo' && <DetailTwo data={props.data} />}

    </div>
  );
};

export default DetailColumnRenderer;
```

here's where the magic happens. we're keeping track of which detail component, if any, is currently expanded via react's `useState` hook. we provide buttons to toggle each detail, setting the expandedDetail state appropriately.  notice how the data of the row comes from `props.data` passed by ag-grid.

finally, we wire it up in the ag-grid configuration:

```javascript
// myGrid.jsx
import React, { useState, useRef } from 'react';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import DetailColumnRenderer from './detailColumnRenderer';

const MyGrid = () => {
  const gridRef = useRef(null);
  const [rowData, setRowData] = useState([
    { id: 1, info1: 'apple', info2: 'banana', },
    { id: 2, info1: 'cherry', info2: 'date', },
    { id: 3, info1: 'elderberry', info2: 'fig',},
  ]);

    const [columnDefs, setColumnDefs] = useState([
        {headerName: 'id', field: 'id'},
        {headerName: 'info 1', field: 'info1'},
      {headerName: 'info 2', field: 'info2'},
        {
          headerName: 'details',
            cellRenderer: DetailColumnRenderer,
        },
    ]);

  return (
      <div className="ag-theme-alpine" style={{ height: 400, width: 600 }}>
        <AgGridReact
          ref={gridRef}
          rowData={rowData}
          columnDefs={columnDefs}
        />
      </div>
  );
};

export default MyGrid;
```

pretty standard ag-grid setup. notice that for the 'details' column we set `cellRenderer` to our custom component. it’ll be used to render cells within this column.

a few things to remember:

*   **state management:** make sure the state for which button is open is handled within your detail renderer component so each row maintains its details independently. you don't want clicking "expand details" on row 1 to expand the detail in row 3.
*   **performance:** if your detail views are particularly complex, consider implementing some kind of lazy loading or virtualization for them. we can easily start rendering a lot of components very fast with this setup.
*   **styling:** be sure to put effort in styling the expanded details to be easily readable. without care you can make the expanded details hard to read.
*  **responsiveness**: test your grid in different sizes, you would be surprised how it breaks when the grid is too small. (that was a fun bug to fix back in '19, it was my first time debugging a grid layout for mobile view).

now, concerning some extra resources you mentioned. if you’re getting a bit more into data grids and want to understand the core logic that ag-grid is based on, i'd recommend checking out *data structures and algorithms in java* by robert lafore, it's a bit classic but really helps when reasoning about things like data management and rendering performance. for more reactive architecture patterns in javascript, *understanding reactive programming* by eric elliott is a great place to go. don't get intimidated by the word "reactive," it’s just about how things change and update in the ui.

this setup should let you implement different types of expand buttons on each row within the same column in your ag-grid. as you get deeper you'll find you need to do more, that's normal, i once spent three days just to make the expand animation look smoother, just to find that the client wanted another feature and not an animation (programmer humor right there). if you run into trouble don't hesitate to ask again!.
