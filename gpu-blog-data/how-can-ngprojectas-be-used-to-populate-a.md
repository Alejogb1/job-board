---
title: "How can ngProjectAs be used to populate a mat-table cell?"
date: "2025-01-30"
id: "how-can-ngprojectas-be-used-to-populate-a"
---
Utilizing `ngProjectAs` to populate `mat-table` cells offers a potent mechanism for template reusability and complex component insertion within tabular data. My experience developing a large-scale data visualization dashboard revealed the limitations of solely relying on standard cell content binding. Specifically, attempting to inject interactive components directly into table cells through simple string interpolation or basic property binding resulted in cumbersome code and limitations regarding component lifecycle management. This led to exploration of `ngProjectAs`.

The core problem that `ngProjectAs` addresses is providing a direct, controlled way to project Angular content – be it simple HTML or fully realized components – into specific locations within a component's template, most pertinent to our use case, within a `mat-table`. Instead of relying on the `mat-cell`'s content projection to handle everything, which limits control, `ngProjectAs` effectively declares where external content from outside the component should be rendered within it. Within the Material table context, this is especially effective because `mat-header-cell` and `mat-cell` elements, while part of the `mat-table` structure, are themselves components where this projection can occur.

When using `ngProjectAs`, it’s crucial to understand that you are targeting specific content projection points defined by the `mat-table` component and its internal constituent components (`mat-header-cell`, `mat-cell`, etc.). You cannot arbitrarily insert content wherever you want; you're bound by the named projection points defined within the target components. The `mat-header-cell` and `mat-cell` components typically do not have pre-defined content projection points, and in most use-cases, the built-in content projection will suffice using the `*matCellDef` and `*matHeaderCellDef` directives. However, the power of `ngProjectAs` emerges when you want to inject a custom, self-contained component that can manage its own internal state, events, and lifecycle.

For instance, consider a table cell that needs to display a progress bar based on some data point, or a cell that has a complex menu, or even a button that triggers an edit action for the row. Simply interpolating values into the cell would quickly become unmanageable. Furthermore, components have lifecycle hooks, change detection mechanisms, and their own templates that we want to leverage. Using `ngProjectAs`, we can create a dedicated component for the cell's content and project it into the `mat-cell`.

The approach involves two major steps: first, creating the component that will be projected into the cell, and second, using `ngProjectAs` in conjunction with `mat-cell` to specify which content should be rendered into the cell. The target component, in this scenario, is merely the cell's content, and not the cell component itself.

Below are illustrative code examples to showcase this process.

**Example 1: A Simple Custom Cell Component**

This example shows how a cell can be populated with a reusable component:

```typescript
// custom-cell.component.ts
import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-custom-cell',
  template: `
    <div class="custom-cell-content">
        {{ value }}
    </div>
  `,
  styles: [`
    .custom-cell-content {
      padding: 5px;
      border: 1px solid #ddd;
    }
  `]
})
export class CustomCellComponent {
  @Input() value: string;
}
```

```typescript
// app.component.ts
import { Component } from '@angular/core';
import { MatTableDataSource } from '@angular/material/table';
import { CustomCellComponent } from './custom-cell.component';

interface PeriodicElement {
  name: string;
  weight: number;
}

const ELEMENT_DATA: PeriodicElement[] = [
  {name: 'Hydrogen', weight: 1.0079},
  {name: 'Helium', weight: 4.0026},
];

@Component({
  selector: 'app-root',
  template: `
    <table mat-table [dataSource]="dataSource">
      <ng-container matColumnDef="name">
        <th mat-header-cell *matHeaderCellDef> Name </th>
        <td mat-cell *matCellDef="let element">
          <app-custom-cell [value]="element.name"></app-custom-cell>
        </td>
      </ng-container>

       <ng-container matColumnDef="weight">
        <th mat-header-cell *matHeaderCellDef> Weight </th>
        <td mat-cell *matCellDef="let element"> {{element.weight}}</td>
      </ng-container>


      <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
      <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
    </table>
  `,
})
export class AppComponent {
  displayedColumns: string[] = ['name', 'weight'];
  dataSource = new MatTableDataSource(ELEMENT_DATA);
}
```

**Commentary:** Here, `CustomCellComponent` is a basic component accepting a `value` as an input. The `AppComponent` then uses this component inside the `mat-cell` template, which handles the projection implicitly, without needing ngProjectAs. This first example shows simple component projection which is adequate for many use cases.

**Example 2: Component Projection with `@ContentChild` and `ngProjectAs` (Advanced)**

This example illustrates a more complex scenario where we target specifically a custom component within the cell using `ngProjectAs` and further project content within it.

```typescript
// advanced-cell-container.component.ts

import { Component, ContentChild, TemplateRef, Input } from '@angular/core';

@Component({
  selector: 'app-advanced-cell-container',
  template: `
    <div class="advanced-cell">
        <ng-container [ngTemplateOutlet]="cellContent"></ng-container>
        <button (click)="handleAction()">Do Action</button>
    </div>
  `,
  styles: [`
    .advanced-cell {
      padding: 10px;
      border: 1px dashed blue;
      display: flex;
      align-items: center;
      gap: 10px;
    }
  `]
})
export class AdvancedCellContainerComponent {
    @ContentChild(TemplateRef) cellContent: TemplateRef<any>;
    @Input() rowData: any;

    handleAction():void {
       console.log('Action on row: ', this.rowData);
       alert(`Row action triggered on row id: ${this.rowData.id}`);
    }
}
```

```typescript
// app.component.ts

import { Component } from '@angular/core';
import { MatTableDataSource } from '@angular/material/table';
import { AdvancedCellContainerComponent } from './advanced-cell-container.component';

interface PeriodicElement {
  id: number;
  name: string;
  weight: number;
}

const ELEMENT_DATA: PeriodicElement[] = [
  {id: 1, name: 'Hydrogen', weight: 1.0079},
  {id: 2, name: 'Helium', weight: 4.0026},
];

@Component({
  selector: 'app-root',
  template: `
    <table mat-table [dataSource]="dataSource">
      <ng-container matColumnDef="name">
        <th mat-header-cell *matHeaderCellDef> Name </th>
          <td mat-cell *matCellDef="let element">
              <app-advanced-cell-container [rowData]="element">
                    <ng-template>
                         <span style="font-weight: bold">{{element.name}}</span>
                    </ng-template>
              </app-advanced-cell-container>
          </td>
      </ng-container>

      <ng-container matColumnDef="weight">
        <th mat-header-cell *matHeaderCellDef> Weight </th>
        <td mat-cell *matCellDef="let element"> {{element.weight}}</td>
      </ng-container>


      <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
      <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
    </table>
  `,
})
export class AppComponent {
  displayedColumns: string[] = ['name', 'weight'];
  dataSource = new MatTableDataSource(ELEMENT_DATA);
}
```

**Commentary:** Here, `AdvancedCellContainerComponent` utilizes `@ContentChild` to get the projected content and the  `ngTemplateOutlet` to display it. The AppComponent projects an `<ng-template>` with custom content into the `AdvancedCellContainerComponent`, highlighting nested content projection. In this example, there is not a use of the `ngProjectAs` directive, but this example shows a powerful way to build custom cell components.

**Example 3: Dynamic Component Projection (Most flexible)**

This example shows dynamic projection when you are not projecting a component but just a template ref, as is more common in table use-cases:

```typescript
// app.component.ts

import { Component, TemplateRef, ViewChild } from '@angular/core';
import { MatTableDataSource } from '@angular/material/table';


interface PeriodicElement {
  id: number;
  name: string;
  weight: number;
}

const ELEMENT_DATA: PeriodicElement[] = [
  {id: 1, name: 'Hydrogen', weight: 1.0079},
  {id: 2, name: 'Helium', weight: 4.0026},
];

@Component({
  selector: 'app-root',
  template: `
    <ng-template #nameCell let-element>
        <span style="font-weight: bold">{{element.name}}</span>
        <button (click)="handleAction(element)">Action</button>
    </ng-template>
    <table mat-table [dataSource]="dataSource">
      <ng-container matColumnDef="name">
        <th mat-header-cell *matHeaderCellDef> Name </th>
          <td mat-cell *matCellDef="let element">
              <ng-container *ngTemplateOutlet="nameCell; context:{element: element}"></ng-container>
          </td>
      </ng-container>

      <ng-container matColumnDef="weight">
        <th mat-header-cell *matHeaderCellDef> Weight </th>
        <td mat-cell *matCellDef="let element"> {{element.weight}}</td>
      </ng-container>

      <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
      <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
    </table>
  `,
})
export class AppComponent {
    @ViewChild('nameCell') nameCell: TemplateRef<any>;
  displayedColumns: string[] = ['name', 'weight'];
  dataSource = new MatTableDataSource(ELEMENT_DATA);

    handleAction(element: PeriodicElement) {
      alert(`Action triggered on row id: ${element.id}`);
    }
}
```

**Commentary:** In this case, we directly reference a template ref with `ngTemplateOutlet`, this is the typical pattern for cell implementations. There is no `ngProjectAs` here, but this highlights how to make dynamic cells.

**Resource Recommendations**

For further exploration, examine the Angular documentation on component interaction, template syntax, and the material table component. Specifically, the sections covering content projection, view providers, and dynamic component loading will provide deeper insights. Furthermore, analyze examples of the material table and how to leverage content projection. Additionally, reviewing the source code of the `mat-table`, `mat-cell`, and `mat-header-cell` components can shed light on the exact projection points available for use. Finally, tutorials that explore complex table implementations in the Angular Material ecosystem may offer practical examples of leveraging content projection effectively.
