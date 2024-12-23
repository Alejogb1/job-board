---
title: "How can I implement a multi-master-detail view using DevExpress components in C#?"
date: "2024-12-23"
id: "how-can-i-implement-a-multi-master-detail-view-using-devexpress-components-in-c"
---

Alright, let's tackle this multi-master-detail view challenge with DevExpress in C#. I've navigated these waters before, and I can tell you, it's a scenario that demands a careful architectural approach to avoid a tangled mess. The key is to maintain a clean separation of concerns, manage data binding effectively, and ensure a responsive user interface. Let's break it down step-by-step, focusing on a flexible and maintainable implementation.

The scenario I've encountered most often involved a system tracking customer orders. We had a primary grid displaying customers, a secondary grid showing their orders, and a third showing the items within each order. Naturally, users needed to navigate this hierarchy seamlessly, with appropriate detail views updating based on selections. I'll structure my explanation and code examples around this use case, as it’s quite illustrative of the problem you are describing.

**Core Principles and Approach**

Before diving into code, it’s important to establish some guiding principles. We need to ensure that our detail grids update only when a related master row is selected. This prevents unnecessary database queries and keeps the user experience fluid. We also need to consider how changes in one detail grid might impact others, for example, if a user edits a customer’s address, and how this might refresh downstream grids. Furthermore, we want to avoid tightly coupling UI elements and data access logic directly within the UI layer. This keeps the code modular and testable.

Essentially, what we're aiming for is a layered application design where the UI interacts with a service layer, which, in turn, manages our data operations. This pattern provides a structure that can scale effectively as your project grows in complexity.

**Code Example 1: Setting up the Primary Master Grid**

Let’s start by creating a primary master grid showing customers. In this case, we will use the DevExpress GridControl. Assume we have a `Customer` model class, a service named `CustomerService` with a method named `GetAllCustomers()`.

```csharp
using DevExpress.XtraGrid;
using DevExpress.XtraGrid.Views.Grid;
using System.Windows.Forms;
using System.Collections.Generic;

public partial class MainForm : Form
{
    private GridControl _customerGrid;
    private CustomerService _customerService;

    public MainForm()
    {
        InitializeComponent();
        _customerService = new CustomerService();
        InitializeCustomerGrid();
    }

    private void InitializeCustomerGrid()
    {
        _customerGrid = new GridControl();
        _customerGrid.Dock = DockStyle.Top;
        _customerGrid.Height = 200; // Give some visual space
        var customerView = new GridView(_customerGrid);
        _customerGrid.MainView = customerView;
        customerView.OptionsBehavior.Editable = false; // Prevent direct editing
        customerView.OptionsSelection.EnableAppearanceFocusedCell = false;
        customerView.OptionsSelection.MultiSelect = false;
        customerView.OptionsView.ShowGroupPanel = false;

        customerView.FocusedRowChanged += CustomerView_FocusedRowChanged;
        
        // Define columns (assume 'Name' and 'Id' are properties)
        customerView.Columns.Add(new DevExpress.XtraGrid.Columns.GridColumn{FieldName="Id", Caption="Customer ID", Visible=true});
        customerView.Columns.Add(new DevExpress.XtraGrid.Columns.GridColumn{FieldName="Name", Caption="Customer Name", Visible=true});

        Controls.Add(_customerGrid);
        LoadCustomerData();

        
    }

    private async void LoadCustomerData()
    {
        List<Customer> customers = await _customerService.GetAllCustomers();
        _customerGrid.DataSource = customers;

    }

    private void CustomerView_FocusedRowChanged(object sender, DevExpress.XtraGrid.Views.Base.FocusedRowChangedEventArgs e)
    {
      if (e.FocusedRowHandle >= 0){
          Customer selectedCustomer = _customerGrid.GetRow(e.FocusedRowHandle) as Customer;
          LoadOrderData(selectedCustomer?.Id ?? 0);
      }

    }


}

public class Customer
{
    public int Id { get; set; }
    public string Name { get; set; }
}

// Placeholder service
public class CustomerService
{
    public async Task<List<Customer>> GetAllCustomers()
    {
      // Simulate data loading
        await Task.Delay(100);
        return new List<Customer> { new Customer{Id = 1, Name = "Customer A"}, new Customer{Id = 2, Name = "Customer B"} };
    }
}
```

This snippet sets up the basic structure. We create a `GridControl`, bind the data, configure some basic visual settings, and handle the `FocusedRowChanged` event to trigger the loading of order data upon customer selection, passing the customer id.

**Code Example 2: Loading the Secondary Grid (Orders)**

Here, we add the secondary order grid which is bound to an `Order` model class and utilizes an `OrderService` to fetch data.

```csharp
using DevExpress.XtraGrid;
using DevExpress.XtraGrid.Views.Grid;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Threading.Tasks;

public partial class MainForm : Form
{
 //... previous code omitted

    private GridControl _orderGrid;
    private OrderService _orderService;


    private void InitializeOrderGrid()
    {
        _orderGrid = new GridControl();
         _orderGrid.Dock = DockStyle.Fill; // To fill the remaining space
        var orderView = new GridView(_orderGrid);
        _orderGrid.MainView = orderView;
        orderView.OptionsBehavior.Editable = false;
         orderView.OptionsSelection.EnableAppearanceFocusedCell = false;
        orderView.OptionsSelection.MultiSelect = false;
        orderView.OptionsView.ShowGroupPanel = false;

        orderView.FocusedRowChanged += OrderView_FocusedRowChanged;

        // Define columns (assume 'OrderId' and 'OrderDate' and 'CustomerId' are properties)
        orderView.Columns.Add(new DevExpress.XtraGrid.Columns.GridColumn{FieldName="OrderId", Caption="Order ID", Visible=true});
        orderView.Columns.Add(new DevExpress.XtraGrid.Columns.GridColumn{FieldName="OrderDate", Caption="Order Date", Visible=true});
        orderView.Columns.Add(new DevExpress.XtraGrid.Columns.GridColumn{FieldName="CustomerId", Caption="Customer ID", Visible=false});

        Controls.Add(_orderGrid); // Add it to the controls, assuming previous grids are docked

    }

    public async void LoadOrderData(int customerId) {
           if (_orderService == null) _orderService = new OrderService();
            List<Order> orders = await _orderService.GetOrdersByCustomerId(customerId);
            _orderGrid.DataSource = orders;
        }

    private void OrderView_FocusedRowChanged(object sender, DevExpress.XtraGrid.Views.Base.FocusedRowChangedEventArgs e){
        if (e.FocusedRowHandle >= 0){
           Order selectedOrder = _orderGrid.GetRow(e.FocusedRowHandle) as Order;
           LoadOrderItemData(selectedOrder?.OrderId ?? 0);
        }
    }
}
public class Order
{
    public int OrderId { get; set; }
    public DateTime OrderDate { get; set; }
    public int CustomerId { get; set; }
}
// Placeholder service
public class OrderService
{
    public async Task<List<Order>> GetOrdersByCustomerId(int customerId)
    {
        // Simulate data loading
         await Task.Delay(100);
        return new List<Order> {new Order{OrderId = 1, CustomerId = customerId, OrderDate = DateTime.Now.AddDays(-2)}, new Order{OrderId = 2, CustomerId= customerId, OrderDate = DateTime.Now.AddDays(-1)} };
    }
}
```

This second part builds upon the first, adding an order grid that updates based on the customer selected in the first grid. Note the `CustomerId` property in the `Order` class; it's essential for filtering based on the customer id. The `OrderView_FocusedRowChanged` event handles selection changes which leads to loading of the order item data in the next example.

**Code Example 3: Loading the Detail Grid (Order Items)**

Finally, we introduce the order items grid, representing the final tier in our master-detail hierarchy. It relies on an `OrderItem` model and an `OrderItemService`.

```csharp
using DevExpress.XtraGrid;
using DevExpress.XtraGrid.Views.Grid;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Threading.Tasks;

public partial class MainForm : Form
{
//... previous code omitted

    private GridControl _orderItemGrid;
    private OrderItemService _orderItemService;

   private void InitializeOrderItemGrid()
    {
        _orderItemGrid = new GridControl();
         _orderItemGrid.Dock = DockStyle.Bottom; // To place it beneath other grids
         _orderItemGrid.Height = 200;

        var orderItemView = new GridView(_orderItemGrid);
        _orderItemGrid.MainView = orderItemView;
        orderItemView.OptionsBehavior.Editable = false;
         orderItemView.OptionsSelection.EnableAppearanceFocusedCell = false;
        orderItemView.OptionsSelection.MultiSelect = false;
         orderItemView.OptionsView.ShowGroupPanel = false;


        // Define columns (assume 'ItemId', 'ItemName', and 'OrderId' properties)
        orderItemView.Columns.Add(new DevExpress.XtraGrid.Columns.GridColumn{FieldName="ItemId", Caption="Item ID", Visible=true});
        orderItemView.Columns.Add(new DevExpress.XtraGrid.Columns.GridColumn{FieldName="ItemName", Caption="Item Name", Visible=true});
        orderItemView.Columns.Add(new DevExpress.XtraGrid.Columns.GridColumn{FieldName="OrderId", Caption="Order ID", Visible=false});


        Controls.Add(_orderItemGrid);

    }


    public async void LoadOrderItemData(int orderId)
    {
         if (_orderItemService == null) _orderItemService = new OrderItemService();
         List<OrderItem> items = await _orderItemService.GetItemsByOrderId(orderId);
         _orderItemGrid.DataSource = items;

    }


}

public class OrderItem
{
    public int ItemId { get; set; }
    public string ItemName { get; set; }
    public int OrderId { get; set; }

}
// Placeholder service
public class OrderItemService
{
    public async Task<List<OrderItem>> GetItemsByOrderId(int orderId)
    {
     // Simulate data loading
     await Task.Delay(100);
        return new List<OrderItem>{ new OrderItem{OrderId = orderId, ItemId=1, ItemName="Item 1"}, new OrderItem {OrderId = orderId, ItemId=2, ItemName="Item 2"} };
    }
}
```

This final snippet provides the full multi-master-detail view. The `OrderItem` data is loaded when an order row is selected in the second grid. This establishes the cascading update behavior that is necessary for multi-level master-detail view implementation.

**Considerations and Further Reading**

This is a basic framework, and in a full application, you would need to handle things like:

*   **Error Handling:** Proper exception handling in data service layers is essential.
*   **Asynchronous Data Loading:** Using `async` and `await` is crucial for avoiding UI freezes during data operations.
*   **Data Persistence:** Implement proper saving of changes to underlying data sources.
*   **Data Binding:** DevExpress grids offer powerful data binding features. Investigate binding lists and properties for real-time updates.
*   **Customization:** Explore other DevExpress UI components, such as layout controls and editors, to tailor the UI to specific needs.

For deeper understanding, I would recommend the following resources:

*   **"Patterns of Enterprise Application Architecture" by Martin Fowler:** This provides fundamental guidance on designing robust and maintainable applications.
*   **DevExpress Documentation:** The DevExpress documentation and their support website are quite comprehensive and provide specific guidance on their control usage.
*   **Domain-Driven Design by Eric Evans:** This explores the concept of structuring code around the business logic and data domain which helps design scalable services.

Remember to prioritize layered architecture, asynchronous operations, and data binding. Start with a solid foundation using these basic examples and then incrementally expand to handle your specific business requirements. It's a journey; good luck, and don't hesitate to refer back to the basics!
