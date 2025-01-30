---
title: "How can embedded SQL C with rowset cursors be compiled on DB2/AIX64 11.1.4.6?"
date: "2025-01-30"
id: "how-can-embedded-sql-c-with-rowset-cursors"
---
Specifically, what are the pitfalls and solutions for compilation problems encountered?

Compiling embedded SQL with rowset cursors on DB2/AIX64 11.1.4.6 presents specific challenges related to precompiler configuration and the handling of dynamic memory allocation for the rowsets themselves. The primary issue stems from the interaction between the DB2 precompiler’s default behavior, its interpretation of embedded SQL constructs, and the underlying AIX64 operating system's memory management. A particular problem I consistently encountered involves the precompiler's inability to accurately derive data type and size information for rowset structures, resulting in compile-time errors. My experience with a large-scale data migration project relying heavily on bulk retrieval methods via rowsets underscores the importance of understanding and mitigating these issues.

The core problem centers around the precompiler's need for explicit structure definitions when dealing with rowset cursors. By default, the DB2 precompiler is less aggressive in automatically inferring structures than typical C compilers.  A conventional singleton select statement typically allows the precompiler to create a structure implicitly, but with rowsets, the lack of explicit structure declaration creates ambiguities, leading to compilation errors. For example, the compiler cannot definitively size the SQLDA (SQL Descriptor Area) nor the host variable structure to contain the number of rows returned by the cursor without specific instructions. This requires the C programmer to define specific struct data types that map directly to the cursor's columns. Failing to do so results in errors, frequently manifesting as “SQLCODE -104” or “Data type mismatch” messages.

Furthermore, the AIX64 architecture’s memory model, specifically concerning data alignment and stack limitations, plays a secondary role in exacerbating compilation problems. Memory allocation for rowsets can quickly become demanding, and if not handled correctly, it can lead to runtime segmentation faults or compiler errors related to insufficient memory. For instance, when dealing with multiple cursors and large data volumes, stack allocations might become insufficient, or the compiler may have problems with data alignment across the architecture. These issues become especially prevalent in embedded SQL code that does not rigorously handle cursor closure or dynamic memory release, potentially leading to resource exhaustion during compilation, let alone runtime.

To address these issues, I've found the following strategies and techniques to be effective:

**1. Explicitly Define Rowset Structures:** The first crucial step is to explicitly define the structure corresponding to the cursor's result set. I use a `typedef` declaration to establish this structure before declaring the cursor in my embedded SQL code. This provides type information for the host variable that will hold the cursor data.

```c
#include <stdio.h>
#include <sql.h>
#include <sqlcli.h>

// Define the structure to hold a row
typedef struct {
    SQLINTEGER col1;
    SQLVARCHAR col2[50];
    SQLDATE col3;
} my_row;

// Declare the structure to hold the rowset
typedef struct {
   my_row data[100];
   SQLINTEGER num_rows;
} my_rowset;

EXEC SQL BEGIN DECLARE SECTION;
  my_rowset sql_rowset;
  SQLINTEGER row_index;
  SQLINTEGER fetch_size;
EXEC SQL END DECLARE SECTION;

int main() {
    // ... DB Connection code (omitted for brevity) ...
    fetch_size = 100;

    EXEC SQL DECLARE cur1 CURSOR FOR
      SELECT col1, col2, col3 FROM mytable;

    EXEC SQL OPEN cur1;

    do {
        EXEC SQL FETCH cur1 FOR :fetch_size ROWS INTO :sql_rowset;
        if(sqlca.sqlcode != 0 && sqlca.sqlcode != 100) {
          printf("Error during fetch: SQLCODE %ld, SQLSTATE: %s\n", sqlca.sqlcode, sqlca.sqlstate);
            break;
        }
        if (sqlca.sqlcode == 0 ) {
            for (row_index = 0; row_index < sql_rowset.num_rows; row_index++) {
                printf("Col1: %ld, Col2: %s, Col3: %s\n",
                 sql_rowset.data[row_index].col1,
                 sql_rowset.data[row_index].col2,
                 sql_rowset.data[row_index].col3);
            }
        }
    } while (sqlca.sqlcode == 0);

    EXEC SQL CLOSE cur1;

    // ... DB Disconnect code (omitted for brevity) ...
    return 0;
}
```

In this example, `my_row` is defined as a struct type matching the column types of the query result. `my_rowset` is then defined as an array of `my_row` structs along with a row count. This enables the precompiler to derive type information and memory sizes for the `sql_rowset` variable, eliminating ambiguity and compilation errors related to type mismatches or implicit type declarations.

**2. Utilize Dynamic Memory Allocation for Large Rowsets:** If handling particularly large datasets or if stack size is a concern, dynamically allocating memory using `malloc` and `free` becomes necessary. This can prevent stack overflows and improve runtime performance, but demands careful memory management practices.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sql.h>
#include <sqlcli.h>

// Define the structure to hold a row
typedef struct {
    SQLINTEGER col1;
    SQLVARCHAR col2[50];
    SQLDATE col3;
} my_row;

// Define the structure to hold the rowset
typedef struct {
   my_row *data;
   SQLINTEGER num_rows;
   SQLINTEGER capacity;
} my_rowset_dyn;

EXEC SQL BEGIN DECLARE SECTION;
  my_rowset_dyn *sql_rowset_dyn;
  SQLINTEGER row_index;
  SQLINTEGER fetch_size;
EXEC SQL END DECLARE SECTION;

int main() {
    // ... DB Connection code (omitted for brevity) ...

    fetch_size = 100;
    sql_rowset_dyn = (my_rowset_dyn *) malloc(sizeof(my_rowset_dyn));
    sql_rowset_dyn->data = (my_row *) malloc(sizeof(my_row) * fetch_size);
    sql_rowset_dyn->capacity = fetch_size;

    EXEC SQL DECLARE cur1 CURSOR FOR
      SELECT col1, col2, col3 FROM mytable;

    EXEC SQL OPEN cur1;

    do {
        EXEC SQL FETCH cur1 FOR :fetch_size ROWS INTO :sql_rowset_dyn->data, :sql_rowset_dyn->num_rows;
         if(sqlca.sqlcode != 0 && sqlca.sqlcode != 100) {
          printf("Error during fetch: SQLCODE %ld, SQLSTATE: %s\n", sqlca.sqlcode, sqlca.sqlstate);
            break;
        }
        if (sqlca.sqlcode == 0 ) {
           for (row_index = 0; row_index < sql_rowset_dyn->num_rows; row_index++) {
                printf("Col1: %ld, Col2: %s, Col3: %s\n",
                 sql_rowset_dyn->data[row_index].col1,
                 sql_rowset_dyn->data[row_index].col2,
                 sql_rowset_dyn->data[row_index].col3);
            }
        }
    } while (sqlca.sqlcode == 0);

    EXEC SQL CLOSE cur1;
     free(sql_rowset_dyn->data);
     free(sql_rowset_dyn);
    // ... DB Disconnect code (omitted for brevity) ...
    return 0;
}
```

This revised code demonstrates how to dynamically allocate memory for the rowset structure, enhancing flexibility and addressing potential stack size limitations. Important here is the freeing of allocated memory after usage with the free command.

**3. Utilize `SQL_ATTR_ROW_ARRAY_SIZE` and `SQL_ROW_STATUS_ARRAY`:** When working with ODBC CLI or similar APIs that have support for SQL statements, the `SQL_ATTR_ROW_ARRAY_SIZE` attribute is particularly useful. This allows you to set the row array size and the precompiler can derive the correct memory size. The corresponding `SQL_ROW_STATUS_ARRAY` is used to check if a row is valid. These attributes need to be configured appropriately during the initial database setup and connection. This is best illustrated with a slightly abstracted snippet of code because the full database connection handling adds excessive noise.

```c
#include <stdio.h>
#include <stdlib.h>
#include <sql.h>
#include <sqlcli.h>

// Define the structure to hold a row
typedef struct {
    SQLINTEGER col1;
    SQLVARCHAR col2[50];
    SQLDATE col3;
} my_row;

// Define the structure to hold the rowset
typedef struct {
   my_row *data;
   SQLINTEGER num_rows;
} my_rowset_cli;

// ... (Abstracted DB connection handle and environment handle)
SQLHANDLE hStmt;
SQLINTEGER rowStatus[100];

int main() {
   SQLRETURN rc;
   SQLINTEGER fetch_size = 100;
    my_rowset_cli sql_rowset_cli;

    sql_rowset_cli.data = malloc(sizeof(my_row) * fetch_size);
    // ... (Database connection code - omitted for brevity)...

    rc = SQLAllocHandle(SQL_HANDLE_STMT, hDbc, &hStmt);
    if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      printf("Error allocating statement handle.\n");
       //Handle error gracefully
    }

    // Set row array size
    SQLSetStmtAttr(hStmt, SQL_ATTR_ROW_ARRAY_SIZE, (SQLPOINTER)fetch_size, 0);
    SQLSetStmtAttr(hStmt, SQL_ATTR_ROW_STATUS_PTR, rowStatus, 0);

    // ...Prepare and execute query...

   SQLBindCol(hStmt, 1, SQL_C_LONG, &sql_rowset_cli.data[0].col1,0 , NULL);
   SQLBindCol(hStmt, 2, SQL_C_CHAR, &sql_rowset_cli.data[0].col2, sizeof(sql_rowset_cli.data[0].col2), NULL);
   SQLBindCol(hStmt, 3, SQL_C_TYPE_DATE , &sql_rowset_cli.data[0].col3, 0, NULL);

   rc = SQLExecute(hStmt); // Execute the prepared SQL

    if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      printf("Error executing statement: SQLCODE %ld, SQLSTATE: %s\n", rc, sqlca.sqlstate);
       // Handle error gracefully
    }

  while (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO) {
      rc = SQLFetch(hStmt);

      if (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO) {
          SQLINTEGER rows_fetched;
           SQLGetStmtAttr(hStmt, SQL_ATTR_ROWS_FETCHED, &rows_fetched, 0, NULL);

          for(int i=0; i < rows_fetched; ++i){
              if(rowStatus[i] == SQL_ROW_SUCCESS){
                  printf("Col1: %ld, Col2: %s, Col3: %s\n",
                     sql_rowset_cli.data[i].col1,
                     sql_rowset_cli.data[i].col2,
                     sql_rowset_cli.data[i].col3);

              }
          }
       } else if( rc == SQL_NO_DATA){
          break; //No more rows to fetch
       } else {
           printf("Error during fetch: SQLCODE %ld, SQLSTATE: %s\n", rc, sqlca.sqlstate);
           break; //Fetch error occurred
       }
    }

   SQLFreeHandle(SQL_HANDLE_STMT, hStmt);
    free(sql_rowset_cli.data);
   // ... (Database disconnection and handle freeing code - omitted for brevity)...
   return 0;
}
```
This approach uses the `SQLSetStmtAttr` API to define the `SQL_ATTR_ROW_ARRAY_SIZE`, which instructs the CLI interface on the size of rowsets. By looping through the returned `rowStatus` array, the code knows which rows were actually fetched and valid. This is a very controlled and effective approach when using the DB2 CLI.

In summary, when compiling embedded SQL C code utilizing rowset cursors on DB2/AIX64 11.1.4.6, pay close attention to the explicit declaration of result set structures, be cautious regarding stack sizes, and be mindful about dynamic memory allocation. Utilizing the CLI API when possible is beneficial due to its specific controls. For further in-depth exploration of these topics, I recommend reviewing the IBM DB2 documentation for embedded SQL programming, as well as resources covering ODBC CLI usage and memory management practices specific to the AIX64 operating system. These sources provide a comprehensive understanding of both the precompilation process and the runtime environment, aiding in resolving compilation issues and optimizing embedded SQL applications using rowsets.
