---
title: "\"Stuck on ORA-06533: 'Subscript beyond count' - Help!\""
date: '2024-11-08'
id: 'stuck-on-ora-06533-subscript-beyond-count-help'
---

```sql
declare
  type salaryvarray is varray(6) of customers.salary%type;
  salary_array salaryvarray:=salaryvarray();
  c_salary customers.salary%type;
  i integer(2);
  counter number(2);
begin
  select count(*) into counter from customers;
  for i in 1..counter loop
    salary_array.extend; -- Extend for each value.
    select salary into c_salary from customers where id =i;
    salary_array(i):=c_salary;
  end loop;
end;
/
```
