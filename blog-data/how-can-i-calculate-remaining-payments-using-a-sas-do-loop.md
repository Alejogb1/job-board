---
title: "How can I calculate remaining payments using a SAS DO loop?"
date: "2024-12-23"
id: "how-can-i-calculate-remaining-payments-using-a-sas-do-loop"
---

,  I recall a particularly tricky project back in my days with a financial analytics firm where we were dealing with a high volume of loan data, needing to project remaining payments across various loan types. Leveraging SAS's do loop effectively became absolutely crucial for both speed and accuracy. What initially looked like a straightforward task quickly revealed its nuances, particularly when factoring in variable interest rates and principal-only payments.

The core challenge, as I understand it, is to iterate through a loan's payment schedule, keeping track of outstanding principal, interest, and the number of payments remaining. The standard approach, and frankly the most efficient within SAS, is indeed to use a `do` loop in conjunction with conditional logic. Let me break this down with a couple of practical examples, drawing from my own past experiences.

First, consider a simplified scenario: a loan with a fixed interest rate and regular, equal payments. We'll work with the following parameters: `loan_amount`, `annual_interest_rate`, `loan_term_months` (total number of payments), and the `monthly_payment` calculated using a standard amortization formula (you can find this in any basic finance textbook, like 'Corporate Finance' by Ross, Westerfield, and Jaffe, if you need a refresher).

Here’s a basic SAS code snippet to illustrate this simple case:

```sas
data loan_schedule;
  loan_amount = 100000;
  annual_interest_rate = 0.05;
  loan_term_months = 60;

  monthly_interest_rate = annual_interest_rate/12;
  monthly_payment = (loan_amount * monthly_interest_rate) /
                    (1 - (1 + monthly_interest_rate)**(-loan_term_months));

  outstanding_principal = loan_amount;
  do payment_number = 1 to loan_term_months;
    interest_paid = outstanding_principal * monthly_interest_rate;
    principal_paid = monthly_payment - interest_paid;
    outstanding_principal = outstanding_principal - principal_paid;
    output;
    if outstanding_principal <= 0 then leave;
  end;
  keep payment_number outstanding_principal interest_paid principal_paid;
run;

proc print data=loan_schedule;
run;
```

In this example, the `do` loop iterates through each payment period. Inside the loop, we calculate `interest_paid` and `principal_paid`, adjust the `outstanding_principal`, and finally output the data. The `if outstanding_principal <= 0 then leave;` statement is crucial to prevent the loop from continuing unnecessarily once the loan is fully paid off. This produces a schedule with remaining principal for each payment period, and from that, you can derive the number of payments remaining by inspecting the last observation or by using subsequent data step aggregation.

Now, let's introduce a more complex scenario: what if the interest rate isn't fixed, but rather it changes at certain points throughout the loan’s life? This was a real headache in our projects; we had several variable rate mortgages where calculating outstanding balances mid-term was quite common. Here's how you could adjust the `do` loop in SAS to accommodate this:

```sas
data loan_schedule_variable;
  loan_amount = 100000;
  loan_term_months = 60;

  * Initial parameters;
  annual_interest_rate_1 = 0.05;
  change_month_1 = 24; *Interest changes after 24 months;
  annual_interest_rate_2 = 0.06;

  outstanding_principal = loan_amount;
  payment_number = 0;
    do while (outstanding_principal > 0);
        payment_number = payment_number+1;

        if payment_number <= change_month_1 then do;
            monthly_interest_rate = annual_interest_rate_1/12;
            monthly_payment = (loan_amount * monthly_interest_rate) /
                (1 - (1 + monthly_interest_rate)**(-loan_term_months));
        end;
        else do;
            monthly_interest_rate = annual_interest_rate_2/12;
            monthly_payment = (outstanding_principal * monthly_interest_rate) /
                (1 - (1 + monthly_interest_rate)**(- (loan_term_months - payment_number + 1))); *adjusted term for re-amortization;
        end;


    interest_paid = outstanding_principal * monthly_interest_rate;
    principal_paid = monthly_payment - interest_paid;
    outstanding_principal = outstanding_principal - principal_paid;
    output;

  end;

keep payment_number outstanding_principal interest_paid principal_paid;
run;

proc print data=loan_schedule_variable;
run;
```

In this second example, we introduced an `if-else` conditional block inside the `do` loop. Before `change_month_1`, the calculations are based on the initial interest rate. After that point, a new interest rate and also re-amortization is applied. I've used a `do while` instead of a `do to` to handle cases with potentially adjusted loan terms and handle re-amortization correctly. The recalculation of the `monthly_payment` is critical here, as it re-amortizes the remaining principal over the remaining term at the new rate. This illustrates how a SAS `do` loop can handle complex real-world scenarios with the addition of conditional logic. Note the `outstanding_principal` is used in the amortizing calculation in the `else` branch, to correctly adjust the remaining payments when the interest rate changes.

Finally, let's consider a case where some payments may be principal-only, as was the case with some of the investment vehicles we were working with. This often occurs when a lump sum payment is made, but the regular monthly payment schedule remains the same. In these cases, we had to explicitly model those irregular principal payments. Here’s how that can be incorporated:

```sas
data loan_schedule_principal;
  loan_amount = 100000;
  annual_interest_rate = 0.05;
  loan_term_months = 60;
  monthly_interest_rate = annual_interest_rate/12;
  monthly_payment = (loan_amount * monthly_interest_rate) /
                    (1 - (1 + monthly_interest_rate)**(-loan_term_months));

  outstanding_principal = loan_amount;
  * Additional principal payments;
    call symput("principal_payment_1_month", 12);
    call symput("principal_payment_1_amount", 10000);
    call symput("principal_payment_2_month", 36);
    call symput("principal_payment_2_amount", 5000);


  payment_number = 0;
  do while (outstanding_principal > 0);
    payment_number = payment_number +1;

    principal_payment=0;
    if payment_number=&principal_payment_1_month then principal_payment=&principal_payment_1_amount;
    else if payment_number=&principal_payment_2_month then principal_payment=&principal_payment_2_amount;

    interest_paid = outstanding_principal * monthly_interest_rate;
    principal_paid = monthly_payment - interest_paid;

    if principal_payment>0 then do;
        principal_paid = principal_paid + principal_payment;
    end;

    outstanding_principal = outstanding_principal - principal_paid;

    output;
  end;
  keep payment_number outstanding_principal interest_paid principal_paid;
run;

proc print data=loan_schedule_principal;
run;
```

In this last example, we've added symbolic macro variables for additional principal payments. Inside the loop, we check if the current payment number matches a month with an additional principal payment and adjust the principal paid accordingly, before updating the `outstanding_principal`.

For further reading, I’d recommend delving into resources on numerical methods; particularly, *Numerical Recipes: The Art of Scientific Computing* by Press et al. is an excellent resource that provides a good grounding in iterative calculations and the mathematics of amortization. Also, for specific financial applications and nuances within SAS, the SAS documentation for procedures like `PROC IML` and `PROC FCMP` is invaluable. These allow more advanced calculations but typically aren’t required for this basic task.

So, the key takeaway here is that the SAS `do` loop, combined with conditional logic and careful handling of variables, is incredibly flexible and powerful for calculating loan schedules and projecting remaining payments. It served me well when tackling large financial datasets, and I'm confident it will be beneficial for you. Remember to test your code thoroughly and adapt these examples to suit your specific use case; the devil is often in the details.
