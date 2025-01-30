---
title: "How to resolve BashOperator errors running two CLI commands in Airflow?"
date: "2025-01-30"
id: "how-to-resolve-bashoperator-errors-running-two-cli"
---
The common pitfall when encountering issues with Airflow's `BashOperator` executing multiple commands stems from how the shell interpreter parses and executes those commands, often not behaving as expected regarding error handling and continuation. I've spent considerable time debugging DAGs failing in seemingly inexplicable ways due to this, and a careful examination of the process reveals several key areas of concern. Specifically, each string passed to the `bash_command` parameter of the `BashOperator` is ultimately treated as a single command line argument, and the entire string is executed using `bash -c`. If any single command within that string fails, the overall exit status of the shell invocation becomes the exit status of the `BashOperator`. However, the default behavior of `bash -c` means that commands after the failing one might still execute but might not affect the operator's success or failure state as a whole.

The primary issue stems from the way `bash -c` interprets multiple commands separated by `&&` and `;`. Using `&&` means that the following command will only execute if the preceding one succeeds (exit code 0). However, if one of the commands in such a chain fails, the `BashOperator` itself will report a failure. The challenge arises when one wants to execute multiple commands irrespective of whether the preceding one fails, or perform some cleanup actions regardless of the success or failure. Using the semicolon `;` allows commands to run irrespective of the status of preceding commands, but this, in turn, can mask underlying errors since the final exit status of the series is not necessarily representative of every command execution. Error handling, therefore, needs to be addressed explicitly within the command string passed to the `bash_command`.

One approach to managing this is through explicit conditional statements within the bash command. By using `set -e`, the script will exit immediately if any command fails. This is quite different from just checking the return status of a command using the `$?` variable (which indicates the success/failure of the last run process). It will force failure if any intermediate component in a chain has failed. To manage more complex scenarios, I often use explicit error capturing in combination with `set -e`. Capturing the exit code of a command can enable actions contingent upon the success or failure of other commands within the same shell invocation.

Here are some practical examples illustrating the problem and a few solutions I employ:

**Example 1: A Basic Failure Scenario**

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="bash_operator_fail",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task_fail = BashOperator(
        task_id="failing_command",
        bash_command="mkdir test_dir && rm -rf non_existent_dir",
    )
```

**Commentary:** In this example, the command tries to create a directory (`mkdir test_dir`) and then immediately tries to remove a non-existent one (`rm -rf non_existent_dir`). The first command will typically succeed, but the second will fail. This specific configuration of `rm` would not typically halt execution but returns a non-zero exit code; thus the BashOperator will also fail. This happens because the overall command returns a non-zero status despite the first command running correctly. This demonstrates the need for controlled error handling.

**Example 2: Error Handling with `set -e` and `&&`**

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="bash_operator_set_e",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task_set_e = BashOperator(
        task_id="set_e_command",
        bash_command="set -e; mkdir test_dir && rm -rf non_existent_dir",
    )
```

**Commentary:** Here, we add `set -e` at the beginning of the bash command. When `rm -rf non_existent_dir` fails, `set -e` ensures that the script will immediately exit, making the failure of the second command the final status of the shell invocation and thereby causing the Airflow `BashOperator` to fail immediately. This is usually the desired behavior for critical steps, stopping the rest of the script and highlighting an issue. Without `set -e`, depending on the nature of the failed command, the script may continue, potentially masking underlying issues.

**Example 3: Conditional Execution and Error Capture**

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="bash_operator_conditional",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task_conditional = BashOperator(
        task_id="conditional_command",
        bash_command="mkdir test_dir ; "
                     "if rm -rf non_existent_dir; then "
                     "echo 'Command succeeded';"
                     "else "
                     "echo 'Command failed with status $?';"
                     "exit 1; "
                     "fi",
    )
```

**Commentary:** This example demonstrates a more robust approach. It first executes `mkdir test_dir`. Then it attempts `rm -rf non_existent_dir`. Unlike example 2 where `set -e` would simply stop the script, this uses `if` to check the exit status of `rm -rf`. If successful, it prints a success message; if it fails, it prints an error message along with the exit status ($?) and explicitly exits with exit code 1. This enables specific behaviors on the basis of each command execution. It prevents the overall BashOperator from succeeding due to the first command, in the case of the second command’s failure.

This approach allows for detailed control over execution flow and better debugging capabilities within `BashOperator`. I would generally advise that relying simply on sequencing via `;` is usually problematic for any mission critical step because the exit status becomes too generic. Employing `set -e`, capturing and inspecting exit codes explicitly, coupled with conditional statements, are the cornerstones of reliable `BashOperator` command executions.

For further exploration, I would suggest diving deep into the documentation for `set` builtin in `bash`. You can also find valuable material in guides about Bash scripting in general, which will provide insights into command exit status handling, how `&&` and `||` work. Airflow documentation itself provides a good introduction to the `BashOperator`, but doesn't necessarily cover more nuanced examples.  Lastly, exploring other Airflow operators, especially `PythonOperator`, will provide an alternative if pure bash scripting becomes overly complex.
