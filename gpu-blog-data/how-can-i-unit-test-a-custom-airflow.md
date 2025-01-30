---
title: "How can I unit test a custom Airflow operator using MagicMock in Airflow 1.10.3?"
date: "2025-01-30"
id: "how-can-i-unit-test-a-custom-airflow"
---
Airflow 1.10.3, while predating some of the more refined testing utilities found in later versions, still accommodates effective unit testing of custom operators through the strategic use of `unittest.mock.MagicMock`. The critical constraint lies in the need to simulate the operator’s execution environment, particularly its interaction with Airflow's core context (e.g., `ti` for task instance, `dag_run`) and potential dependencies. The absence of a genuine Airflow execution environment during unit testing requires reliance on mocking for these interactions.

A custom Airflow operator encapsulates specific logic, often interfacing with external systems or databases. Unit testing, therefore, focuses on verifying the correctness of this core logic, isolating it from the broader complexities of a full Airflow DAG run. I have frequently encountered situations where custom operator logic, reliant on context injected at runtime, was difficult to validate without properly simulating that context.

The core principle when using `MagicMock` for testing Airflow operators is to replicate the expected behavior of the Airflow context and its injected objects. We do not test the core Airflow infrastructure; instead, we focus solely on our operator's internal mechanisms. We create `MagicMock` objects that mimic the behavior of objects like `ti`, `dag_run`, or external API clients called within our operator. This allows us to verify that the operator interacts with these mock objects as intended, without requiring an actual Airflow execution.

Let's consider a hypothetical custom operator designed to call an external API, fetch data, and potentially write it to an xcom. Below I will outline a basic implementation of such an operator and its accompanying unit tests.

**Example 1: A Simple Custom Operator**

First, I will show a simple operator that utilizes an external API and stores results as XCom.

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import requests

class ExternalApiOperator(BaseOperator):
    @apply_defaults
    def __init__(self, api_url, api_param, **kwargs):
        super().__init__(**kwargs)
        self.api_url = api_url
        self.api_param = api_param

    def execute(self, context):
        response = requests.get(self.api_url, params={'key': self.api_param})
        response.raise_for_status()
        data = response.json()
        context['ti'].xcom_push(key='api_data', value=data)
        return data
```

This `ExternalApiOperator` takes an API URL and parameter as inputs. Within `execute`, it calls the API, parses the JSON response and stores it in XCom before returning the parsed data. To unit test this, we will use `MagicMock` to simulate the `requests` library and the `context` dictionary.

**Example 2:  Unit Test of the Operator**

Here’s how I might structure a unit test for `ExternalApiOperator`, utilizing `unittest.mock.MagicMock` to create isolated and controllable conditions:

```python
import unittest
from unittest.mock import MagicMock, patch
from your_module import ExternalApiOperator # Assuming your_module.py is where ExternalApiOperator is defined

class TestExternalApiOperator(unittest.TestCase):

    def test_execute_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status.return_value = None

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_response

        mock_context = {'ti': MagicMock()}

        operator = ExternalApiOperator(api_url='test_url', api_param='test_param', task_id='test_task')

        with patch('your_module.requests', mock_requests): # patching the requests import in operator's module.
            result = operator.execute(mock_context)


        self.assertEqual(result, {"result": "success"})
        mock_requests.get.assert_called_once_with('test_url', params={'key': 'test_param'})
        mock_context['ti'].xcom_push.assert_called_once_with(key='api_data', value={"result": "success"})

    def test_execute_api_error(self):

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("API Error")

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_response


        mock_context = {'ti': MagicMock()}

        operator = ExternalApiOperator(api_url='test_url', api_param='test_param', task_id='test_task')

        with patch('your_module.requests', mock_requests):
            with self.assertRaises(requests.exceptions.HTTPError):
                operator.execute(mock_context)


        mock_context['ti'].xcom_push.assert_not_called()
```
In the above code, `unittest` provides the framework for testing. `MagicMock` objects `mock_response` and `mock_requests` simulate the response from API and the request itself. The `mock_context` mocks the Airflow context with a mock `ti`. I am using `patch` decorator from the `unittest.mock` to make sure the instance of `requests` imported within the operator is replaced by the mocked instance, so that the test can isolate the operator logic.

The `test_execute_success` checks that the API response is correctly parsed and returned and that the xcom_push operation has been invoked. Conversely, the `test_execute_api_error` verifies that HTTP error exception is handled correctly and xcom push does not happen when the API request fails. Both tests use the assert methods provided by `unittest` to verify the behavior.

**Example 3: Testing with more complex context interaction**

Let’s extend our hypothetical example and imagine an operator that relies on a value extracted from the DAG's configuration at runtime. In such a scenario, the `context` passed to `execute` would contain more information than just the `ti` object. We might access a DAG run property like this:

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import requests

class ConfigurableApiOperator(BaseOperator):
    @apply_defaults
    def __init__(self, api_url, api_param_template, **kwargs):
         super().__init__(**kwargs)
         self.api_url = api_url
         self.api_param_template = api_param_template

    def execute(self, context):
        dag_run_config = context['dag_run'].conf
        effective_param = self.api_param_template.format(config_value = dag_run_config.get('config_value', 'default_value'))
        response = requests.get(self.api_url, params={'key': effective_param})
        response.raise_for_status()
        data = response.json()
        context['ti'].xcom_push(key='api_data', value=data)
        return data
```

The core logic now incorporates pulling a config value passed to the DAG execution to construct the query parameter.

The unit test for this version would need to mock the `dag_run` object within the context as follows:

```python
import unittest
from unittest.mock import MagicMock, patch
from your_module import ConfigurableApiOperator # Assuming your_module.py is where ConfigurableApiOperator is defined

class TestConfigurableApiOperator(unittest.TestCase):

    def test_execute_with_config(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status.return_value = None

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_response

        mock_dag_run = MagicMock()
        mock_dag_run.conf = {"config_value": "test_config"}

        mock_context = {'ti': MagicMock(), 'dag_run': mock_dag_run}

        operator = ConfigurableApiOperator(
            api_url='test_url',
            api_param_template='param_{config_value}',
            task_id='test_task'
        )

        with patch('your_module.requests', mock_requests):
            result = operator.execute(mock_context)

        self.assertEqual(result, {"result": "success"})
        mock_requests.get.assert_called_once_with('test_url', params={'key': 'param_test_config'})
        mock_context['ti'].xcom_push.assert_called_once_with(key='api_data', value={"result": "success"})

    def test_execute_with_default_config(self):
         mock_response = MagicMock()
         mock_response.json.return_value = {"result": "success"}
         mock_response.raise_for_status.return_value = None

         mock_requests = MagicMock()
         mock_requests.get.return_value = mock_response

         mock_dag_run = MagicMock()
         mock_dag_run.conf = {} #empty config

         mock_context = {'ti': MagicMock(), 'dag_run': mock_dag_run}

         operator = ConfigurableApiOperator(
             api_url='test_url',
             api_param_template='param_{config_value}',
             task_id='test_task'
         )

         with patch('your_module.requests', mock_requests):
             result = operator.execute(mock_context)

         self.assertEqual(result, {"result": "success"})
         mock_requests.get.assert_called_once_with('test_url', params={'key': 'param_default_value'})
         mock_context['ti'].xcom_push.assert_called_once_with(key='api_data', value={"result": "success"})
```

Here, two tests are set up; one with explicit configuration, and the second to verify a fallback to default config if the `dag_run.conf` does not provide the expected value. The `mock_dag_run` is constructed to simulate the runtime context of the dag run. This demonstrates that even intricate operator interactions can be tested effectively by mocking out the expected inputs using `MagicMock` objects.

In practical scenarios, it is important to consider edge cases such as empty API responses, various HTTP error codes, and potentially more sophisticated interaction with the task instance’s methods. We also need to ensure that mocks are correctly configured and all needed methods within the mock objects are mocked to reflect the real objects.

**Resource Recommendations**

For more in-depth information regarding testing in Python, I suggest focusing on the official Python documentation on the `unittest` module and, critically, the `unittest.mock` library. Understanding the nuances of `MagicMock` and its variations (`Mock`, `patch`) is essential. Additionally, studying examples of unit tests from prominent Python libraries or frameworks can greatly enhance comprehension. For Airflow specific testing strategies, the official Airflow documentation provides helpful examples, even though it may be more relevant for later versions; the general principles often apply to 1.10.3 as well. Finally, exploring books focused on Python testing methodologies (e.g., “Test-Driven Development with Python” by Harry Percival) provides a broader context.
