INOUT = '''Given a problem statement and a Python program that aims to solve it, your task is to **write test cases** that uncover any potential bugs.

### **Task Overview**

You should output a JSON object that contains a list of test cases for the provided program. Each test case should include:
1. **input_str**: The exact text to feed into stdin.
2. **expected_output**: The exact text the program should print.

We will run each test by feeding `input_str` into the program and comparing its stdout against `expected_output`.

### **Required Format**

```json
[
  {{
    "input_str": "input 1",
    "expected_output": "output 1"
  }},
  {{
    "input_str": "input 2",
    "expected_output": "output 2"
  }}
  // ... up to 20 test cases total
]
```

### **Constraints**

* Generate **1–20** test cases.
* Don't include comments or extra fields in the JSON.
* Each input_str and expected_output must be a valid JSON string.'''


FUNCTION = '''Given a problem statement and a Python function that aims to solve it, your task is to **write test cases** that uncover any potential bugs.

### **Task Overview**

You should output a JSON object that contains a list of test cases for the provided function. Each test case should include:
1. **input_args**: A dict mapping the target function's parameter names to test values.
2. **expected_output**: The raw output the function should return when run with `input_args` as kwargs.

We will run each test by feeding `input_args` into the function as kwargs and comparing the returned value against `expected_output`.

### **Required Format**

```json
[
  {{
    "input_args": input 1,
    "expected_output": output 1
  }},
  {{
    "input_args": input 2,
    "expected_output": output 2
  }}
  // ... up to 20 test cases total
]
```

### **Constraints**

* Generate **1–20** test cases.
* Don't include comments or extra fields in the JSON.
* Each input_args and expected_output must be a valid JSON object.'''


MESSAGE_TEMPLATE = {
    "input/output": [{
        "role": "user",
        "content": INOUT + """\n\nThe problem is as follows:\n{description}\n\nAnd the program is as follows:\n```python\n{testing_code}\n```""",
    }],
    "functional": [{
        "role": "user",
        "content": FUNCTION + """\n\nThe problem is as follows:\n{description}\n\nAnd the program is as follows:\n```python\n{testing_code}\n```""",
    }],
}