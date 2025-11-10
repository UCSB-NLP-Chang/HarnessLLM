INOUT = '''Given a problem statement and a Python program that aims to solve it, your task is to **write a test harness** that uncovers any potential bugs.

### **Task Overview**

You will deliver **a single** code block to define functions that can be run by our framework to generate inputs, run the program, and validate its outputs.
Consider two categories of test cases:
- **Hardcoded cases**: Manually crafted input-output pairs that expose known or likely bugs.
- **Dynamic cases**: Programmatically generated inputs that stress-test the implementation (e.g., randomized, combinatorial, large or edge-case inputs).

### **Required Functions**

```python
from typing import List

def generate_input_1() -> List[str]:
    """
    Return between 1 and 4 valid input strings, each a complete stdin payload for the target program.
    Consider the following strategies:
      - Manually craft inputs that expose bugs.
      - Dynamically generate randomized, combinatorial, large, or edge-case inputs for stress testing.
    """
    # Your code here
    return input_list

def generate_input_2() -> List[str]:
    """
    Another function to return between 1 and 4 valid input strings.
    Employ a different strategy than previous input generation functions.
    """
    # Your code here
    return input_list

# You may add up to 3 more functions named generate_input_3(), generate_input_4(), etc.

def check_output(generated_input: str, captured_output: str) -> None:
    """
    Validate the output for a single generated input.
    Inputs:
        - generated_input: The input string passed to the target program.
        - captured_output: The exact stdout produced by the target program.
    
    Hints: When exact outputs are hard to predict, avoid asserting them. Instead, consider:
      - Check key properties or invariants, e.g., output is sorted, has correct length, matches a pattern, has correct value ranges, etc.
      - Compare against a simple brute-force implementation
    """
    # Your code here
```

### **Execution Flow**

1. The framework calls generate input functions to obtain a list of test strings.
2. For each string:
   * It runs the target program with that string on stdin.
   * Captures stdout into `captured_output`.
   * Calls `check_output(generated_input, captured_output)`.
3. If any assertion fails, the test suite reports an error.

### **Constraints**

* Provide one contiguous block of Python code that defines all required/optional functions. Do not invoke the functions yourself-only define them.
* Define up to 5 input generation functions, each returning between 1 and 4 inputs.
* The dynamic input functions must employ diverse strategies to generate inputs. Avoid generating inputs with the same logic or from the same distribution.
* Runtime limit per check_output call: 5 seconds.'''


FUNCTION = '''Given a problem statement and a Python function that aims to solve it, your task is to **write a test harness** that uncovers any potential bugs.

### **Task Overview**

You will deliver **a single** code block to define functions that can be run by our framework to generate inputs, run the program, and validate its outputs.
Consider two categories of test cases:
- **Hardcoded cases**: Manually crafted input-output pairs that expose known or likely bugs.
- **Dynamic cases**: Programmatically generated inputs that stress-test the implementation (e.g., randomized, combinatorial, large or edge-case inputs).

### **Required Functions**

```python
from typing import List, Dict, Any

def generate_input_1() -> List[Dict[str, Any]]:
    """
    Return 1-4 input dictionaries, each mapping the target function's parameter names to test values.
    Consider the following strategies:
      - Manually craft inputs that expose bugs.
      - Dynamically generate randomized, combinatorial, large, or edge-case inputs for stress testing.
    """
    # Your code here
    return input_list

def generate_input_2() -> List[Dict[str, Any]]:
    """
    Another function to return between 1 and 4 valid inputs.
    Employ a different strategy than previous input generation functions.
    """
    # Your code here
    return input_list

# You may add up to 3 more functions named generate_input_3(), generate_input_4(), etc.

def check_output(generated_input: Dict[str, Any], captured_output: Any) -> None:
    """
    Validate the output for a single generated input.
    Inputs:
        - generated_input: The input dictionary passed to the target program.
        - captured_output: The exact output produced by the target function.
    
    Hints: When exact outputs are hard to predict, avoid asserting them. Instead, consider:
      - Check key properties or invariants, e.g., output is sorted, has correct length, matches a pattern, has correct value ranges, etc.
      - Compare against a simple brute-force implementation
    """
    # Your code here
```

### **Execution Flow**

1. The framework calls generate input functions to obtain a list of test dictionaries.
2. For each dict:
   * It runs the target function with that dict as kwargs.
   * Captures its return value as `captured_output`.
   * Calls `check_output(generated_input, captured_output)`.
3. If any assertion fails, the test suite reports an error.

### **Constraints**

* Provide one contiguous block of Python code that defines all required/optional functions. Do not invoke the functions yourself-only define them.
* Define up to 5 input generation functions, each returning between 1 and 4 inputs.
* The dynamic input functions must employ diverse strategies to generate inputs. Avoid generating inputs with the same logic or from the same distribution.
* Runtime limit per check_output call: 5 seconds.'''


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