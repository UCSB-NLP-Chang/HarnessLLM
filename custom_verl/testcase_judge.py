import ast
import builtins
import faulthandler
import inspect
import json
import linecache
import os
import platform
import re
import signal
import sys
import time
import traceback
from collections.abc import Sequence
from contextlib import contextmanager
from types import ModuleType
from typing import List

import numpy as np
import torch

from custom_verl.code_judge import OnlineJudge, string_equal
from custom_verl.reward_utils import ExecutionResult, RewardType

PY_IMPORTS = "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\nfrom functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\ninf = float('inf')\n\n"

GLOBAL_TIMEOUT = int(os.getenv("TIMEOUT_OVERRIDE", "0"))  # 0 ➜ “no override”


def _effective_timeout(requested: int) -> int:
    """
    If the caller passed 0 or a negative number we respect the per-site value.
    Otherwise we return the global override so *every* call uses one number.
    """
    return GLOBAL_TIMEOUT or requested


def simple_extract_code(text, language="python"):
    if language == "python":
        matches = re.findall(r"```python\n(.*?)\n\s*```", text, re.DOTALL)
    elif language == "json":
        matches = re.findall(r"```json\n(.*?)\n\s*```", text, re.DOTALL)
    else:
        raise ValueError(f"Unsupported language: {language}")
    return matches[-1] if matches else ""


def get_func_defs(fullcode, tree=None):
    if tree is None:
        try:
            tree = ast.parse(fullcode)
        except Exception:
            return []
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]


def remove_func_def(fullcode, func_name="foo"):
    # If func_name() is copied from the original code, remove it
    try:
        tree = ast.parse(fullcode)
    except Exception:
        return fullcode
    foo_node = next(
        (node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == func_name),
        None
    )
    if foo_node is not None:
        lines = fullcode.splitlines()
        start_line = foo_node.lineno - 1
        end_line = foo_node.end_lineno
        fullcode = "\n".join(lines[:start_line] + lines[end_line:]).strip()
    return fullcode


def extract_function(fullcode, function_name, tree=None):
    # Extract the function definition from the code
    if tree is None:
        try:
            tree = ast.parse(fullcode)
        except Exception:
            return None
    foo_node = next(
        (node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name),
        None
    )
    if foo_node is not None:
        lines = fullcode.splitlines()
        start_line = foo_node.lineno - 1
        end_line = foo_node.end_lineno
        function_code = "\n".join(lines[start_line:end_line]).strip()
        return function_code
    return None


def preprocess_code(
    solution_str, code_pair, timeout=10, test_harness=True, in_memory=True
):
    if isinstance(code_pair, str):
        code_pair = json.loads(code_pair)
    good_code, bad_code = (
        code_pair.get("good_code", None),
        code_pair.get("bad_code"),
    )
    code_type = code_pair.get(
        "code_type", "input/output"
    )
    
    language = "python" if test_harness else "json"
    fullcode = simple_extract_code(solution_str, language=language)

    # Remove repeated definitions
    if language == "python" and isinstance(bad_code, str):
        remove_funcs = []
        if isinstance(good_code, dict):
            remove_funcs.append(good_code["entry_point"])
        bad_code_str = bad_code["code"] if isinstance(bad_code, dict) else bad_code
        remove_funcs += get_func_defs(bad_code_str)
        for func_name in remove_funcs:
            if f"def {func_name}(" in fullcode:
                fullcode = remove_func_def(fullcode, func_name=func_name)

    oj = OnlineJudge(
        timeout=timeout, do_parallel=False, early_exit=True, ignore_output=True, in_memory=in_memory
    )
    return fullcode, good_code, bad_code, oj, code_type


def contain_assert(code):
    """Check if the code contains assert statements"""
    if "assert" not in code:
        return False
    lines = code.split("\n")
    lines = [l.split("#")[0].split('"')[0].split("'")[0].strip() for l in lines]
    for line in lines:
        if "assert" in line:
            return True
    return False


def check_format(
    fullcode,
    code_type,
    test_code_tree,
    test_harness=True,
    entry_point=None,
):
    """Check if the code is formatted correctly"""
    if (
        fullcode == ""
        or (entry_point and f"def {entry_point}(" in fullcode)
    ):
        return False
    if test_harness:
        funcs = get_func_defs(fullcode, test_code_tree)
        if not any(func.startswith("generate_input_") for func in funcs):
            return False
        if "check_output" in funcs and not contain_assert(fullcode):
            return False
    else:
        try:
            testcases = json.loads(fullcode)
            required_keys = (["input_str", "expected_output"] if code_type == "input/output"
                             else ["input_args", "expected_output"])
            if not isinstance(testcases, list) or len(testcases) < 1:
                return False
            for test in testcases:
                if not isinstance(test, dict):
                    return False
                if any(key not in test for key in required_keys):
                    return False
                if code_type == "input/output":
                    if not isinstance(test["input_str"], str) or not isinstance(
                        test["expected_output"], str
                    ):
                        return False
                else:
                    if not isinstance(test["input_args"], dict):
                        return False
        except Exception:
            return False
    return True


class TimeoutException(Exception):
    pass

class FormatException(Exception):
    pass


@contextmanager
def max_exec_time(limit):
    """Timeout context that composes properly with nesting."""
    def _raise(_s, _f):  # noqa: D401
        raise TimeoutException

    prev_handler = signal.getsignal(signal.SIGALRM)
    prev_remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    start_monotonic = time.monotonic()
    signal.signal(signal.SIGALRM, _raise)
    limit = _effective_timeout(limit)
    signal.setitimer(signal.ITIMER_REAL, limit)

    try:
        yield
    finally:
        elapsed = time.monotonic() - start_monotonic
        signal.setitimer(signal.ITIMER_REAL, 0)
        if prev_remaining > 0:
            new_remaining = max(prev_remaining - elapsed, 0.1)
            signal.setitimer(signal.ITIMER_REAL, new_remaining)
        signal.signal(signal.SIGALRM, prev_handler)


def run_input_generator(input_generator, arg_list, code_type="input/output"):
    """Run input generator to get inputs"""
    try:
        with max_exec_time(10):
            input_list = input_generator()
            expected_type = str if code_type == "input/output" else dict
            if (
                not isinstance(input_list, list)
                or len(input_list) < 1
                or not all(isinstance(i, expected_type) for i in input_list)
            ):
                return None
            if code_type == "functional":
                # check if input matches the function signature
                arg_set = set(arg_list)
                if any(len(arg_set - set(i.keys())) > 0 for i in input_list):
                    return None
                input_list = [{k_: v_ for k_, v_ in i.items() if k_ in arg_set} for i in input_list]
            return input_list
    except TimeoutException as e:
        print("====== Input generator timed out ======")
        return None
    except Exception as e:
        return None


def run_output_checker(output_checker, input_i, output_i):
    """Run output checker to check the output"""
    try:
        with max_exec_time(10):
            output_checker(input_i, output_i)
    except AssertionError as e:
        tb_str = traceback.format_exc()
        return False, ExecutionResult.AssertionError, tb_str
    except TimeoutException as e:
        print("====== Output checker timed out ======")
        return False, ExecutionResult.TimeoutError, "Output checker timed out"
    except Exception as e:
        tb_str = traceback.format_exc()
        return False, ExecutionResult.RuntimeError, tb_str
    return True, ExecutionResult.Correct, ""


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        fn_name = fn_name.replace("Solution().", "")
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        return None


def compile_code(code: str, module_name: str = "code_sol", use_solution: bool = True):
    try:
        tmp_sol = ModuleType(module_name, "")
        filename = f"<{module_name}>"
        # ① Tell linecache where to find the lines for that *virtual* file
        linecache.cache[filename] = (
            len(code),                 # length (so it knows the file is “complete”)
            None,                      # mtime – None is fine
            code.splitlines(True),     # list of lines *with* '\n'
            filename,
        )
        
        with max_exec_time(10):
            # ② Compile with that filename, then exec into the module namespace
            exec(compile(code, filename, "exec"), tmp_sol.__dict__)

            if use_solution and "class Solution" in code:
                # leetcode wraps solutions in `Solution`
                # this is a hack to check if it is leetcode solution or not
                compiled_sol = tmp_sol.Solution()
            else:
                # do nothing in the other case since function is accesible
                compiled_sol = tmp_sol

            assert compiled_sol is not None
            return compiled_sol
    except TimeoutException as e:
        print("====== Compilation timed out ======")
        return None
    except Exception as e:
        print(f"====== Compilation error: {e} ======")
        print(f"Code:\n{code}")
        return None


def run_function(method, input_dict, arg_list):
    """Run target function and capture the output"""
    faulthandler.enable()
    try:
        input_list = [input_dict[arg] for arg in arg_list]
        with max_exec_time(10):
            output = method(*input_list)
            err = ""
    except TimeoutException as e:
        print("====== Function execution timed out ======")
        output = ExecutionResult.TimeoutError
        err = "Function execution timed out"
    except Exception as e:
        err = traceback.format_exc()
        output = ExecutionResult.RuntimeError
    finally:
        faulthandler.disable()
    return output, err


def get_arg_list(fn) -> List[str]:
    """
    Given any callable `fn`, return its parameter names in declaration order,
    including `*args` and `**kwargs` (prefixed accordingly).
    """
    sig = inspect.signature(fn)
    params = []
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            params.append(f"*{name}")
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            params.append(f"**{name}")
        else:
            params.append(name)
    return params


def is_builtin_class(obj):
    """Return True if *obj* (a class **or** an instance) is a built-in type."""
    cls = obj if isinstance(obj, type) else type(obj)
    return cls.__module__ == "builtins"


def is_equal(obj1, obj2):
    try:
        if is_builtin_class(obj1) and is_builtin_class(obj2):
            if ((isinstance(obj1, tuple) or isinstance(obj2, tuple))
                and isinstance(obj1, Sequence)
                and isinstance(obj2, Sequence)
            ):
                return list(obj1) == list(obj2)
            elif isinstance(obj1, str) and isinstance(obj2, str):
                return string_equal(obj1, obj2)
            is_none = [obj1 is None, obj2 is None]
            if any(is_none) and not all(is_none):
                return False
            return obj1 == obj2
        if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
            return np.array_equal(obj1, obj2)
        if isinstance(obj1, torch.Tensor) and isinstance(obj2, torch.Tensor):
            return torch.equal(obj1, obj2)
        if type(obj1) is not type(obj2):
            return False
    except Exception as e:
        print(f"Error comparing objects: {e}")
    return None


def prepare_target_code(code, code_type, module_name="target_code"):
    code_str = code["code"] if isinstance(code, dict) else code
    if code_type == "functional":
        fn_name = code["entry_point"]
        code_str = PY_IMPORTS + code_str
        compiled_sol = compile_code(code_str, module_name)
        if compiled_sol is None:
            return None, None
        code = get_function(compiled_sol, fn_name)
        if code is None:
            return None, None
        arg_list = get_arg_list(code) # because model generates harness based on bad code
    else:
        arg_list = None
    return code, arg_list


def run_input_output(
    test_code,
    good_code,
    bad_code,
    oj,
    num_testcases,
    code_type,
    wrong_test_penalty=0.0,
    good_test_reward=1.0,
    good_input_reward=0.1,
    test_pair=True,
):
    bad_code, arg_list = prepare_target_code(bad_code, code_type, "candidate_solution")
    if bad_code is None:
        if test_pair:
            return 0.0, (RewardType.FailGood, None, None), (0, ExecutionResult.CompileError), -1
        else:
            return 0, -1
    if good_code is not None:
        good_code, _ = prepare_target_code(good_code, code_type)
        if good_code is None:
            return 0.0, (RewardType.FailGood, None, None), (0, ExecutionResult.CompileError), -1
    
    testcases = json.loads(test_code)
    input_key = "input_str" if code_type == "input/output" else "input_args"
    inputs_ = [i[input_key] for i in testcases]
    outputs_ = [i['expected_output'] for i in testcases]
    input_list, output_list = [], []
    for input_i, output_i in zip(inputs_, outputs_):
        if any(is_equal(input_i, i) for i in input_list):
            continue
        input_list.append(input_i)
        output_list.append(output_i)
    input_list = input_list[:num_testcases]
    output_list = output_list[:num_testcases]
    
    if test_pair:
        result, fail_reason = run_test_cases_pair(
            good_code,
            bad_code,
            input_list,
            arg_list,
            code_type,
            oj,
            output_list=output_list,
        )
        reward = get_reward(
            result,
            wrong_test_penalty,
            good_input_reward,
            good_test_reward,
        )
        return reward, (result, None, None), fail_reason, len(input_list)
    else:
        assert good_code is None
        return run_test_cases(
            bad_code,
            input_list,
            arg_list,
            code_type,
            oj,
            output_list=output_list,
        ), len(input_list)


def get_reward(
    reward_type,
    wrong_test_penalty,
    good_input_reward,
    good_test_reward,
):
    if reward_type == RewardType.FailGood:
        return wrong_test_penalty
    elif reward_type == RewardType.PassGoodPassBad:
        return 0.0
    elif reward_type in {RewardType.GoodInputFailGood, RewardType.GoodInputPassGoodPassBad}:
        return good_input_reward
    elif reward_type == RewardType.PassGoodFailBad:
        return good_test_reward
    else:
        raise ValueError(f"Unexpected reward type: {reward_type}")


def remove_set_seed(code):
    """Remove set_seed from the code"""
    lines = code.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith("random.seed("):
            continue
        new_lines.append(line)
    return "\n".join(new_lines)


def set_seed(code, seed):
    """Set random seed in the code"""
    code = remove_set_seed(code)
    code = f"random.seed({seed})\n" + code
    code = "import random\n" + code
    return code


def run_test_harness(
    test_code,
    test_code_tree,
    good_code,
    bad_code,
    oj,
    num_testcases,
    code_type,
    wrong_test_penalty=0.0,
    good_test_reward=1.0,
    good_input_reward=0.1,
    format_penalty=-1.0,
    num_cases_per_generator=4,
    num_repeats=1,
    seed=None,
    test_pair=True,
    num_generators=5,
):
    test_code = "from typing import List, Dict, Tuple, Any\n" + test_code
    if seed is not None:
        test_code = set_seed(test_code, seed)
    compiled_test_code = compile_code(test_code, "output_checker", False)
    if compiled_test_code is None:
        if test_pair:
            return format_penalty, (RewardType.FormatError, RewardType.FormatError, RewardType.FormatError), None, -1
        else:
            raise FormatException
    funcs = get_func_defs(test_code, test_code_tree)
    generator_funcs = [f for f in funcs if f.startswith("generate_input_")]
    generator_funcs = sorted(generator_funcs)[:num_generators]
    input_generators = [
        get_function(compiled_test_code, f) for f in generator_funcs
    ]
    output_checker = get_function(compiled_test_code, "check_output")
    if not input_generators or not input_generators[0] or not output_checker:
        if test_pair:
            return format_penalty, (RewardType.FormatError, RewardType.FormatError, RewardType.FormatError), None, -1
        else:
            raise FormatException
    
    bad_code, arg_list = prepare_target_code(bad_code, code_type, "candidate_solution")
    if bad_code is None:
        if test_pair:
            return 0.0, (RewardType.FailGood, RewardType.FailGood, RewardType.FailGood), (0, ExecutionResult.CompileError), -1
        else:
            return 0, -1
    if good_code is not None:
        good_code, _ = prepare_target_code(good_code, code_type)
        if good_code is None:
            return 0.0, (RewardType.FailGood, RewardType.FailGood, RewardType.FailGood), (0, ExecutionResult.CompileError), -1
    
    possible_results = {
        RewardType.FailGood,
        RewardType.PassGoodPassBad,
        RewardType.GoodInputFailGood,
        RewardType.GoodInputPassGoodPassBad,
        RewardType.PassGoodFailBad,
    }
    
    inputs_ = []
    # Get inputs
    for input_gen in input_generators:
        for _ in range(num_repeats):
            inputs_i = run_input_generator(input_gen, arg_list, code_type)
            if inputs_i is not None:
                inputs_.extend(inputs_i[:num_cases_per_generator])
    input_list = []
    for input_i in inputs_:
        if any(is_equal(input_i, i) for i in input_list):
            continue
        input_list.append(input_i)
    input_list = input_list[:num_testcases]
    
    if test_pair:
        # Run test cases on good and bad code
        if len(input_list) == 0:
            return format_penalty, (RewardType.FormatError, RewardType.FormatError, RewardType.FormatError), None, -1
        result, fail_reason = run_test_cases_pair(
            good_code,
            bad_code,
            input_list,
            arg_list,
            code_type,
            oj,
            output_checker=output_checker,
        )
        assert result in possible_results, f"Unexpected result: {result}"
        reward = get_reward(
            result,
            wrong_test_penalty,
            good_input_reward,
            good_test_reward,
        )
        return reward, (result, None, None), fail_reason, len(input_list)
    else:
        assert good_code is None
        return run_test_cases(
            bad_code,
            input_list,
            arg_list,
            code_type,
            oj,
            output_checker=output_checker,
        ), len(input_list)


def run_test_cases_pair(
    good_code,
    bad_code,
    input_list,
    arg_list,
    code_type,
    oj,
    output_list = None,
    output_checker = None,
):
    assert (output_list is None) != (output_checker is None)
    if output_list is not None:
        assert len(input_list) == len(output_list)
    # Run good code
    good_outputs = []
    for index, input_i in enumerate(input_list):
        if code_type == "input/output":
            testcases = {"inputs": [input_i], "outputs": [""]}
            goodcode_output = oj.run(good_code, testcases)
            if goodcode_output[0][0] == 0:
                if goodcode_output[1][0] == ExecutionResult.TimeoutError:
                    good_outputs.append(ExecutionResult.TimeoutError)
                else:
                    return RewardType.FailGood, (index, ExecutionResult.ExecutionError)
            else:
                good_outputs.append(goodcode_output[2][0])
        else:
            output_i, _ = run_function(good_code, input_i, arg_list)
            if isinstance(output_i, ExecutionResult):
                if output_i == ExecutionResult.TimeoutError:
                    good_outputs.append(ExecutionResult.TimeoutError)
                else:
                    return RewardType.FailGood, (index, ExecutionResult.ExecutionError)
            else:
                good_outputs.append(output_i)
    
    # Run bad code
    bad_outputs = []
    for index, input_i in enumerate(input_list):
        if code_type == "input/output":
            testcases = {"inputs": [input_i], "outputs": [""]}
            badcode_output = oj.run(bad_code, testcases)
            if badcode_output[0][0] == 0:
                bad_outputs.append(badcode_output[1][0])
            else:
                bad_outputs.append(badcode_output[2][0])
        else:
            output_i, _ = run_function(bad_code, input_i, arg_list)
            bad_outputs.append(output_i)

    # Compare outputs
    assert len(good_outputs) == len(bad_outputs) == len(input_list)
    diff_indices, equal_indices = [], []
    for index, (good, bad) in enumerate(zip(good_outputs, bad_outputs)):
        if isinstance(good, ExecutionResult):
            assert good == ExecutionResult.TimeoutError
            continue
        if isinstance(bad, ExecutionResult):
            assert bad not in [ExecutionResult.Correct, ExecutionResult.Wrong, ExecutionResult.EarlyStopped]
            diff_indices.append(index)
            continue
        equal = is_equal(good, bad)
        if equal is None:
            continue
        if equal:
            equal_indices.append(index)
        else:
            diff_indices.append(index)
    
    def check_outputs(outputs, indices):
        assert len(outputs) == len(indices)
        results, fail_reasons = [], []
        for index, output_i in zip(indices, outputs):
            if isinstance(output_i, ExecutionResult):
                return False, (index, output_i)
            if output_checker is not None:
                is_correct, fail_reason, _ = run_output_checker(
                    output_checker, input_list[index], output_i
                )
            else:
                is_correct = is_equal(output_i, output_list[index])
                fail_reason = ExecutionResult.Correct if is_correct else ExecutionResult.AssertionError
            if not is_correct:
                return False, (index, fail_reason)
            results.append(is_correct)
            fail_reasons.append(fail_reason)
        return True, None
    
    if len(diff_indices) == 0:
        # all outputs are equal, only check good outputs
        is_correct, fail_reason = check_outputs(
            [o for o in good_outputs if o != ExecutionResult.TimeoutError],
            [i for i, o in enumerate(good_outputs) if o != ExecutionResult.TimeoutError],
        )
        if is_correct:
            return RewardType.PassGoodPassBad, None
        else:
            return RewardType.FailGood, fail_reason
    # some outputs are different
    # check good outputs are correct first
    good_correct, good_fail_reason = check_outputs(
        [o for o in good_outputs if o != ExecutionResult.TimeoutError],
        [i for i, o in enumerate(good_outputs) if o != ExecutionResult.TimeoutError],
    )
    if not good_correct:
        return RewardType.GoodInputFailGood, good_fail_reason
    # check bad outputs for the diff indices
    bad_correct, _ = check_outputs(
        [o for i, o in enumerate(bad_outputs) if i not in equal_indices],
        [i for i in range(len(bad_outputs)) if i not in equal_indices],
    )
    if bad_correct:
        return RewardType.GoodInputPassGoodPassBad, None
    else:
        return RewardType.PassGoodFailBad, None


def compute_score(
    solution_str,
    code_pair,
    num_testcases=20,
    wrong_test_penalty=0.0,
    good_test_reward=1.0,
    good_input_reward=0.1,
    format_penalty=-1.0,
    timeout=10,
    test_harness=True,
    num_repeats=1,
    seed=None,
    in_memory=True,
    num_generators=5,
):
    fullcode, good_code, bad_code, oj, code_type = preprocess_code(
        solution_str, code_pair, timeout=timeout, test_harness=test_harness, in_memory=in_memory
    )
    entry_point = good_code["entry_point"] if code_type == "functional" else None
    try:
        tree = ast.parse(fullcode)
    except Exception:
        tree = None
    format_correct = check_format(
        fullcode,
        code_type,
        tree,
        test_harness=test_harness,
        entry_point=entry_point,
    )

    if not format_correct:
        return format_penalty, [(RewardType.FormatError, None, None)], [None], -1
    
    reliability_guard()

    if test_harness:
        score, reward_type, fail_reason, num_inputs = run_test_harness(
            fullcode,
            tree,
            good_code,
            bad_code,
            oj,
            num_testcases,
            code_type,
            wrong_test_penalty,
            good_test_reward,
            good_input_reward,
            format_penalty,
            num_repeats=num_repeats,
            seed=seed,
            num_generators=num_generators,
        )
    else:
        score, reward_type, fail_reason, num_inputs = run_input_output(
            fullcode,
            good_code,
            bad_code,
            oj,
            num_testcases,
            code_type,
            wrong_test_penalty,
            good_test_reward,
            good_input_reward,
        )

    return score, [reward_type], [fail_reason], num_inputs


def run_test_cases(
    code,
    input_list,
    arg_list,
    code_type,
    oj,
    output_list = None,
    output_checker = None,
):
    assert (output_list is None) != (output_checker is None)
    if output_list is not None:
        assert len(input_list) == len(output_list)
    
    # Run code
    exec_outputs = []
    for index, input_i in enumerate(input_list):
        if code_type == "input/output":
            testcases = {"inputs": [input_i], "outputs": [""]}
            output = oj.run(code, testcases)
            if output[0][0] == 0:
                exec_outputs.append(ExecutionResult.ExecutionError)
            else:
                exec_outputs.append(output[2][0])
        else:
            output_i, _ = run_function(code, input_i, arg_list)
            if isinstance(output_i, ExecutionResult):
                exec_outputs.append(ExecutionResult.ExecutionError)
            else:
                exec_outputs.append(output_i)
    
    def check_outputs(outputs, indices):
        assert len(outputs) == len(indices)
        num_correct = 0
        for index, output_i in zip(indices, outputs):
            if output_i == ExecutionResult.ExecutionError:
                continue
            if output_checker is not None:
                is_correct, fail_reason, err_i = run_output_checker(
                    output_checker, input_list[index], output_i
                )
            else:
                is_correct = is_equal(output_i, output_list[index])
            if is_correct == True:  # noqa: E712
                num_correct += 1
        return num_correct
    
    # check outputs
    return check_outputs(exec_outputs, list(range(len(input_list))))


def get_feedback(
    solution_str,
    code_pairs,
    num_testcases=20,
    timeout=10,
    test_harness=True,
    num_repeats=1,
    seed=None,
    num_generators=5,
    num_cases_per_generator=4,
):
    reliability_guard()
    num_passed = []
    for code_pair in code_pairs:
        fullcode, good_code, bad_code, oj, code_type = preprocess_code(
            solution_str, code_pair, timeout=timeout, test_harness=test_harness
        )
        entry_point = bad_code["entry_point"] if code_type == "functional" else None
        try:
            tree = ast.parse(fullcode)
        except Exception:
            tree = None
        format_correct = check_format(
            fullcode,
            code_type,
            tree,
            test_harness=test_harness,
            entry_point=entry_point,
        )

        if not format_correct:
            # cannot run test cases, assume passed 0 for all programs
            return [0] * len(code_pairs), -1

        good_code = None
        try:
            if test_harness:
                num_correct, num_test = run_test_harness(
                    fullcode,
                    tree,
                    good_code,
                    bad_code,
                    oj,
                    num_testcases,
                    code_type,
                    num_repeats=num_repeats,
                    seed=seed,
                    test_pair=False,
                    num_generators=num_generators,
                    num_cases_per_generator=num_cases_per_generator,
                )
            else:
                num_correct, num_test = run_input_output(
                    fullcode,
                    good_code,
                    bad_code,
                    oj,
                    num_testcases,
                    code_type,
                    test_pair=False,
                )
            num_passed.append(num_correct)
        except FormatException:
            # cannot run test cases, assume passed 0 for all programs
            return [0] * len(code_pairs), -1

    return num_passed, num_test


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if getattr(reliability_guard, "_already_run", False):
        return
    reliability_guard._already_run = True
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    builtins.quit = None

    os.kill = None
    os.system = None
    os.remove = None
    os.removedirs = None
    os.fchdir = None
    os.truncate = None
    os.replace = None
    os.fchmod = None
    os.fchown = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None