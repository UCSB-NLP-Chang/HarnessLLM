import os
import signal
import subprocess
import tempfile
from typing import List

import prctl

from custom_verl.reward_utils import ExecutionResult


def _preexec():
    os.setsid()
    prctl.set_pdeathsig(signal.SIGKILL)


def string_equal(s1: str, s2: str) -> bool:
    s1 = s1.strip()
    s2 = s2.strip()
    if s1 == s2:
        return True
    s1_lines = s1.splitlines()
    s2_lines = s2.splitlines()
    if len(s1_lines) != len(s2_lines):
        return False
    for line1, line2 in zip(s1_lines, s2_lines):
        if line1.strip() != line2.strip():
            return False
    return True


class OnlineJudge(object):
    def __init__(
        self, timeout=1, do_parallel=True, early_exit=True, ignore_output=False, in_memory=True
    ):
        self.timeout = timeout
        self.do_parallel = do_parallel
        self.early_exit = early_exit
        self.ignore_output = ignore_output
        self.in_memory = in_memory

    def check_language(self, code_string):
        return "python"

    def run_test_case(self, code_string, lan_bin, one_in, one_out, checker):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Paths for our temp files
            input_path = os.path.join(tmpdir, "input.txt")
            # Write out input/output files
            with open(input_path, "wb") as f:
                f.write(one_in.encode())
            if self.in_memory:
                cmd = [lan_bin, "-u", "-c", code_string]
            else:
                code_path = os.path.join(tmpdir, "candidate_solution.py")
                with open(code_path, "w") as f:
                    f.write(code_string)
                cmd = [lan_bin, "-u", code_path]
            try:
                with open(input_path, "rb") as input_file:
                    with subprocess.Popen(
                        cmd,
                        stdin=input_file,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=_preexec,
                        shell=isinstance(cmd, str),  # shell=True only for C++
                    ) as proc:
                        try:
                            output, errors = proc.communicate(timeout=self.timeout)
                            if errors:
                                self.eval_error += 1
                                errors = errors.decode("utf-8")
                                # check error message
                                if "AssertionError" in errors:
                                    res = (0, ExecutionResult.AssertionError, None, errors)
                                else:
                                    res = (0, ExecutionResult.RuntimeError, None, errors)
                            else:
                                if self.ignore_output:
                                    res = 1
                                else:
                                    res = 1 if string_equal(output.decode("utf-8"), one_out) else 0
                                res = (
                                    res,
                                    ExecutionResult.Correct if res == 1 else ExecutionResult.Wrong,
                                    output.decode("utf-8"),
                                    "",  # No error message
                                )
                        except subprocess.TimeoutExpired:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                            proc.wait()
                            res = (0, ExecutionResult.TimeoutError, None, "Execution timed out")
                        finally:
                            # Ensure no process remains running after the with block
                            try:
                                os.killpg(
                                    os.getpgid(proc.pid), signal.SIGKILL
                                )  # Force kill process group
                            except ProcessLookupError:
                                pass  # Process already terminated
                            proc.wait()  # Reap process

                        return res
            except Exception:
                self.execute_error += 1
                return (0, ExecutionResult.ExecutionError, None, "Execution failed due to an exception")

    def run(self, code_string, test_cases) -> List[int]:
        self.execute_error = 0
        self.eval_error = 0

        # Create a list to store results in order.
        if "inputs" in test_cases:
            in_key = "inputs"
            out_key = "outputs"
        else:
            raise ValueError("Test cases must have 'inputs' and 'outputs' keys.")
        
        results = [None] * len(test_cases[in_key])
        reward_types = [None] * len(test_cases[out_key])
        captured_output = [None] * len(test_cases[out_key])
        std_errs = [None] * len(test_cases[out_key])

        if not code_string:
            return (
                [0] * len(test_cases[in_key]),
                [ExecutionResult.CompileError] * len(test_cases[in_key]),
                [""] * len(test_cases[in_key]),
                ["No code provided"] * len(test_cases[in_key]),
            )
        lan_bin = self.check_language(code_string)
        checker = test_cases.get("generated_checker", "")

        # Prepare a list of test cases with their index so we can restore order.
        cases = list(
            zip(
                range(len(test_cases[in_key])),
                test_cases[in_key],
                test_cases[out_key],
            )
        )

        # Sequential execution
        for idx, one_in, one_out in cases:
            try:
                full_res = self.run_test_case(
                    code_string, lan_bin, str(one_in), str(one_out), checker
                )
            except Exception:
                full_res = (0, ExecutionResult.ExecutionError, None, "Execution failed due to an exception")
            finally:
                self.execute_error += 1 if full_res[0] == 0 else 0

            results[idx] = full_res[0]
            reward_types[idx] = full_res[1]
            captured_output[idx] = full_res[2]
            std_errs[idx] = full_res[3]
            if self.early_exit and full_res[0] == 0:
                # Mark all remaining test cases as failed and break.
                for j in range(idx + 1, len(test_cases[in_key])):
                    results[j] = 0
                    reward_types[j] = ExecutionResult.EarlyStopped
                break

        return results, reward_types, captured_output, std_errs
