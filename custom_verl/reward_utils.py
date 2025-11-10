from enum import Enum


# We list all possible outcomes for a reward function
class RewardType(Enum):
    FormatError = 0
    PassGoodFailBad = 1
    PassGoodPassBad = 2
    FailGood = 3
    Truncated = 4
    GoodInputFailGood = 5
    GoodInputPassGoodPassBad = 6


class ExecutionResult(Enum):
    CompileError = 0
    ExecutionError = 1
    AssertionError = 2
    RuntimeError = 3
    TimeoutError = 4
    Correct = 5
    Wrong = 6
    EarlyStopped = 7  # this run case does not return because others early exit
    InputGeneratorError = 8
