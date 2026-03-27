from typing import List, Union, Literal, TypedDict, Any, Optional

TargetSelector = List[Union[str, int]]

class ValidationRule(TypedDict):
    id: str
    targets: List[TargetSelector]
    operator: Literal[">", ">=", "<", "<=", "=="]
    thresholds: List[float]
    weights: Optional[List[float]]
    master_threshold: Optional[float]

def resolve_path(data: Any, path: List[Union[str, int]]) -> Any:
    for key in path:
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            return None
    return data

def evaluate_condition(observed: float, operator: str, threshold: float) -> bool:
    ops = {
        ">=": lambda a, b: a >= b,
        ">":  lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<":  lambda a, b: a < b,
        "==": lambda a, b: a == b,
    }
    return ops.get(operator, lambda a, b: False)(observed, threshold)