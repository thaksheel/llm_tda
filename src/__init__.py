"""Pipeline package with deferred imports for optional ML dependencies."""

from importlib import import_module


_EXPORTS = {
    "Tuner": (".finetune", "Tuner"),
    "TDA": (".tda", "TDA"),
    "TDAEvaluation": (".tda_evaluation", "TDAEvaluation"),
    "Result": (".utils", "Result"),
    "Params": (".utils", "Params"),
    "BasicProjector": ("modules", "BasicProjector"),
    "ProjectionType": ("modules", "ProjectionType"),
    "get_tokenized_dataset": (".utils", "get_tokenized_dataset"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
