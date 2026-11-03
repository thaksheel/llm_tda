"""Scalar labels and sets of fact IDs shared by attribution runners."""

import json


def label_set(value):
    """Decode JSON fact lists while preserving existing scalar-label behavior."""
    if isinstance(value, str) and value.startswith("["):
        value = json.loads(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(value)
    return frozenset([value])


def selection_key(value):
    """Make list labels hashable for optional round-robin source selection."""
    labels = label_set(value)
    if len(labels) == 1:
        return next(iter(labels))
    return json.dumps(sorted(labels, key=str), ensure_ascii=True)


def missing_query_labels(sources, queries):
    available = set()
    for value in sources["label"]:
        available.update(label_set(value))
    return {str(value) for value in queries["label"] if not label_set(value) & available}
