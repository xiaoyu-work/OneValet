"""
FlowAgent State Merger - Merge state from parallel agents

This module provides:
- StateMerger: Merge states using configured strategies
- merge_values: Apply merge strategy to a list of values
"""

from typing import Dict, Any, List, Optional, Callable, Union, Set
from collections import defaultdict

from .models import MergeStrategy, MergeConfig


class MergeError(Exception):
    """Raised when merging fails"""
    pass


def merge_values(
    values: List[Any],
    strategy: MergeStrategy,
    custom_fn: Optional[Callable] = None,
    ignore_none: bool = True
) -> Any:
    """
    Merge a list of values using the specified strategy.

    Args:
        values: List of values to merge
        strategy: Merge strategy to use
        custom_fn: Custom merge function (for CUSTOM strategy)
        ignore_none: Skip None values when merging

    Returns:
        Merged value

    Examples:
        >>> merge_values([1, 2, 3], MergeStrategy.ADD)
        6
        >>> merge_values([[1,2], [3,4]], MergeStrategy.ADD)
        [1, 2, 3, 4]
        >>> merge_values([{'a':1}, {'b':2}], MergeStrategy.MERGE)
        {'a': 1, 'b': 2}
        >>> merge_values([5, 3, 8], MergeStrategy.MAX)
        8
    """
    # Filter None values if requested
    if ignore_none:
        values = [v for v in values if v is not None]

    if not values:
        return None

    if strategy == MergeStrategy.REPLACE:
        # Last value wins
        return values[-1]

    elif strategy == MergeStrategy.FIRST:
        # First value wins
        return values[0]

    elif strategy == MergeStrategy.ADD:
        # Add/concatenate values
        first = values[0]

        if isinstance(first, (int, float)):
            # Add numbers
            return sum(values)
        elif isinstance(first, list):
            # Concatenate lists
            result = []
            for v in values:
                if isinstance(v, list):
                    result.extend(v)
                else:
                    result.append(v)
            return result
        elif isinstance(first, str):
            # Concatenate strings
            return "".join(values)
        else:
            # Fall back to list
            return list(values)

    elif strategy == MergeStrategy.MERGE:
        # Merge dictionaries
        result = {}
        for v in values:
            if isinstance(v, dict):
                result.update(v)
        return result

    elif strategy == MergeStrategy.MAX:
        # Maximum value
        try:
            return max(values)
        except TypeError:
            # Can't compare, return last
            return values[-1]

    elif strategy == MergeStrategy.MIN:
        # Minimum value
        try:
            return min(values)
        except TypeError:
            return values[-1]

    elif strategy == MergeStrategy.UNION:
        # Set union (unique values)
        result: Set = set()
        for v in values:
            if isinstance(v, (list, tuple, set)):
                result.update(v)
            else:
                result.add(v)
        return result

    elif strategy == MergeStrategy.CUSTOM:
        # Use custom function
        if custom_fn is None:
            raise MergeError("Custom merge function required for CUSTOM strategy")
        return custom_fn(values)

    else:
        raise MergeError(f"Unknown merge strategy: {strategy}")


class StateMerger:
    """
    Merge state from multiple parallel agents.

    Applies configured merge strategies to each field.
    Fields without explicit strategy use REPLACE (last write wins).

    Example:
        merger = StateMerger(
            merge_strategies={
                "results": MergeStrategy.ADD,
                "confidence": MergeStrategy.MAX,
                "sources": MergeStrategy.UNION,
            }
        )

        merged = merger.merge([
            {"results": [1, 2], "confidence": 0.8, "sources": ["google"]},
            {"results": [3, 4], "confidence": 0.9, "sources": ["arxiv"]},
        ])
        # merged = {
        #     "results": [1, 2, 3, 4],
        #     "confidence": 0.9,
        #     "sources": {"google", "arxiv"},
        # }
    """

    def __init__(
        self,
        merge_strategies: Optional[Dict[str, MergeStrategy]] = None,
        custom_merge_fns: Optional[Dict[str, Callable]] = None,
        default_strategy: MergeStrategy = MergeStrategy.REPLACE,
        ignore_none: bool = True
    ):
        """
        Initialize state merger.

        Args:
            merge_strategies: Strategy for each field
            custom_merge_fns: Custom functions for CUSTOM strategy
            default_strategy: Default strategy for unspecified fields
            ignore_none: Skip None values when merging
        """
        self.merge_strategies = merge_strategies or {}
        self.custom_merge_fns = custom_merge_fns or {}
        self.default_strategy = default_strategy
        self.ignore_none = ignore_none

    def merge(
        self,
        states: List[Dict[str, Any]],
        additional_strategies: Optional[Dict[str, MergeStrategy]] = None
    ) -> Dict[str, Any]:
        """
        Merge multiple state dictionaries.

        Args:
            states: List of state dictionaries to merge
            additional_strategies: Extra strategies to apply

        Returns:
            Merged state dictionary
        """
        if not states:
            return {}

        # Combine strategies
        strategies = dict(self.merge_strategies)
        if additional_strategies:
            strategies.update(additional_strategies)

        # Collect all values for each field
        field_values: Dict[str, List[Any]] = defaultdict(list)

        for state in states:
            for key, value in state.items():
                field_values[key].append(value)

        # Merge each field
        result = {}
        for field, values in field_values.items():
            strategy = strategies.get(field, self.default_strategy)
            custom_fn = self.custom_merge_fns.get(field)

            result[field] = merge_values(
                values=values,
                strategy=strategy,
                custom_fn=custom_fn,
                ignore_none=self.ignore_none
            )

        return result

    def merge_collected_fields(
        self,
        agent_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge collected_fields from agent results.

        Args:
            agent_results: List of agent results with collected_fields

        Returns:
            Merged collected_fields
        """
        states = [
            r.get("collected_fields", {})
            for r in agent_results
            if isinstance(r, dict)
        ]
        return self.merge(states)

    def set_strategy(
        self,
        field: str,
        strategy: MergeStrategy,
        custom_fn: Optional[Callable] = None
    ) -> None:
        """Set merge strategy for a field"""
        self.merge_strategies[field] = strategy
        if custom_fn:
            self.custom_merge_fns[field] = custom_fn

    def get_strategy(self, field: str) -> MergeStrategy:
        """Get merge strategy for a field"""
        return self.merge_strategies.get(field, self.default_strategy)


# Convenience functions for common merge operations
def merge_lists(values: List[List[Any]]) -> List[Any]:
    """Merge multiple lists into one"""
    return merge_values(values, MergeStrategy.ADD)


def merge_dicts(values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple dicts into one"""
    return merge_values(values, MergeStrategy.MERGE)


def merge_unique(values: List[Any]) -> Set[Any]:
    """Get unique values from all inputs"""
    return merge_values(values, MergeStrategy.UNION)


def merge_max(values: List[Any]) -> Any:
    """Get maximum value"""
    return merge_values(values, MergeStrategy.MAX)


def merge_min(values: List[Any]) -> Any:
    """Get minimum value"""
    return merge_values(values, MergeStrategy.MIN)
