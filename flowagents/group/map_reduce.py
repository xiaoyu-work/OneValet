"""
FlowAgent Map-Reduce - Dynamic parallelism for processing collections

This module provides:
- MapReduceExecutor: Execute map-reduce operations on collections
- Built-in reduce functions
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Dict, Any, List, Optional, Callable, Awaitable,
    TypeVar, Generic, Union
)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class MapResult:
    """Result from a single map operation"""
    index: int
    input_item: Any
    output: Any
    success: bool = True
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class MapReduceResult(Generic[R]):
    """Result from a complete map-reduce operation"""
    total_items: int
    successful_items: int
    failed_items: int

    # Individual map results
    map_results: List[MapResult] = field(default_factory=list)

    # Final reduced result
    reduced_result: Optional[R] = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.successful_items / self.total_items

    def get_successful_outputs(self) -> List[Any]:
        """Get outputs from successful map operations"""
        return [r.output for r in self.map_results if r.success and r.output is not None]

    def get_failed_items(self) -> List[Any]:
        """Get input items that failed"""
        return [r.input_item for r in self.map_results if not r.success]


class MapReduceExecutor:
    """
    Execute map-reduce operations on collections.

    Provides parallel processing with:
    - Concurrency control via semaphore
    - Error handling per item
    - Progress tracking
    - Custom reduce functions

    Example:
        executor = MapReduceExecutor(max_concurrency=10)

        async def analyze_email(email):
            return await llm.analyze(email)

        def filter_important(results):
            return [r for r in results if r["important"]]

        result = await executor.execute(
            items=emails,
            map_fn=analyze_email,
            reduce_fn=filter_important
        )
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        continue_on_error: bool = True,
        timeout_per_item: Optional[float] = None
    ):
        """
        Initialize map-reduce executor.

        Args:
            max_concurrency: Maximum parallel operations
            continue_on_error: Continue processing if item fails
            timeout_per_item: Timeout per item in seconds
        """
        self.max_concurrency = max_concurrency
        self.continue_on_error = continue_on_error
        self.timeout_per_item = timeout_per_item

    async def execute(
        self,
        items: List[T],
        map_fn: Callable[[T], Awaitable[R]],
        reduce_fn: Optional[Callable[[List[R]], Any]] = None,
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> MapReduceResult:
        """
        Execute map-reduce on a collection.

        Args:
            items: Items to process
            map_fn: Async function to apply to each item
            reduce_fn: Function to combine results (default: return list)
            progress_callback: Called with (completed, total) for progress

        Returns:
            MapReduceResult with individual and reduced results
        """
        result = MapReduceResult(
            total_items=len(items),
            successful_items=0,
            failed_items=0,
            started_at=datetime.now()
        )

        if not items:
            result.completed_at = datetime.now()
            result.reduced_result = reduce_fn([]) if reduce_fn else []
            return result

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Track completed count
        completed_count = 0

        async def process_item(index: int, item: T) -> MapResult:
            nonlocal completed_count

            map_result = MapResult(
                index=index,
                input_item=item,
                output=None,
                started_at=datetime.now()
            )

            async with semaphore:
                try:
                    if self.timeout_per_item:
                        output = await asyncio.wait_for(
                            map_fn(item),
                            timeout=self.timeout_per_item
                        )
                    else:
                        output = await map_fn(item)

                    map_result.output = output
                    map_result.success = True

                except asyncio.TimeoutError:
                    map_result.success = False
                    map_result.error = f"Timeout after {self.timeout_per_item}s"
                except Exception as e:
                    map_result.success = False
                    map_result.error = str(e)

                    if not self.continue_on_error:
                        raise

                finally:
                    map_result.completed_at = datetime.now()
                    completed_count += 1

                    # Call progress callback
                    if progress_callback:
                        try:
                            await progress_callback(completed_count, len(items))
                        except Exception:
                            pass

            return map_result

        # Process all items in parallel
        tasks = [
            process_item(i, item)
            for i, item in enumerate(items)
        ]

        map_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for r in map_results:
            if isinstance(r, Exception):
                # Task itself raised an exception
                result.failed_items += 1
                result.map_results.append(MapResult(
                    index=-1,
                    input_item=None,
                    output=None,
                    success=False,
                    error=str(r)
                ))
            else:
                result.map_results.append(r)
                if r.success:
                    result.successful_items += 1
                else:
                    result.failed_items += 1

        # Sort by original index
        result.map_results.sort(key=lambda r: r.index)

        # Apply reduce function
        successful_outputs = result.get_successful_outputs()
        if reduce_fn:
            result.reduced_result = reduce_fn(successful_outputs)
        else:
            result.reduced_result = successful_outputs

        result.completed_at = datetime.now()
        return result

    async def map_only(
        self,
        items: List[T],
        map_fn: Callable[[T], Awaitable[R]]
    ) -> List[R]:
        """
        Execute map without reduce (convenience method).

        Args:
            items: Items to process
            map_fn: Async function to apply

        Returns:
            List of successful outputs
        """
        result = await self.execute(items, map_fn)
        return result.get_successful_outputs()


# Built-in reduce functions
def reduce_list(results: List[Any]) -> List[Any]:
    """Return results as list (default)"""
    return results


def reduce_flatten(results: List[List[Any]]) -> List[Any]:
    """Flatten nested lists"""
    flat = []
    for r in results:
        if isinstance(r, list):
            flat.extend(r)
        else:
            flat.append(r)
    return flat


def reduce_concat(results: List[str]) -> str:
    """Concatenate strings"""
    return "\n".join(str(r) for r in results if r)


def reduce_sum(results: List[Union[int, float]]) -> Union[int, float]:
    """Sum numeric results"""
    return sum(r for r in results if isinstance(r, (int, float)))


def reduce_max(results: List[Any]) -> Any:
    """Get maximum value"""
    return max(results) if results else None


def reduce_min(results: List[Any]) -> Any:
    """Get minimum value"""
    return min(results) if results else None


def reduce_count(results: List[Any]) -> int:
    """Count non-None results"""
    return sum(1 for r in results if r is not None)


def reduce_filter(predicate: Callable[[Any], bool]) -> Callable[[List[Any]], List[Any]]:
    """Create filter reduce function"""
    def reducer(results: List[Any]) -> List[Any]:
        return [r for r in results if predicate(r)]
    return reducer


def reduce_first(results: List[Any]) -> Any:
    """Get first non-None result"""
    for r in results:
        if r is not None:
            return r
    return None


def reduce_dict_merge(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge all dictionaries"""
    merged = {}
    for r in results:
        if isinstance(r, dict):
            merged.update(r)
    return merged
