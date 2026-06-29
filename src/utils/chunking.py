"""Shared keyword-chunking primitive - Claude Generated.

The classic pipeline (``PipelineStepExecutor._execute_chunked_keyword_analysis``)
and the agentic v4 path (``LLMAgentStep``) both split an oversized keyword list
into equal chunks with identical arithmetic. This is the single home for that
algorithm so the two paths cannot drift apart.
"""

from __future__ import annotations

from typing import Any, List


def split_into_equal_chunks(items: List[Any], threshold: int) -> List[List[Any]]:
    """Split ``items`` into equal chunks using classic-pipeline semantics.

    At or below ``threshold`` everything goes into ONE chunk. Above it, items are
    distributed into EQUAL chunks (2 chunks up to 1.5×threshold, otherwise
    ``ceil(total/threshold)``) rather than fixed-size slices with a small tail.
    The remainder is spread one item at a time across the first chunks, so chunk
    sizes differ by at most one and no item is lost or reordered.
    """
    total = len(items)
    if total == 0:
        return []
    if threshold <= 0 or total <= threshold:
        return [list(items)]

    if total <= threshold * 1.5:
        num_chunks = 2
    else:
        num_chunks = max(2, (total + threshold - 1) // threshold)

    base_size = total // num_chunks
    remainder = total % num_chunks
    chunks: List[List[Any]] = []
    start = 0
    for i in range(num_chunks):
        size = base_size + (1 if i < remainder else 0)
        chunks.append(list(items[start:start + size]))
        start += size
    return chunks
