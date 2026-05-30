"""Adapters — the only place FiftyOne is touched.

:class:`base.DatasetAdapter` is the seam. :class:`memory.InMemoryAdapter` is a
FiftyOne-free implementation used for tests and dry-runs;
:class:`fiftyone_adapter.FiftyOneAdapter` is the real one (lazy-imports FiftyOne).
Swapping away from FiftyOne later means rewriting only this package.
"""
