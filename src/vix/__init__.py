"""VIX — Vision Integrity eXplainability.

A Data-Centric AI "data gatekeeper" that combines YOLO confidence and DINOv2
embedding distance to surface edge cases and label-definition drift, built as a
thin layer on top of FiftyOne.

Architecture rule (v0.1): :mod:`vix.core` has **zero** FiftyOne dependency and is
fully unit-testable on its own. Only :mod:`vix.adapters.fiftyone_adapter` imports
FiftyOne, behind the :class:`vix.adapters.base.DatasetAdapter` protocol.
"""

__version__ = "0.1.0"
