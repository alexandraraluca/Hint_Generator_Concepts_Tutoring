"""JSON schemas for all annotation artifacts (Stage 2).

We use jsonschema (draft 2020-12) to validate every artifact at write time.
Keeping the schemas in code (rather than separate .json files) means the
project is self-contained and editable in one place.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

CONCEPT_DAG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ConceptsDAG",
    "type": "object",
    "required": ["version", "concepts", "edges"],
    "properties": {
        "version": {"type": "string"},
        "categories": {
            "type": "array",
            "items": {"type": "string"},
        },
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "category"],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "array",
            "description": "DAG edges: 'from' is a prerequisite of 'to'",
            "items": {
                "type": "object",
                "required": ["from", "to"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["prerequisite", "extends", "uses"],
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

PROBLEMS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ProblemsAnnotations",
    "type": "object",
    "required": ["version", "problems"],
    "properties": {
        "version": {"type": "string"},
        "problems": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["problem_id", "year", "tema", "pid", "concepts"],
                "properties": {
                    "problem_id": {"type": "string"},
                    "year": {"type": "string"},
                    "tema": {"type": "string"},
                    "pid": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "concept ids referencing concepts_dag.json",
                    },
                    "primary_concept": {"type": "string"},
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard", "very_hard"],
                    },
                    "expected_complexity": {"type": "string"},
                    "common_pitfalls": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "annotation_source": {
                        "type": "string",
                        "enum": ["manual", "llm", "llm+human"],
                    },
                    "llm_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

TESTS_LABELS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TestsLabels",
    "type": "object",
    "required": ["version", "tests"],
    "properties": {
        "version": {"type": "string"},
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["problem_id", "test_id", "size_class"],
                "properties": {
                    "problem_id": {"type": "string"},
                    "test_id": {"type": "string"},
                    "size_class": {
                        "type": "string",
                        "enum": ["tiny", "small", "medium", "large", "stress"],
                    },
                    "input_bytes": {"type": "integer"},
                    "n_param_estimate": {"type": ["integer", "null"]},
                    "edge_case": {"type": "boolean"},
                    "tested_concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "discriminates": {
                        "type": "string",
                        "enum": [
                            "correctness",
                            "complexity",
                            "edge_case",
                            "memory",
                            "io_format",
                        ],
                    },
                    "annotation_source": {
                        "type": "string",
                        "enum": ["rule", "llm", "manual"],
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

ERRORS_TAXONOMY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ErrorsTaxonomy",
    "type": "object",
    "required": ["version", "L1", "L2", "L3"],
    "properties": {
        "version": {"type": "string"},
        "L1": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "L2": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "examples"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "applies_to": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
            },
        },
        "L3": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "concept_id"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "concept_id": {
                        "type": "string",
                        "description": "concept this error is tied to in DAG",
                    },
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

HINT_RUBRIC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HintSet",
    "description": (
        "A graded set of 1-4 hints for a single (problem, code) pair. "
        "Encodes the rubric criteria used everywhere downstream."
    ),
    "type": "object",
    "required": ["problem_id", "hints"],
    "properties": {
        "problem_id": {"type": "string"},
        "anon_id": {"type": ["string", "null"]},
        "submission_name": {"type": ["string", "null"]},
        "language": {
            "type": "string",
            "enum": ["cpp", "java", "unknown"],
        },
        "verdict": {
            "type": "string",
            "enum": ["OK", "WA", "TLE", "RE", "CE", "MLE", "OTHER"],
        },
        "issues": {"type": "array", "items": {"type": "string"}},
        "concepts_targeted": {
            "type": "array",
            "items": {"type": "string"},
        },
        "hints": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "required": ["level", "text"],
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["macro", "structural", "specific", "very_specific"],
                    },
                    "text": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": 400,
                    },
                    "concept_id": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "source": {
            "type": "string",
            "enum": ["silver_diff", "llm_bootstrap", "gold_human"],
        },
        "validator_passed": {"type": "boolean"},
        "validator_violations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": False,
}


_VALIDATORS = {
    "concepts_dag": Draft202012Validator(CONCEPT_DAG_SCHEMA),
    "problems": Draft202012Validator(PROBLEMS_SCHEMA),
    "tests_labels": Draft202012Validator(TESTS_LABELS_SCHEMA),
    "errors_taxonomy": Draft202012Validator(ERRORS_TAXONOMY_SCHEMA),
    "hints": Draft202012Validator(HINT_RUBRIC_SCHEMA),
}


def validate(name: str, obj: Any) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    if name not in _VALIDATORS:
        raise KeyError(f"unknown schema {name!r}")
    return [
        f"{'/'.join(map(str, e.absolute_path))}: {e.message}"
        for e in _VALIDATORS[name].iter_errors(obj)
    ]
