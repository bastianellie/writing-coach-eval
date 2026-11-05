from typing import Literal

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Jigsaw constants
# ---------------------------------------------------------------------------

TOXIC_ASPECTS = Literal[
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

ALL_ASPECTS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

AspectFilter = dict[str, int]
"""
A mapping from aspect name to the required binary value (0 or 1).

Examples::

    # Only rows labelled as toxic
    {"toxic": 1}

    # Rows that are both obscene AND an insult, but not a threat
    {"obscene": 1, "insult": 1, "threat": 0}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_toxic_comments(
    n: int,
    seed: int = 42,
    aspect_filter: AspectFilter | None = None,
    min_text_length: int | None = None,
    max_text_length: int | None = None,
) -> list[dict]:
    """
    Sample ``n`` rows from the
    `thesofakillers/jigsaw-toxic-comment-classification-challenge
    <https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge>`_
    HuggingFace dataset.

    Dataset structure
    -----------------
    Each row contains:

    * ``id`` (str): Unique comment identifier.
    * ``comment_text`` (str): The raw comment text.
    * ``toxic`` (int): 1 if the comment is toxic, 0 otherwise.
    * ``severe_toxic`` (int): 1 if the comment is severely toxic.
    * ``obscene`` (int): 1 if the comment contains obscene language.
    * ``threat`` (int): 1 if the comment contains a threat.
    * ``insult`` (int): 1 if the comment is insulting.
    * ``identity_hate`` (int): 1 if the comment contains identity-based hatred.

    All label columns hold binary integer values (0 or 1).  The full dataset
    contains 159 571 rows (``train`` split only).

    Filtering
    ---------
    ``aspect_filter`` is an optional dictionary that maps one or more aspect
    names to the required value (0 or 1).  A row is included only when **all**
    specified aspects match their required values simultaneously
    (logical AND).

    Valid aspect names are: ``"toxic"``, ``"severe_toxic"``, ``"obscene"``,
        ``"threat"``, ``"insult"``, ``"identity_hate"``.

    Examples::

        # No filter – sample from the full dataset
        sample_toxic_comments(10)

        # Only rows flagged as toxic
        sample_toxic_comments(10, aspect_filter={"toxic": 1})

        # Rows that are both obscene AND insulting, but not a threat
        sample_toxic_comments(
            10,
            aspect_filter={"obscene": 1, "insult": 1, "threat": 0},
        )

        # Rows with no toxic label at all (clean comments)
        sample_toxic_comments(
            10,
            aspect_filter={aspect: 0 for aspect in ALL_ASPECTS},
        )

        # Only rows whose comment_text is between 50 and 200 characters
        sample_toxic_comments(10, min_text_length=50, max_text_length=200)

    Args:
        n (int): Number of rows to sample.
        seed (int): Random seed for reproducibility.  Defaults to 42.
        aspect_filter (AspectFilter | None): Multi-aspect filter dictionary.
            Keys must be valid aspect names; values must be 0 or 1.
            If *None*, no filtering is applied.  Defaults to *None*.
        min_text_length (int | None): Minimum character length (inclusive) of
            ``comment_text`` to consider.  If *None*, no lower bound is
            applied.  Defaults to *None*.
        max_text_length (int | None): Maximum character length (inclusive) of
            ``comment_text`` to consider.  If *None*, no upper bound is
            applied.  Defaults to *None*.

    Returns:
        list[dict]: A list of ``n`` rows, each a dictionary with keys
        ``"id"``, ``"comment_text"``, ``"toxic"``, ``"severe_toxic"``,
        ``"obscene"``, ``"threat"``, ``"insult"``, and ``"identity_hate"``.

    Raises:
        ValueError: If any key in ``aspect_filter`` is not a valid aspect name.
        ValueError: If any value in ``aspect_filter`` is not 0 or 1.
        ValueError: If ``min_text_length`` or ``max_text_length`` is negative.
        ValueError: If ``min_text_length`` > ``max_text_length`` when both are
            provided.
        ValueError: If ``n`` is greater than the number of rows that satisfy
            the filter.
    """
    if aspect_filter is not None:
        _validate_aspect_filter(aspect_filter)
    _validate_text_length_bounds(min_text_length, max_text_length)

    dataset = load_dataset(
        "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        split="train",
    )

    if aspect_filter:
        dataset = dataset.filter(
            lambda row: all(
                row[aspect] == value for aspect, value in aspect_filter.items()
            )
        )

    if min_text_length is not None or max_text_length is not None:
        dataset = dataset.filter(
            lambda row: (min_text_length is None or len(row["comment_text"]) >= min_text_length)
            and (max_text_length is None or len(row["comment_text"]) <= max_text_length)
        )

    total = len(dataset)
    if n > total:
        filter_desc = f" matching filter {aspect_filter}" if aspect_filter else ""
        raise ValueError(
            f"Requested {n} samples but only {total} rows are available"
            f"{filter_desc}."
        )

    sampled = dataset.shuffle(seed=seed).select(range(n))
    return [
        {
            "id": row["id"],
            "comment_text": row["comment_text"],
            "toxic": row["toxic"],
            "severe_toxic": row["severe_toxic"],
            "obscene": row["obscene"],
            "threat": row["threat"],
            "insult": row["insult"],
            "identity_hate": row["identity_hate"],
        }
        for row in sampled
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_aspect_filter(aspect_filter: AspectFilter) -> None:
    """Raise ``ValueError`` if *aspect_filter* contains invalid keys or values."""
    invalid_keys = set(aspect_filter) - set(ALL_ASPECTS)
    if invalid_keys:
        raise ValueError(
            f"Unknown aspect(s): {invalid_keys}. "
            f"Valid aspects are: {set(ALL_ASPECTS)}."
        )

    invalid_values = {k: v for k, v in aspect_filter.items() if v not in (0, 1)}
    if invalid_values:
        raise ValueError(
            f"Aspect values must be 0 or 1, got: {invalid_values}."
        )


def _validate_text_length_bounds(
    min_text_length: int | None,
    max_text_length: int | None,
) -> None:
    """Raise ``ValueError`` if text-length bounds are invalid."""
    if min_text_length is not None and min_text_length < 0:
        raise ValueError(
            f"min_text_length must be non-negative, got {min_text_length}."
        )
    if max_text_length is not None and max_text_length < 0:
        raise ValueError(
            f"max_text_length must be non-negative, got {max_text_length}."
        )
    if (
        min_text_length is not None
        and max_text_length is not None
        and min_text_length > max_text_length
    ):
        raise ValueError(
            f"min_text_length ({min_text_length}) must be ≤ "
            f"max_text_length ({max_text_length})."
        )


# ---------------------------------------------------------------------------
# Civil Comments constants
# ---------------------------------------------------------------------------

CIVIL_ASPECTS = Literal[
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

ALL_CIVIL_ASPECTS: tuple[str, ...] = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
)

_CIVIL_VALID_VALUES: frozenset[float] = frozenset(round(x * 0.1, 1) for x in range(11))

CivilAspectFilter = dict[str, tuple[float, float]]
"""
A mapping from aspect name to an inclusive ``(min, max)`` range of accepted
float values.

Aspect scores in the dataset run from 0.0 to 1.0 in steps of 0.1.

Examples::

    # Rows where toxicity is at least 0.5
    {"toxicity": (0.5, 1.0)}

    # Rows with a moderate threat score and no sexual content
    {"threat": (0.2, 0.5), "sexual_explicit": (0.0, 0.0)}
"""


# ---------------------------------------------------------------------------
# Civil Comments public API
# ---------------------------------------------------------------------------


def sample_civil_comments(
    n: int,
    seed: int = 42,
    aspect_filter: CivilAspectFilter | None = None,
    min_text_length: int | None = None,
    max_text_length: int | None = None,
) -> list[dict]:
    """
    Sample ``n`` rows from the `google/civil_comments
    <https://huggingface.co/datasets/google/civil_comments>`_ HuggingFace
    dataset.

    Dataset structure
    -----------------
    Each row contains:

    * ``text`` (str): The raw comment text.
    * ``toxicity`` (float): Overall toxicity score.
    * ``severe_toxicity`` (float): Severe toxicity score.
    * ``obscene`` (float): Obscene language score.
    * ``threat`` (float): Threat score.
    * ``insult`` (float): Insult score.
    * ``identity_attack`` (float): Identity-based attack score.
    * ``sexual_explicit`` (float): Sexually explicit content score.

    All label columns hold ``float32`` values from 0.0 to 1.0 in steps of
    0.1.  The ``train`` split contains 1 804 874 rows.

    Filtering
    ---------
    ``aspect_filter`` is an optional dictionary that maps one or more aspect
    names to an inclusive ``(min, max)`` range.  A row is included only when
    **all** specified aspects fall within their respective ranges
    simultaneously (logical AND).

    Both ``min`` and ``max`` must be valid aspect values (0.0 – 1.0, steps of
    0.1), and ``min`` must be ≤ ``max``.

    Valid aspect names are: ``"toxicity"``, ``"severe_toxicity"``,
    ``"obscene"``, ``"threat"``, ``"insult"``, ``"identity_attack"``,
    ``"sexual_explicit"``.

    Examples::

        # No filter – sample from the full dataset
        sample_civil_comments(10)

        # Only rows where toxicity is at least 0.5
        sample_civil_comments(10, aspect_filter={"toxicity": (0.5, 1.0)})

        # Rows with a moderate threat score and no sexual content
        sample_civil_comments(
            10,
            aspect_filter={"threat": (0.2, 0.5), "sexual_explicit": (0.0, 0.0)},
        )

        # Rows with no toxic signal at all (clean comments)
        sample_civil_comments(
            10,
            aspect_filter={aspect: (0.0, 0.0) for aspect in ALL_CIVIL_ASPECTS},
        )

        # Only rows whose text is between 50 and 300 characters
        sample_civil_comments(10, min_text_length=50, max_text_length=300)

    Args:
        n (int): Number of rows to sample.
        seed (int): Random seed for reproducibility.  Defaults to 42.
        aspect_filter (CivilAspectFilter | None): Multi-aspect filter
            dictionary.  Keys must be valid aspect names; values must be
            ``(min, max)`` tuples of valid float scores (0.0 – 1.0, steps of
            0.1) with ``min`` ≤ ``max``.  If *None*, no filtering is applied.
            Defaults to *None*.
        min_text_length (int | None): Minimum character length (inclusive) of
            ``text`` to consider.  If *None*, no lower bound is applied.
            Defaults to *None*.
        max_text_length (int | None): Maximum character length (inclusive) of
            ``text`` to consider.  If *None*, no upper bound is applied.
            Defaults to *None*.

    Returns:
        list[dict]: A list of ``n`` rows, each a dictionary with keys
        ``"text"``, ``"toxicity"``, ``"severe_toxicity"``, ``"obscene"``,
        ``"threat"``, ``"insult"``, ``"identity_attack"``, and
        ``"sexual_explicit"``.

    Raises:
        ValueError: If any key in ``aspect_filter`` is not a valid aspect name.
        ValueError: If any bound in ``aspect_filter`` is not a valid score
            value (0.0 – 1.0, steps of 0.1).
        ValueError: If ``min`` > ``max`` for any aspect range.
        ValueError: If ``min_text_length`` or ``max_text_length`` is negative.
        ValueError: If ``min_text_length`` > ``max_text_length`` when both are
            provided.
        ValueError: If ``n`` is greater than the number of rows that satisfy
            the filter.
    """
    if aspect_filter is not None:
        _validate_civil_aspect_filter(aspect_filter)
    _validate_text_length_bounds(min_text_length, max_text_length)

    dataset = load_dataset("google/civil_comments", split="train")

    if aspect_filter:
        dataset = dataset.filter(
            lambda row: all(
                lo <= row[aspect] <= hi
                for aspect, (lo, hi) in aspect_filter.items()
            )
        )

    if min_text_length is not None or max_text_length is not None:
        dataset = dataset.filter(
            lambda row: (min_text_length is None or len(row["text"]) >= min_text_length)
            and (max_text_length is None or len(row["text"]) <= max_text_length)
        )

    total = len(dataset)
    if n > total:
        filter_desc = f" matching filter {aspect_filter}" if aspect_filter else ""
        raise ValueError(
            f"Requested {n} samples but only {total} rows are available"
            f"{filter_desc}."
        )

    sampled = dataset.shuffle(seed=seed).select(range(n))
    return [
        {
            "text": row["text"],
            "toxicity": row["toxicity"],
            "severe_toxicity": row["severe_toxicity"],
            "obscene": row["obscene"],
            "threat": row["threat"],
            "insult": row["insult"],
            "identity_attack": row["identity_attack"],
            "sexual_explicit": row["sexual_explicit"],
        }
        for row in sampled
    ]


# ---------------------------------------------------------------------------
# Civil Comments helpers
# ---------------------------------------------------------------------------


def _validate_civil_aspect_filter(aspect_filter: CivilAspectFilter) -> None:
    """Raise ``ValueError`` if *aspect_filter* contains invalid keys or ranges."""
    invalid_keys = set(aspect_filter) - set(ALL_CIVIL_ASPECTS)
    if invalid_keys:
        raise ValueError(
            f"Unknown aspect(s): {invalid_keys}. "
            f"Valid aspects are: {set(ALL_CIVIL_ASPECTS)}."
        )

    for aspect, (lo, hi) in aspect_filter.items():
        for bound_name, bound in (("min", lo), ("max", hi)):
            rounded = round(bound, 1)
            if rounded not in _CIVIL_VALID_VALUES:
                raise ValueError(
                    f"Invalid {bound_name} value {bound!r} for aspect {aspect!r}. "
                    f"Values must be in the range [0.0, 1.0] in steps of 0.1."
                )
        if lo > hi:
            raise ValueError(
                f"min ({lo}) must be ≤ max ({hi}) for aspect {aspect!r}."
            )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Jigsaw: obscene + insult ===")
    jigsaw_out = sample_toxic_comments(3, aspect_filter={"obscene": 1, "insult": 1}, min_text_length=30)
    for row in jigsaw_out:
        print(row)

    print("\n=== Civil Comments: threat 0.1–0.2 ===")
    civil_out = sample_civil_comments(3, aspect_filter={"threat": (0.1, 0.2)}, min_text_length=130)
    for row in civil_out:
        print(row)
