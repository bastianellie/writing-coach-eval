import random

from datasets import load_dataset


def sample_harmful_behaviors(n: int, seed: int = 42) -> list[dict]:
    """
    Sample `n` datapoints from the mlabonne/harmful_behaviors HuggingFace dataset.

    The dataset contains harmful instruction/response pairs and is intended for
    safety evaluation purposes. Each datapoint has the following fields:
        - source (str): The harmful instruction / prompt.
        - target (str): The expected harmful response.

    Args:
        n (int): Number of datapoints to sample.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        list[dict]: A list of `n` datapoints, each represented as a dictionary
                    with "source" and "target" keys.

    Raises:
        ValueError: If `n` is greater than the total number of available datapoints.
    """
    dataset = load_dataset("mlabonne/harmful_behaviors", split="train")

    total = len(dataset)
    if n > total:
        raise ValueError(
            f"Requested {n} samples but the dataset only contains {total} datapoints."
        )

    sampled = dataset.shuffle(seed=seed).select(range(n))
    return [{"text": row["text"]} for row in sampled]


_LABEL_MAP: dict[str, bool] = {"harmless": True, "harmful": False}


def sample_harmful_harmless_instructions(
    n: int,
    seed: int = 42,
    label_filter: str | None = None,
) -> list[dict]:
    """
    Sample `n` datapoints from the justinphan3110/harmful_harmless_instructions
    HuggingFace dataset.

    Each raw dataset row contains a pair of entries:
        - sentence (list[str]): A list of two instruction strings.
        - label (list[bool]): A list of two booleans, where the value at index
          `i` indicates whether the sentence at index `i` is harmless (True)
          or harmful (False).

    This function unpacks each row into individual (sentence, label) pairs
    before sampling.

    Args:
        n (int): Number of individual (sentence, label) pairs to sample.
        seed (int): Random seed for reproducibility. Defaults to 42.
        label_filter (str | None): If provided, only sample pairs matching
            this category. Accepted values are "harmless" (maps to True) or
            "harmful" (maps to False). Defaults to None (no filtering,
            sample from all unpacked pairs).

    Returns:
        list[dict]: A list of `n` datapoints, each represented as a dictionary
                    with "sentence" (str) and "label" (bool) keys.

    Raises:
        ValueError: If `label_filter` is not None, "harmless", or "harmful".
        ValueError: If `n` is greater than the total number of available
                    (filtered) unpacked pairs.
    """
    if label_filter is not None and label_filter not in _LABEL_MAP:
        raise ValueError(
            f"label_filter must be 'harmless', 'harmful', or None, got {label_filter!r}."
        )

    bool_filter: bool | None = _LABEL_MAP[label_filter] if label_filter is not None else None

    dataset = load_dataset(
        "justinphan3110/harmful_harmless_instructions", split="train"
    )

    # Unpack each row's paired lists into individual (sentence, label) dicts.
    pairs = [
        {"sentence": sentence, "label": label}
        for row in dataset
        for sentence, label in zip(row["sentence"], row["label"])
    ]

    if bool_filter is not None:
        pairs = [p for p in pairs if p["label"] == bool_filter]

    total = len(pairs)
    if n > total:
        raise ValueError(
            f"Requested {n} samples but the dataset only contains {total}"
            + (f" {label_filter}" if label_filter is not None else "")
            + " datapoints."
        )

    rng = random.Random(seed)
    return rng.sample(pairs, n)


if __name__ == "__main__":
    # out = sample_harmful_behaviors(5)
    out = sample_harmful_harmless_instructions(5, label_filter="harmful")
    print(out)
