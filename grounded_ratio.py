import json
from collections import Counter

INPUT_FILE = "test_data/writing_coach_halu_detection_20260327_o3-mini_decisions.json"
GROUNDED_THRESHOLD = 85.0


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    counter = {"grounded": 0, "non-grounded": 0}
    entail_counter = Counter()

    results = data["results"]
    tot_claims = sum(result["# cited claims"] for result in results)
    print(f"Evaluating {len(results)} ({tot_claims} total claims) results with grounded threshold of {GROUNDED_THRESHOLD}% entailment...")

    for result in results:
        distribution = result["claim_distribution_per_summary"]
        entailment = distribution["entailment(%)"]
        unverifiable = distribution["unverifiable(%)"]
        tot = entailment + unverifiable * 0.87  # weight of amount of error on unverifiable

        if tot >= GROUNDED_THRESHOLD:
            counter["grounded"] += 1
        else:
            print(distribution)
            counter["non-grounded"] += 1

        for claim in result["claims"]:
            if claim["claim_decision"] == "Entailment":
                entail_counter.update(["Entailment"])
            elif "Unverifiable" in claim["claim_decision"]:
                entail_counter.update(["Unverifiable"])

    total = counter["grounded"] + counter["non-grounded"]
    grounded_pct = (counter["grounded"] / total * 100) if total > 0 else 0.0

    print(f"Total results   : {total}")
    print(f"Grounded        : {counter['grounded']}")
    print(f"Non-grounded    : {counter['non-grounded']}")
    print(f"Grounded ratio  : {grounded_pct:.1f}%")
    print(f"Entailment      : {entail_counter['Entailment']}")
    print(f"Unverifiable    : {entail_counter['Unverifiable']}")


if __name__ == "__main__":
    main()
