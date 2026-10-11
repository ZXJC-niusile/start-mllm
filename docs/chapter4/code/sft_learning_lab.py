"""CPU-only teaching example; no model, tokenizer, training, or external packages."""
import math

IGNORE = -100
INPUT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
LABELS = [IGNORE] * 6 + [7, 8, 9, IGNORE]
# Each entry is a miniature output distribution at that input position.
# Only three positions have a supervised next token in this example.
DISTRIBUTIONS = [{} for _ in INPUT_IDS]
DISTRIBUTIONS[5] = {7: 0.5, 8: 0.5}
DISTRIBUTIONS[6] = {8: 0.25, 7: 0.75}
DISTRIBUTIONS[7] = {9: 0.8, 7: 0.2}


def supervised_pairs(labels):
    """Model-side shift: output at t is compared to the target at t+1."""
    return [(t, labels[t + 1]) for t in range(len(labels) - 1)
            if labels[t + 1] != IGNORE]


def shifted_loss(distributions, labels):
    if len(distributions) != len(labels):
        raise ValueError("positions must align before shifting")
    pairs = supervised_pairs(labels)
    if not pairs:
        raise ValueError("no supervised targets")
    losses = []
    for position, target in pairs:
        probability = distributions[position].get(target, 0.0)
        if not 0.0 < probability <= 1.0:
            raise ValueError("target probability must be in (0, 1]")
        losses.append(-math.log(probability))
    return sum(losses) / len(losses)


def lora_counts(d_in, d_out, rank):
    return rank * (d_in + d_out), d_in * d_out


def shared_groups(train, evaluation):
    return sorted({row["source_group"] for row in train}
                  & {row["source_group"] for row in evaluation})


def main():
    pairs = supervised_pairs(LABELS)
    assert pairs == [(5, 7), (6, 8), (7, 9)]
    loss = shifted_loss(DISTRIBUTIONS, LABELS)
    assert math.isclose(loss, -math.log(0.5 * 0.25 * 0.8) / 3)
    # Deliberately wrong: a collator pre-shifts labels and the model shifts again.
    double_shift = LABELS[1:] + [IGNORE]
    target_for_position_5 = dict(supervised_pairs(double_shift))[5]
    assert target_for_position_5 == 8
    try:
        shifted_loss(DISTRIBUTIONS, [IGNORE] * len(INPUT_IDS))
    except ValueError as exc:
        assert str(exc) == "no supervised targets"
    else:
        raise AssertionError("all-masked labels must fail")
    adapted, original = lora_counts(4096, 4096, 8)
    assert (adapted, original) == (65536, 16777216)
    assert lora_counts(1024, 2048, 4) == (12288, 2097152)
    rows = [{"id": "A-q1", "source_group": "receipt-A"},
            {"id": "B-q1", "source_group": "receipt-B"},
            {"id": "A-q2", "source_group": "receipt-A"}]
    row_overlap = shared_groups(rows[:2], rows[2:])
    train = [row for row in rows if row["source_group"] == "receipt-A"]
    evaluation = [row for row in rows if row["source_group"] != "receipt-A"]
    group_overlap = shared_groups(train, evaluation)
    assert row_overlap == ["receipt-A"] and not group_overlap
    assert len(train) + len(evaluation) == len(rows)
    print("targets after one shift:", [target for _, target in pairs])
    print(f"loss: {loss:.6f}")
    print(f"first target after double shift: {target_for_position_5} (wrong; expected 7)")
    print(f"LoRA parameters: {adapted} / {original} = {100 * adapted / original:.6f}%")
    print("row split shared groups:", row_overlap)
    print("group split shared groups:", group_overlap)
    print("self-tests: PASS")


if __name__ == "__main__":
    main()
