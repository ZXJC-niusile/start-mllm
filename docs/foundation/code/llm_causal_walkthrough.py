"""手算 LLM 数据流；仅用标准库，不训练模型、不下载权重。

从仓库根运行：python docs/foundation/code/llm_causal_walkthrough.py
反例：python docs/foundation/code/llm_causal_walkthrough.py --no-mask
"""

import argparse
import math
import random


VOCAB = ["我", "爱", "猫", "<eos>"]
EMBEDDINGS = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
WQ = [[0.0], [math.log(3)]]
WK = [[0.0], [1.0]]
WV = [[1.0, 0.0], [0.0, 1.0]]
WOUT = [[0.0] * 4, [4 * math.log(n) / 3 for n in (1, 2, 3, 4)]]


def matmul(a, b):
    """二维矩阵乘法：[行, 内维] @ [内维, 列]。"""
    if not a or not b or any(len(row) != len(b) for row in a):
        raise ValueError("矩阵乘法内侧维度不匹配")
    if any(len(row) != len(b[0]) for row in b):
        raise ValueError("右矩阵各行必须等长")
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def softmax(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("采样温度必须大于 0；贪心选择直接用 argmax")
    scaled = [x / temperature for x in logits]
    largest = max(scaled)
    exps = [math.exp(x - largest) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def attention(x, causal=True):
    q, k, v = matmul(x, WQ), matmul(x, WK), matmul(x, WV)
    # 本例 d_k=1，所以 sqrt(d_k)=1；仍保留一般写法。
    scores = matmul(q, list(map(list, zip(*k))))
    scores = [[s / math.sqrt(len(k[0])) for s in row] for row in scores]
    weights = []
    for i, row in enumerate(scores):
        masked = [s if not causal or j <= i else -math.inf for j, s in enumerate(row)]
        weights.append(softmax(masked))
    return weights, matmul(weights, v)


def top_p(probs, threshold):
    if not 0 < threshold <= 1:
        raise ValueError("top-p 必须在 (0, 1] 内")
    chosen, total = [], 0.0
    for token_id in sorted(range(len(probs)), key=lambda i: probs[i], reverse=True):
        chosen.append(token_id)
        total += probs[token_id]
        if total >= threshold:
            break
    return chosen, [probs[i] / total for i in chosen]


def close(a, b):
    if len(a) != len(b) or any(len(r) != len(s) for r, s in zip(a, b)):
        return False
    return all(math.isclose(x, y, abs_tol=1e-12) for r, s in zip(a, b) for x, y in zip(r, s))


def fmt(values):
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-mask", action="store_true", help="移除因果掩码，观察未来泄漏")
    args = parser.parse_args()
    # 刻意用空格分词的小词表，不冒充真实模型的 tokenizer。
    ids = [VOCAB.index(piece) for piece in "我 爱 猫".split()]
    x = [EMBEDDINGS[i][:] for i in ids]
    weights, hidden = attention(x, causal=not args.no_mask)
    logits = matmul(hidden, WOUT)
    probs = [softmax(row) for row in logits]
    labels = ids[1:] + [VOCAB.index("<eos>")]
    losses = [-math.log(row[target]) for row, target in zip(probs, labels)]
    altered = [row[:] for row in x]
    altered[2] = [9.0, -2.0]  # 只改变第三个位置，前两位置输入完全不动。
    _, changed_hidden = attention(altered, causal=not args.no_mask)
    prefix_unchanged = close(hidden[:2], changed_hidden[:2])
    print("mode:", "UNMASKED (counterexample)" if args.no_mask else "CAUSAL")
    print("ids:", ids, "labels:", labels)
    print("shapes: X=(3,2) Q=K=(3,1) V=(3,2) A=(3,3) H=(3,2) logits=(3,4)")
    print("position 2 attention:", fmt(weights[1]))
    print("position 2 hidden:", fmt(hidden[1]))
    print("position 2 probabilities:", fmt(probs[1]))
    print("per-position loss:", fmt(losses))
    print(f"mean loss: {sum(losses) / len(losses):.6f}")
    print("future change leaves first 2 positions unchanged:", prefix_unchanged)
    # 下列检查始终用正确的因果版本，对照本章的精确手算结果。
    a, h = attention(x)
    p = softmax(matmul(h, WOUT)[1])
    assert close([a[1]], [[0.25, 0.75, 0.0]])
    assert close([h[1]], [[0.25, 0.75]])
    assert close([p], [[0.1, 0.2, 0.3, 0.4]])
    assert math.isclose(-math.log(p[labels[1]]), -math.log(0.3))
    assert close(h[:2], attention(altered)[1][:2])
    assert close(h[:2], attention(x[:2])[1])  # 整段并行与独立前缀一致。
    assert not close(attention(x, False)[1][:2], attention(altered, False)[1][:2])
    assert prefix_unchanged == (not args.no_mask)
    assert not close([], [[1.0]])
    assert not close([[1.0]], [[1.0, 2.0]])
    assert all(math.isclose(sum(row), 1.0) for row in a)
    assert close([softmax([1000, 1001])], [softmax([0, 1])])
    assert top_p([0.60, 0.25, 0.15], 0.80)[0] == [0, 1]
    assert top_p([0.60, 0.25, 0.15], 0.90)[0] == [0, 1, 2]
    candidates, normalized = top_p(p, 0.65)
    assert candidates == [3, 2]
    assert close([normalized], [[4 / 7, 3 / 7]])
    draw = random.Random(7).choices(candidates, weights=normalized, k=1)[0]
    print("causal baseline top-p=0.65 candidates:", [VOCAB[i] for i in candidates], "weights:", fmt(normalized))
    print("one seeded sample:", VOCAB[draw], "(not an accuracy test)")
    print("checks: PASS (numeric values, normalized rows, shifted loss, causal invariance, top-p)")


if __name__ == "__main__":
    main()
