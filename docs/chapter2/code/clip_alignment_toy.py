"""用手工向量复算 CLIP 的双向对比目标；不是 CLIP 模型或训练器。

在仓库根目录运行：python docs/chapter2/code/clip_alignment_toy.py
只使用 Python 标准库；不下载权重、不访问网络、不需要 GPU。
"""

import math


def normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("零向量不能计算余弦相似度")
    return [value / norm for value in vector]


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("点积要求两边维度相同")
    return sum(a * b for a, b in zip(left, right))


def softmax(logits):
    # 减去最大值不改变比例，避免 exp(很大的数) 溢出。
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    return [value / sum(weights) for value in weights]


def contrastive_loss(images, texts, temperature):
    if temperature <= 0:
        raise ValueError("temperature 必须为正数")
    if not images or len(images) != len(texts):
        raise ValueError("需要数量相同且非空的配对图像、文本向量")
    images = [normalize(vector) for vector in images]
    texts = [normalize(vector) for vector in texts]
    logits = [[dot(image, text) / temperature for text in texts] for image in images]
    rows = [softmax(row) for row in logits]
    columns = [softmax(column) for column in zip(*logits)]
    # 每行/列的正确配对都位于索引 i；这里用 logsumexp 形式防止 log(0)。
    def mean_cross_entropy(scores):
        losses = []
        for i, row in enumerate(scores):
            peak = max(row)
            losses.append(peak + math.log(sum(math.exp(x - peak) for x in row)) - row[i])
        return sum(losses) / len(losses)

    loss_i = mean_cross_entropy(logits)
    loss_t = mean_cross_entropy(list(zip(*logits)))
    return logits, rows, columns, loss_i, loss_t, (loss_i + loss_t) / 2


def main():
    # 4x4 灰度图，2x2 patch；按先行后列切块，每块也按行展平。
    pixels = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    patches = [
        [pixels[r + dr][c + dc] for dr in range(2) for dc in range(2)]
        for r in (0, 2) for c in (0, 2)
    ]
    # 人工选的 4x2 投影矩阵：比较左右两列。真实 ViT 的投影由训练学习。
    weights = [[1, 0], [0, 1], [1, 0], [0, 1]]
    embeddings = [[dot(patch, column) for column in zip(*weights)] for patch in patches]
    assert patches[0] == [1, 2, 5, 6]
    assert embeddings == [[6, 8], [10, 12], [22, 24], [26, 28]]
    print("patches (4, 4):", patches)
    print("projected (4, 2):", embeddings)

    # 这些是独立指定的“编码器输出”，并非从上面的像素计算而来。
    images = [[1, 0], [0, 1]]
    texts = [[1, 0], [3, 4]]
    logits, rows, columns, loss_i, loss_t, loss = contrastive_loss(images, texts, 0.5)
    print("logits:", [[round(x, 4) for x in row] for row in logits])
    print("image->text:", [[round(x, 4) for x in row] for row in rows])
    print("text->image:", [[round(x, 4) for x in row] for row in columns])
    print(f"loss_i={loss_i:.4f}, loss_t={loss_t:.4f}, mean={loss:.4f}")
    print(f"tau=1.0: mean={contrastive_loss(images, texts, 1.0)[-1]:.4f}")
    print(f"swapped pairs: mean={contrastive_loss(images, texts[::-1], 0.5)[-1]:.4f}")
    print("one candidate:", [round(x, 4) for x in softmax([2.0])])
    print("duplicate candidate:", [round(x, 4) for x in softmax([2.0, 1.2, 2.0])])

    assert math.isclose(loss, 0.29873616756976035, abs_tol=1e-10)
    assert all(math.isclose(sum(row), 1.0) for row in rows + columns)
    assert round(contrastive_loss(images, texts, 1.0)[-1], 4) == 0.4489
    assert contrastive_loss(images, texts[::-1], 0.5)[-1] > loss
    assert math.isclose(contrastive_loss([[1, 0]], [[0, 1]], 0.5)[-1], 0.0)
    assert normalize([3, 4]) == [0.6, 0.8]
    print("All toy checks passed.")


if __name__ == "__main__":
    main()
