"""基础篇 01：用 Python 列表观察张量运算；不依赖第三方包。

这里只实现非空、规则矩阵和共享权重的批量投影，不是通用张量库。
"""


def shape(value):
    """逐层检查列表；拒绝长短不一的行，避免静默截断。"""
    if not isinstance(value, list):
        return ()
    if not value:
        raise ValueError("本练习不支持空列表")
    child_shape = shape(value[0])
    if any(shape(child) != child_shape for child in value):
        raise ValueError("每一层必须等长，不能是参差列表")
    return (len(value),) + child_shape


def dot(u, v):
    if len(shape(u)) != 1 or shape(u) != shape(v):
        raise ValueError("点积需要两个等长向量")
    return sum(a * b for a, b in zip(u, v))


def elementwise(u, v):
    if len(shape(u)) != 1 or shape(u) != shape(v):
        raise ValueError("本练习的逐元素乘需要两个等长向量")
    return [a * b for a, b in zip(u, v)]


def transpose(matrix):
    if len(shape(matrix)) != 2:
        raise ValueError("本练习只转置二维矩阵")
    return [list(column) for column in zip(*matrix)]


def matmul(left, right):
    left_shape, right_shape = shape(left), shape(right)
    if len(left_shape) != 2 or len(right_shape) != 2:
        raise ValueError("matmul 需要两个二维矩阵")
    if left_shape[1] != right_shape[0]:
        raise ValueError("矩阵乘法的内侧维度必须相等")
    columns = transpose(right)
    return [[dot(row, column) for column in columns] for row in left]


def project_batch(features, weight):
    if len(shape(features)) != 3:
        raise ValueError("features 必须是 [B,N,D] 三维列表")
    # 每张图的 N 个 token 使用同一份权重；不把不同图片混起来。
    return [matmul(one_image, weight) for one_image in features]


def must_reject(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("应当拒绝错误形状，却继续计算了")


def main():
    u, v = [1, 2, 3], [4, 5, 6]
    x = [[1, 2, 0], [0, 1, 3]]
    w = [[1, 0], [0, 1], [1, -1]]
    features = [x, [[2, 0, 1], [1, 1, 1]]]
    y = project_batch(features, w)
    assert dot(u, v) == 32
    assert elementwise(u, v) == [4, 10, 18]
    assert matmul(x, w) == [[1, 2], [3, -2]]
    assert y == [[[1, 2], [3, -2]], [[3, -1], [2, 0]]]
    assert shape(y) == (2, 2, 2)
    assert matmul(x, transpose(x)) == [[5, 2], [2, 10]]
    # 批量投影等于逐张投影。改第二张图片，不应改变第一张输出。
    changed = [features[0], [[0, 0, 0], [0, 0, 0]]]
    assert project_batch(changed, w)[0] == y[0]
    must_reject(lambda: dot([1, 2], [3]))
    must_reject(lambda: shape([[1, 2], [3]]))
    must_reject(lambda: matmul(x, w[:-1]))
    must_reject(lambda: project_batch(x, w))
    print("dot:", dot(u, v))
    print("elementwise:", elementwise(u, v))
    print("X @ W:", matmul(x, w))
    print("F shape:", shape(features))
    print("Y shape:", shape(y))
    print("Y:", y)
    print("X @ X.T:", matmul(x, transpose(x)))
    print("self-test: PASS (values, shapes, batch isolation, invalid inputs)")


if __name__ == "__main__":
    main()
