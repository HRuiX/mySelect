"""
生成示例数据用于预实验测试
"""

import json
import random
import argparse
from pathlib import Path


# 示例数据模板
MATH_TEMPLATES = [
    {"instruction": "计算 {a} + {b} 的值", "output": "{c}"},
    {"instruction": "求解方程 {a}x + {b} = {d}", "output": "x = {e}"},
    {"instruction": "一个矩形的长为 {a} 厘米，宽为 {b} 厘米，求面积", "output": "面积 = {a} × {b} = {c} 平方厘米"},
    {"instruction": "计算 {a} 的 {b} 次方", "output": "{c}"},
    {"instruction": "将分数 {a}/{b} 化为小数", "output": "{e}"},
]

CODE_TEMPLATES = [
    {"instruction": "写一个 Python 函数计算斐波那契数列的第 n 项", "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
    {"instruction": "用 Python 实现冒泡排序", "output": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"},
    {"instruction": "写一个函数判断一个字符串是否是回文", "output": "def is_palindrome(s):\n    return s == s[::-1]"},
    {"instruction": "用 Python 实现二分查找", "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"},
    {"instruction": "写一个 Python 装饰器用于计时", "output": "import time\ndef timer(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} took {time.time()-start:.2f}s')\n        return result\n    return wrapper"},
]

INSTRUCTION_TEMPLATES = [
    {"instruction": "解释什么是机器学习中的过拟合现象", "output": "过拟合是指模型在训练数据上表现很好，但在未见过的测试数据上表现较差的现象。这通常是因为模型过度学习了训练数据中的噪声和细节，而不是学习到了真正的规律。"},
    {"instruction": "总结深度学习的三个关键组成部分", "output": "深度学习的三个关键组成部分是：1. 神经网络架构 2. 损失函数 3. 优化算法"},
    {"instruction": "什么是注意力机制？", "output": "注意力机制是一种让模型在处理输入时能够聚焦于最相关部分的技术。它通过计算查询与键的相似度来分配注意力权重，然后对值进行加权求和。"},
    {"instruction": "解释梯度下降的基本原理", "output": "梯度下降是一种优化算法，通过计算损失函数关于参数的梯度，沿梯度的反方向更新参数，逐步最小化损失函数。"},
    {"instruction": "什么是 Transformer 架构？", "output": "Transformer 是一种基于自注意力机制的神经网络架构，完全摒弃了循环和卷积结构。它由编码器和解码器组成，使用多头注意力和前馈网络。"},
]


def generate_math_sample(idx):
    template = random.choice(MATH_TEMPLATES)
    a = random.randint(1, 100)
    b = random.randint(1, 50)
    c = a + b if "+" in template["instruction"] else a * b
    d = random.randint(1, 200)
    e = round((d - b) / max(a, 1), 2)

    instruction = template["instruction"].format(a=a, b=b, c=c, d=d, e=e)
    output = template["output"].format(a=a, b=b, c=c, d=d, e=e)

    return {"id": f"math_{idx:05d}", "instruction": instruction, "output": output, "category": "math"}


def generate_code_sample(idx):
    template = random.choice(CODE_TEMPLATES)
    return {"id": f"code_{idx:05d}", "instruction": template["instruction"], "output": template["output"], "category": "code"}


def generate_instruction_sample(idx):
    template = random.choice(INSTRUCTION_TEMPLATES)
    return {"id": f"inst_{idx:05d}", "instruction": template["instruction"], "output": template["output"], "category": "instruction"}


def main():
    parser = argparse.ArgumentParser(description="生成示例数据")
    parser.add_argument("--output", "-o", default="./data/sample_data.jsonl", help="输出文件路径")
    parser.add_argument("--n-samples", "-n", type=int, default=1000, help="样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generators = [generate_math_sample, generate_code_sample, generate_instruction_sample]
    weights = [0.4, 0.3, 0.3]  # 40% math, 30% code, 30% instruction

    samples = []
    for i in range(args.n_samples):
        gen = random.choices(generators, weights=weights, k=1)[0]
        samples.append(gen(i))

    random.shuffle(samples)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"生成 {len(samples)} 个样本 -> {output_path}")

    # 统计
    categories = {}
    for s in samples:
        cat = s["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
