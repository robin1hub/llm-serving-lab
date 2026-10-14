"""纯标准库的新增测试矩阵检查；不读取GPU或修改训练数据。"""

import json
from pathlib import Path


def load_matrix_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan["schema_version"] != 1 or plan["purpose"] != "frozen_test_only_half_repeat_half_novel":
        raise ValueError("未知矩阵用途或版本")
    for key in ("seed", "token_base", "warmups", "repeats"):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f"{key}必须是正整数")
    shapes = plan["shapes"]
    ids = set()
    for shape in shapes:
        if shape["phase"] not in ("Prefill", "Decode") or shape["split"] not in ("repeat", "novel"):
            raise ValueError("本矩阵只接受纯P/D测试形状")
        if type(shape["batch_size"]) is not int or not 1 <= shape["batch_size"] <= 8:
            raise ValueError("batch人数需在1至8之间")
        if type(shape["length"]) is not int or not 0 < shape["length"] <= 4000:
            raise ValueError("长度需为正整数且保留输出空间")
        key = shape["phase"], shape["batch_size"], shape["length"]
        if key in ids:
            raise ValueError("重复形状")
        ids.add(key)
    if len(shapes) != 16 or sum(shape["split"] == "novel" for shape in shapes) != 8:
        raise ValueError("要求16形状且一半为新形状")
    return plan
