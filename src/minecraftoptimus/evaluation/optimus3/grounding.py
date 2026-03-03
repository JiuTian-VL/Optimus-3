import argparse
import json
import re

import numpy as np
from rich import print


def calculate_iou(bbox_pred, bbox_gt):
    """
    计算两个边界框之间的 IoU (Intersection over Union)。

    Args:
        bbox_pred (list or tuple): 预测的边界框 [x1, y1, x2, y2]。
                                   (x1, y1) 是左上角坐标。
                                   (x2, y2) 是右下角坐标。
        bbox_gt (list or tuple):   真实的边界框 [x1, y1, x2, y2]。

    Returns:
        float: 计算得到的 IoU 值，范围在 [0, 1] 之间。
               如果任一框无效或并集面积为0，则返回 0。
    """
    # 确保输入格式正确
    if not (
        isinstance(bbox_pred, (list, tuple))
        and len(bbox_pred) == 4
        and isinstance(bbox_gt, (list, tuple))
        and len(bbox_gt) == 4
    ):
        raise ValueError("输入的边界框必须是包含4个数字的列表或元组 [x1, y1, x2, y2]")

    # 提取坐标
    pred_x1, pred_y1, pred_x2, pred_y2 = bbox_pred
    gt_x1, gt_y1, gt_x2, gt_y2 = bbox_gt

    # 检查坐标是否有效 (x1 < x2 and y1 < y2)
    if pred_x1 >= pred_x2 or pred_y1 >= pred_y2 or gt_x1 >= gt_x2 or gt_y1 >= gt_y2:
        print(f"警告: 无效的边界框坐标。 Pred: {bbox_pred}, GT: {bbox_gt}. 返回 IoU = 0。")
        return 0.0

    # 1. 计算交集区域的坐标
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)

    # 2. 计算交集区域的面积
    #    如果交集区域无效 (宽度或高度 <= 0)，则面积为 0
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    # 3. 计算每个边界框的面积
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

    # 4. 计算并集区域的面积
    union_area = pred_area + gt_area - intersection_area

    # 5. 计算 IoU
    #    处理并集面积为0的情况（例如两个框面积都为0）
    if union_area <= 0:
        # 如果交集也是0，那么可以认为IoU是0；如果交集不是0（理论上不可能在此情况下发生），
        # 或者两个框完全重合且面积为0，可以返回1或0，这里选择返回0避免除零错误。
        # 正常情况下，有效的框不会让 union_area 为 0。
        return 0.0
    else:
        iou = intersection_area / union_area
        return iou


def evaluate_grounding(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    评估模型定位能力，返回IoU均值和召回率
    pred_boxes: 预测框列表，每个为[x1, y1, x2, y2]
    gt_boxes:   GT框列表，每个为[x1, y1, x2, y2]
    """
    ious = []
    correct = 0
    for pred_box, gt_box in zip(pred_boxes, gt_boxes):
        iou = calculate_iou(pred_box, gt_box)
        ious.append(iou)
        if iou >= iou_threshold:
            correct += 1
    mean_iou = np.mean(ious)
    std_iou = np.std(ious)
    recall = correct / len(gt_boxes)
    return {"mean_iou": mean_iou.item(), "std_iou": std_iou.item(), "Accuracy@0.5": recall}


def extract_bbox(sample: dict, is_norm: bool = False):
    def _extract(result):
        numbers = re.findall(r"\d+", result)
        bbox = [int(num) for num in numbers][1:5]
        if len(bbox) != 4:
            bbox = bbox + [0] * (4 - len(bbox))
        if is_norm:
            ...
        return bbox

    answer = sample["answer"]
    output = sample["output"]

    return _extract(answer), _extract(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_file", type=str, default="datas/Optimus3/benchmark/grounding/20240422_grounding_gui_test_2197.json"
    )
    parser.add_argument("--is_norm", action="store_true")
    args = parser.parse_args()
    eval_file = args.eval_file

    with open(eval_file, "r") as fi:
        data = json.load(fi)

    pred_boxes = [extract_bbox(sample, args.is_norm)[1] for sample in data]
    gt_boxes = [extract_bbox(sample)[0] for sample in data]
    results = evaluate_grounding(pred_boxes, gt_boxes)

    print(results)

    with open(args.eval_file.replace(".json", "_result.json"), "w") as fi:
        json.dump(
            {
                "eval_file": eval_file,
                "mean_iou": results["mean_iou"],
                "std_iou": results["std_iou"],
                "Accuracy@0.5": results["Accuracy@0.5"],
            },
            fi,
            indent=4,
        )
