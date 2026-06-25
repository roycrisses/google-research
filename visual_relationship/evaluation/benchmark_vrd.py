
import time
import numpy as np
import dataclasses
from visual_relationship.evaluation import evaluate_vrd_lib

def benchmark():
    # Create dummy data
    n_gt = 1000
    n_pred = 1000

    gt_records = []
    for i in range(n_gt):
        box_a = evaluate_vrd_lib.Box('img', 'obj1', 0.1, 0.1, 0.2, 0.2)
        box_b = evaluate_vrd_lib.Box('img', 'obj2', 0.3, 0.3, 0.4, 0.4)
        gt_records.append(evaluate_vrd_lib.Record(box_a, box_b, 1, 1))

    pred_records = []
    for i in range(n_pred):
        # Boxes that do NOT overlap with ground truth to force full scan
        box_a = evaluate_vrd_lib.Box('img', 'obj1', 0.8, 0.8, 0.9, 0.9)
        box_b = evaluate_vrd_lib.Box('img', 'obj2', 0.8, 0.8, 0.9, 0.9)
        pred_records.append(evaluate_vrd_lib.Record(box_a, box_b, 1, 1))

    evaluator = evaluate_vrd_lib.VRDEvaluator.__new__(evaluate_vrd_lib.VRDEvaluator)
    evaluator.check_entity = False
    evaluator.iou_threshold = 0.5
    evaluator.NUM_LABELS = 4
    evaluator.TP_IDX = 0
    evaluator.FP_IDX = 1
    evaluator.FN_IDX = 2

    start = time.time()
    for _ in range(5):
        _ = evaluator.evaluate_example(pred_records, gt_records)
    end = time.time()

    print(f"Average time for evaluate_example (no matches): {(end - start) / 5:.4f}s")

    # Mix of matches and no matches
    pred_records = []
    for i in range(n_pred):
        if i % 2 == 0:
            # Match
            box_a = evaluate_vrd_lib.Box('img', 'obj1', 0.1, 0.1, 0.2, 0.2)
            box_b = evaluate_vrd_lib.Box('img', 'obj2', 0.3, 0.3, 0.4, 0.4)
        else:
            # No match
            box_a = evaluate_vrd_lib.Box('img', 'obj1', 0.8, 0.8, 0.9, 0.9)
            box_b = evaluate_vrd_lib.Box('img', 'obj2', 0.8, 0.8, 0.9, 0.9)
        pred_records.append(evaluate_vrd_lib.Record(box_a, box_b, 1, 1))

    start = time.time()
    for _ in range(5):
        _ = evaluator.evaluate_example(pred_records, gt_records)
    end = time.time()
    print(f"Average time for evaluate_example (mixed): {(end - start) / 5:.4f}s")

if __name__ == "__main__":
    benchmark()
