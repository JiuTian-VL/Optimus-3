#!/bin/bash
set -euo pipefail

# This script only computes metrics from already generated outputs.
# Metrics are appended to LOG_FILE; generation output is not touched.

PLAN_GPU=${PLAN_GPU:-0}
GROUNDING_GPU=${GROUNDING_GPU:-1}
REFLECTION_GPU=${REFLECTION_GPU:-2}
CAPTION_GPU=${CAPTION_GPU:-3}
VQA_GPU=${VQA_GPU:-4}

OUTPUT_ROOT=${OUTPUT_ROOT:-output/eval/20260302}

PLAN_EVAL_FILE=${PLAN_EVAL_FILE:-${OUTPUT_ROOT}/plan.json}
GROUNDING_EVAL_FILE=${GROUNDING_EVAL_FILE:-${OUTPUT_ROOT}/grounding.json}
REFLECTION_EVAL_FILE=${REFLECTION_EVAL_FILE:-${OUTPUT_ROOT}/reflection.json}
CAPTION_EVAL_FILE=${CAPTION_EVAL_FILE:-${OUTPUT_ROOT}/caption.json}
VQA_EVAL_FILE=${VQA_EVAL_FILE:-${OUTPUT_ROOT}/vqa.json}

LOG_FILE=${LOG_FILE:-${OUTPUT_ROOT}/benchmark_eval.txt}
mkdir -p "$(dirname "$LOG_FILE")"

# single-model judge settings (for caption/vqa only)
JUDGE_LLM=${JUDGE_LLM:-gpt-4.1}
JUDGE_API_KEY=${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-https://api.openai.com/v1}
JUDGE_MAX_CONCURRENT=${JUDGE_MAX_CONCURRENT:-16}
CAPTION_JUDGE_FILE=${CAPTION_JUDGE_FILE:-output/eval/judge/$(basename "${OUTPUT_ROOT}")_caption_judge.json}
VQA_JUDGE_FILE=${VQA_JUDGE_FILE:-output/eval/judge/$(basename "${OUTPUT_ROOT}")_vqa_judge.json}

# log_metrics <gpu_id> <cmd...>
log_metrics() {
    local gpu=$1
    shift
    CUDA_VISIBLE_DEVICES=${gpu} "$@" 2>&1 | tee -a "$LOG_FILE"
}

log_metrics_silent() {
    local gpu=$1
    shift
    CUDA_VISIBLE_DEVICES=${gpu} "$@" >/dev/null 2>&1
}

log_line() {
    echo "$1" | tee -a "$LOG_FILE"
}

print_task_header() {
    local task_name=$1
    log_line ""
    log_line "###### [${task_name}] Evaluation Summary ######"
}

print_task_footer() {
    log_line "###############################################"
    log_line ""
}

print_plan_summary() {
    local eval_file=$1
    python - "$eval_file" <<'PY' | tee -a "$LOG_FILE"
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"Total Samples: 0")
    print("Metric(pass_count): 0/0")
    print("Metric(pass_ratio): 0.0000")
    print(f"Warning: eval file not found: {path}")
    raise SystemExit(0)

data = json.loads(path.read_text(encoding="utf-8"))

def extract_answer(content):
    if content is None:
        return None
    match = re.search(r"<answer>(.*?)</answer>", str(content), re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(content)

def extract_steps(content):
    steps = []
    if content is None:
        return steps
    pattern = r"step\s+(\d+):\s*(?:[^\d]*?)(\d+)\s+([^\n]+)"
    for step_num, count, item in re.findall(pattern, str(content), re.IGNORECASE):
        steps.append({"step": int(step_num), "count": int(count), "item": item.strip()})
    return steps

def is_same_steps(answer, pred):
    if len(pred) == 0:
        return False
    all_items = [item["item"] for item in answer]
    item_cnt = {item["item"]: item["count"] for item in answer}
    for item in pred:
        if item["item"] not in all_items:
            return False
        if item["count"] < item_cnt[item["item"]]:
            return False
    if not answer or not pred:
        return False
    return answer[-1]["item"] == pred[-1]["item"]

total = len(data) if isinstance(data, list) else 0
passed = 0
for sample in data if isinstance(data, list) else []:
    answer_steps = extract_steps(extract_answer(sample.get("answer", "")))
    pred_steps = extract_steps(extract_answer(sample.get("output", "")))
    if answer_steps and is_same_steps(answer_steps, pred_steps):
        passed += 1

ratio = (passed / total) if total else 0.0
print(f"Total Samples: {total}")
print(f"Metric(pass_count): {passed}/{total}")
print(f"Metric(pass_ratio): {ratio:.4f}")
PY
}

print_grounding_summary() {
    local eval_file=$1
    python - "$eval_file" <<'PY' | tee -a "$LOG_FILE"
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("Total Samples: 0")
    print("Metric(mean_iou): 0.0000")
    print("Metric(std_iou): 0.0000")
    print("Metric(Accuracy@0.5): 0.0000")
    print(f"Warning: eval file not found: {path}")
    raise SystemExit(0)

data = json.loads(path.read_text(encoding="utf-8"))

def extract_bbox(text):
    numbers = re.findall(r"\d+", str(text))
    bbox = [int(num) for num in numbers][1:5]
    if len(bbox) != 4:
        bbox = bbox + [0] * (4 - len(bbox))
    return bbox

def iou(pred, gt):
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    if px1 >= px2 or py1 >= py2 or gx1 >= gx2 or gy1 >= gy2:
        return 0.0
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    pa = (px2 - px1) * (py2 - py1)
    ga = (gx2 - gx1) * (gy2 - gy1)
    union = pa + ga - inter
    if union <= 0:
        return 0.0
    return inter / union

ious = []
acc_count = 0
for sample in data if isinstance(data, list) else []:
    gt = extract_bbox(sample.get("answer", ""))
    pred = extract_bbox(sample.get("output", ""))
    val = iou(pred, gt)
    ious.append(val)
    if val >= 0.5:
        acc_count += 1

total = len(ious)
mean = (sum(ious) / total) if total else 0.0
std = (sum((x - mean) ** 2 for x in ious) / total) ** 0.5 if total else 0.0
acc = (acc_count / total) if total else 0.0

print(f"Total Samples: {total}")
print(f"Metric(mean_iou): {mean:.4f}")
print(f"Metric(std_iou): {std:.4f}")
print(f"Metric(Accuracy@0.5): {acc:.4f}")
PY
}

print_reflection_summary() {
    local eval_file=$1
    python - "$eval_file" <<'PY' | tee -a "$LOG_FILE"
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("Total Samples: 0")
    print("Metric(correct_count): 0/0")
    print("Metric(accuracy): 0.0000")
    print(f"Warning: eval file not found: {path}")
    raise SystemExit(0)

data = json.loads(path.read_text(encoding="utf-8"))

def extract_answer(pred):
    if not isinstance(pred, str):
        return ""
    ans = re.search(r"<answer>\s*(.*?)\s*</answer>", pred, flags=re.S | re.I)
    if ans:
        return ans.group(1).strip()
    no_think = re.sub(r"<think>.*?</think>", "", pred, flags=re.S | re.I)
    return no_think.strip()

def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()

total = 0
correct = 0
for item in data if isinstance(data, list) else []:
    gt = normalize(item.get("answer", ""))
    pred = normalize(extract_answer(item.get("output", "")))
    if gt == pred:
        correct += 1
    total += 1

acc = (correct / total) if total else 0.0
print(f"Total Samples: {total}")
print(f"Metric(correct_count): {correct}/{total}")
print(f"Metric(accuracy): {acc:.4f}")
PY
}

print_judge_summary() {
    local eval_file=$1
    local judge_file=$2
    python - "$eval_file" "$judge_file" <<'PY' | tee -a "$LOG_FILE"
import json
import sys
from pathlib import Path

eval_path = Path(sys.argv[1])
judge_path = Path(sys.argv[2])
stats_path = Path(str(judge_path).replace(".json", "_stats.json"))

if not eval_path.exists() or not judge_path.exists():
    print("Total Samples(eval): 0")
    print("Total Samples(judged): 0")
    print("Metric(avg_score): 0.0000")
    print("Metric(total_score): 0")
    print(f"Warning: missing eval/judge file. eval={eval_path.exists()} judge={judge_path.exists()}")
    raise SystemExit(0)

eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
judge_data = json.loads(judge_path.read_text(encoding="utf-8"))
eval_total = len(eval_data) if isinstance(eval_data, list) else 0
judged_total = len(judge_data) if isinstance(judge_data, list) else 0

avg_score = 0.0
score_hist = {str(s): 0 for s in range(10, -1, -1)}
if stats_path.exists():
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    avg_score = float(stats.get("avg_score", 0.0))
    saved_hist = stats.get("score_hist", {})
    for s in range(10, -1, -1):
        score_hist[str(s)] = int(saved_hist.get(str(s), 0))
else:
    score_sum = 0
    for item in judge_data if isinstance(judge_data, list) else []:
        score = int(item.get("score", 0))
        score = max(0, min(10, score))
        score_hist[str(score)] += 1
        score_sum += score
    avg_score = (score_sum / judged_total) if judged_total else 0.0

total_score = sum(int(k) * int(v) for k, v in score_hist.items())
print(f"Total Samples(eval): {eval_total}")
print(f"Total Samples(judged): {judged_total}")
print(f"Metric(avg_score): {avg_score:.4f}")
print(f"Metric(total_score): {total_score}")
print("Segmented Score Stats:")
for s in range(10, -1, -1):
    cnt = score_hist[str(s)]
    pct = (cnt / judged_total * 100.0) if judged_total else 0.0
    print(f"Segment(score={s}): count={cnt}, ratio={pct:.2f}%")
PY
}

run_plan() {
    echo "===== [$(date '+%F %T')] METRICS PLAN on GPU ${PLAN_GPU}"
    log_metrics_silent "${PLAN_GPU}" \
      python -m minecraftoptimus.evaluation.optimus3.plan \
        --eval_file "${PLAN_EVAL_FILE}"
    print_task_header "PLAN"
    print_plan_summary "${PLAN_EVAL_FILE}"
    print_task_footer
}

run_grounding() {
    echo "===== [$(date '+%F %T')] METRICS GROUNDING on GPU ${GROUNDING_GPU}"
    log_metrics_silent "${GROUNDING_GPU}" \
      python -m minecraftoptimus.evaluation.optimus3.grounding \
        --eval_file "${GROUNDING_EVAL_FILE}"
    print_task_header "GROUNDING"
    print_grounding_summary "${GROUNDING_EVAL_FILE}"
    print_task_footer
}

run_reflection() {
    echo "===== [$(date '+%F %T')] METRICS REFLECTION on GPU ${REFLECTION_GPU}"
    log_metrics_silent "${REFLECTION_GPU}" \
      python -m minecraftoptimus.evaluation.optimus3.reflection \
        --eval_file "${REFLECTION_EVAL_FILE}"
    print_task_header "REFLECTION"
    print_reflection_summary "${REFLECTION_EVAL_FILE}"
    print_task_footer
}

run_caption() {
    echo "===== [$(date '+%F %T')] METRICS CAPTION on GPU ${CAPTION_GPU}"
    log_metrics "${CAPTION_GPU}" \
      python src/minecraftoptimus/evaluation/optimus3/single_model_eval_async.py \
        --eval_file "${CAPTION_EVAL_FILE}" \
        --judge_llm "${JUDGE_LLM}" \
        --api_key "${JUDGE_API_KEY}" \
        --base_url "${JUDGE_BASE_URL}" \
        --max_concurrent "${JUDGE_MAX_CONCURRENT}" \
        --output_file "${CAPTION_JUDGE_FILE}"
    print_task_header "CAPTION"
    print_judge_summary "${CAPTION_EVAL_FILE}" "${CAPTION_JUDGE_FILE}"
    print_task_footer
}

run_vqa() {
    echo "===== [$(date '+%F %T')] METRICS VQA on GPU ${VQA_GPU}"
    log_metrics "${VQA_GPU}" \
      python src/minecraftoptimus/evaluation/optimus3/single_model_eval_async.py \
        --eval_file "${VQA_EVAL_FILE}" \
        --judge_llm "${JUDGE_LLM}" \
        --api_key "${JUDGE_API_KEY}" \
        --base_url "${JUDGE_BASE_URL}" \
        --max_concurrent "${JUDGE_MAX_CONCURRENT}" \
        --output_file "${VQA_JUDGE_FILE}"
    print_task_header "VQA"
    print_judge_summary "${VQA_EVAL_FILE}" "${VQA_JUDGE_FILE}"
    print_task_footer
}

case "${1:-all}" in
    plan) run_plan ;;
    grounding) run_grounding ;;
    reflection) run_reflection ;;
    caption) run_caption ;;
    vqa) run_vqa ;;
    all)
        run_plan &
        run_grounding &
        run_reflection &
        run_caption &
        run_vqa &
        wait
        ;;
    *)
        echo "Usage: $0 [plan|grounding|reflection|caption|vqa|all]" >&2
        exit 1
        ;;
esac
