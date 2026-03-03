#!/bin/bash
set -euo pipefail

# One script dedicated to generation only (no metrics logging).
# Use MODEL to override the shared model path; OUTPUT_ROOT to redirect outputs.

PLAN_GPU=${PLAN_GPU:-0}
GROUNDING_GPU=${GROUNDING_GPU:-1}
REFLECTION_GPU=${REFLECTION_GPU:-2}
CAPTION_GPU=${CAPTION_GPU:-3}
VQA_GPU=${VQA_GPU:-4}

MODEL=${MODEL:-model dir}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/eval/20260302}

PLAN_EVAL_FILE=${PLAN_EVAL_FILE:-${OUTPUT_ROOT}/plan.json}
GROUNDING_EVAL_FILE=${GROUNDING_EVAL_FILE:-${OUTPUT_ROOT}/grounding.json}
REFLECTION_EVAL_FILE=${REFLECTION_EVAL_FILE:-${OUTPUT_ROOT}/reflection.json}
CAPTION_EVAL_FILE=${CAPTION_EVAL_FILE:-${OUTPUT_ROOT}/caption.json}
VQA_EVAL_FILE=${VQA_EVAL_FILE:-${OUTPUT_ROOT}/vqa.json}

TMP_DIR=${TMP_DIR:-/tmp/optimus3_eval_configs}
mkdir -p "${TMP_DIR}"

make_config() {
    local src_yaml=$1
    local dest_yaml=$2
    local save_path=$3
    python - <<'PY' "$src_yaml" "$dest_yaml" "$MODEL" "$save_path"
import sys, yaml, pathlib
src, dst, model, save = sys.argv[1:5]
with open(src, 'r') as f:
    data = yaml.safe_load(f)
data['model_name_or_path'] = model
data['save_dir'] = save
pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, 'w') as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY
}

run_plan() {
    echo "===== [$(date '+%F %T')] GENERATE PLAN on GPU ${PLAN_GPU}"
    PLAN_CFG="${TMP_DIR}/plan.yaml"
    make_config configs/optimus3/eval/9B/coldstart/plan.yaml "${PLAN_CFG}" "${PLAN_EVAL_FILE}"
    CUDA_VISIBLE_DEVICES=${PLAN_GPU} \
    python -m minecraftoptimus.evaluation.evaluator "${PLAN_CFG}"
}

run_grounding() {
    echo "===== [$(date '+%F %T')] GENERATE GROUNDING on GPU ${GROUNDING_GPU}"
    GROUNDING_CFG="${TMP_DIR}/grounding.yaml"
    make_config configs/optimus3/eval/9B/coldstart/grounding.yaml "${GROUNDING_CFG}" "${GROUNDING_EVAL_FILE}"
    CUDA_VISIBLE_DEVICES=${GROUNDING_GPU} \
    python -m minecraftoptimus.evaluation.evaluator "${GROUNDING_CFG}"
}

run_reflection() {
    echo "===== [$(date '+%F %T')] GENERATE REFLECTION on GPU ${REFLECTION_GPU}"
    REFLECTION_CFG="${TMP_DIR}/reflection.yaml"
    make_config configs/optimus3/eval/9B/coldstart/reflection.yaml "${REFLECTION_CFG}" "${REFLECTION_EVAL_FILE}"
    CUDA_VISIBLE_DEVICES=${REFLECTION_GPU} \
    python -m minecraftoptimus.evaluation.evaluator "${REFLECTION_CFG}"
}

run_caption() {
    echo "===== [$(date '+%F %T')] GENERATE CAPTION on GPU ${CAPTION_GPU}"
    CAPTION_CFG="${TMP_DIR}/caption.yaml"
    make_config configs/optimus3/eval/9B/coldstart/caption.yaml "${CAPTION_CFG}" "${CAPTION_EVAL_FILE}"
    CUDA_VISIBLE_DEVICES=${CAPTION_GPU} \
    python -m minecraftoptimus.evaluation.evaluator "${CAPTION_CFG}"
}

run_vqa() {
    echo "===== [$(date '+%F %T')] GENERATE VQA on GPU ${VQA_GPU}"
    VQA_CFG="${TMP_DIR}/vqa.yaml"
    make_config configs/optimus3/eval/9B/coldstart/vqa.yaml "${VQA_CFG}" "${VQA_EVAL_FILE}"
    CUDA_VISIBLE_DEVICES=${VQA_GPU} \
    python -m minecraftoptimus.evaluation.evaluator "${VQA_CFG}"
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
