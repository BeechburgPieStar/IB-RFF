#!/bin/bash
# ============================================================
# Two-stage parallel training script
#   Stage 1: pretrain=0, alpha=0.0  (baseline weights, shared by CR/CRD)
#   Stage 2: pretrain=1, best alpha per (dataset, round)
# Each stage runs the 4 rounds in parallel on GPUs 0,1,2,3.
#
# NOTE: patch_size differs per dataset:
#   ManyRx  -> 16
#   ManySig -> 32
# ============================================================
set -u  # error on undefined vars; do NOT use -e (we want all 4 GPUs to finish even if one fails)

# ---------- paths ----------
LOG_DIR="logs"
mkdir -p "${LOG_DIR}/stage1" "${LOG_DIR}/stage2" weights

# ---------- common hyper-params ----------
SEED=2023
EPOCHS=200
BATCH_SIZE=128
LR=0.001
TRAIN_DATE="1 2"      # passed as nargs='+'

# ---------- per-dataset patch_size ----------
patch_size_for() {
    case "$1" in
        ManyRx)  echo 16 ;;
        ManySig) echo 32 ;;
        *) echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
}

# ---------- helper: launch one job on a given GPU ----------
# args: GPU dataset exp pretrain alpha round logfile
run_one() {
    local gpu=$1
    local dataset=$2
    local exp=$3
    local pretrain=$4
    local alpha=$5
    local round=$6
    local logfile=$7

    local patch
    patch=$(patch_size_for "${dataset}")

    CUDA_VISIBLE_DEVICES=${gpu} python main.py \
        --gpu 0 \
        --dataset_name "${dataset}" \
        --exp "${exp}" \
        --patch_size "${patch}" \
        --train_date ${TRAIN_DATE} \
        --all_test_round 4 \
        --test_round "${round}" \
        --pre_train "${pretrain}" \
        --alpha "${alpha}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --seed "${SEED}" \
        --code_state train_test \
        > "${logfile}" 2>&1
}

# ============================================================
# Stage 1: pretrain=0, alpha=0.0
# CR/CRD share weights (same filename), so we only run exp=CR.
# 2 datasets x 4 rounds = 8 jobs, in 2 batches of 4.
# ============================================================
echo "=============================================="
echo " Stage 1: pretrain=0, alpha=0.0  (baseline)"
echo "=============================================="

for DATASET in ManySig ManyRx; do
    PATCH=$(patch_size_for "${DATASET}")
    echo "[Stage 1] Dataset: ${DATASET} (patch_size=${PATCH}) -- launching 4 rounds in parallel"
    for R in 0 1 2 3; do
        LOG="${LOG_DIR}/stage1/${DATASET}_pre0_a0.0_round${R}.log"
        run_one "${R}" "${DATASET}" "CR" 0 0.0 "${R}" "${LOG}" &
    done
    wait
    echo "[Stage 1] Dataset: ${DATASET}  -- done"
done

echo ""
echo "Stage 1 finished. Logs in ${LOG_DIR}/stage1/"
echo ""

# ============================================================
# Stage 2: pretrain=1, best alpha per (dataset, exp, round)
# Layout of BEST_ALPHA arrays: index = round (0..3)
# ============================================================
echo "=============================================="
echo " Stage 2: pretrain=1, best alpha per round"
echo "=============================================="

# rows ordered by round 0,1,2,3
ALPHA_MANYSIG_CR=(0.1   0.004 0.08  0.1)
ALPHA_MANYSIG_CRD=(0.04 0.001 0.08  0.1)
ALPHA_MANYRX_CR=(0.08  0.008 0.004 0.01)
ALPHA_MANYRX_CRD=(0.01 0.008 0.08  0.004)

run_stage2_group() {
    local dataset=$1
    local exp=$2
    local -n alphas=$3   # nameref to the alpha array

    local patch
    patch=$(patch_size_for "${dataset}")

    echo "[Stage 2] ${dataset} / ${exp} (patch_size=${patch}) -- launching 4 rounds in parallel"
    for R in 0 1 2 3; do
        local A=${alphas[$R]}
        local LOG="${LOG_DIR}/stage2/${dataset}_${exp}_pre1_a${A}_round${R}.log"
        run_one "${R}" "${dataset}" "${exp}" 1 "${A}" "${R}" "${LOG}" &
    done
    wait
    echo "[Stage 2] ${dataset} / ${exp}  -- done"
}

run_stage2_group ManySig CR  ALPHA_MANYSIG_CR
run_stage2_group ManySig CRD ALPHA_MANYSIG_CRD
run_stage2_group ManyRx  CR  ALPHA_MANYRX_CR
run_stage2_group ManyRx  CRD ALPHA_MANYRX_CRD

echo ""
echo "=============================================="
echo " All done. Logs:"
echo "   Stage 1: ${LOG_DIR}/stage1/"
echo "   Stage 2: ${LOG_DIR}/stage2/"
echo "=============================================="