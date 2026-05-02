#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Full preflight + phase-1/phase-2 wrapper for the strict symbolic-input
# oracle-dynamics path. Use this when corpus/vocab/UMA readiness should be
# checked before training. Use train_full_selected_250m_oracle_dynamics_direct.sh
# when those artifacts already exist and you want to jump directly to training.
export ENABLE_TROPICAL_ATTENTION="${ENABLE_TROPICAL_ATTENTION:-1}"
export ENABLE_UMA_COORDINATE_HEAD="${ENABLE_UMA_COORDINATE_HEAD:-1}"
export ENABLE_UMA_INTERNAL_COORDINATES="${ENABLE_UMA_INTERNAL_COORDINATES:-1}"
export FULL_TRAIN_BATCH_SIZE="${FULL_TRAIN_BATCH_SIZE:-1}"
export FULL_TRAIN_EVAL_BATCH_SIZE="${FULL_TRAIN_EVAL_BATCH_SIZE:-1}"
export FULL_TRAIN_GRAD_ACCUM="${FULL_TRAIN_GRAD_ACCUM:-36}"
export FULL_TRAIN_SKIP_POLICY_CHECK="${FULL_TRAIN_SKIP_POLICY_CHECK:-0}"

UMA_CONTACT_CONFIG="${UMA_CONTACT_CONFIG:-config/train/overrides/uma_contact_geometry_loss.yaml}"
UMA_INTERNAL_CONFIG="${UMA_INTERNAL_CONFIG:-config/train/overrides/uma_internal_coordinates.yaml}"
ENABLE_UMA_CONTACT_GEOMETRY="${ENABLE_UMA_CONTACT_GEOMETRY:-1}"
if [[ "$ENABLE_UMA_INTERNAL_COORDINATES" == "1" ]]; then
  if [[ ! -f "$UMA_INTERNAL_CONFIG" ]]; then
    printf 'UMA internal-coordinate config not found: %s\n' "$UMA_INTERNAL_CONFIG" >&2
    exit 1
  fi
  case " ${EXTRA_TRAIN_CONFIGS:-} " in
    *" $UMA_INTERNAL_CONFIG "*) ;;
    *) export EXTRA_TRAIN_CONFIGS="${EXTRA_TRAIN_CONFIGS:+$EXTRA_TRAIN_CONFIGS }$UMA_INTERNAL_CONFIG" ;;
  esac
fi

if [[ "$ENABLE_UMA_CONTACT_GEOMETRY" == "1" ]]; then
  if [[ ! -f "$UMA_CONTACT_CONFIG" ]]; then
    printf 'UMA contact geometry config not found: %s\n' "$UMA_CONTACT_CONFIG" >&2
    exit 1
  fi
  case " ${EXTRA_TRAIN_CONFIGS:-} " in
    *" $UMA_CONTACT_CONFIG "*) ;;
    *) export EXTRA_TRAIN_CONFIGS="${EXTRA_TRAIN_CONFIGS:+$EXTRA_TRAIN_CONFIGS }$UMA_CONTACT_CONFIG" ;;
  esac
  export WANDB_TAGS="${WANDB_TAGS:-full-run,phase1,phase2,tokengt,4090,250m,hybrid-flash-mhta,mhta,uma-coordinate-head,uma-internal-coordinates,uma-force-dynamics,uma-contact-geometry,bioselfies}"
else
  export WANDB_TAGS="${WANDB_TAGS:-full-run,phase1,phase2,tokengt,4090,250m,hybrid-flash-mhta,mhta,uma-coordinate-head,uma-internal-coordinates,uma-force-dynamics,bioselfies}"
fi

exec "$ROOT/scripts/run_full_phase1_phase2_training_250m.sh" "$@"
