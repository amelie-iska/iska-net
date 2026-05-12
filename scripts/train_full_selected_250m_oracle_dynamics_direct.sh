#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Direct-to-training wrapper for the strict symbolic-input oracle-dynamics path.
# It assumes the full selected corpus, 250M vocab, context config, FairChem repo,
# and UMA weights have already been prepared by the full wrapper/readiness path.
export ENABLE_TROPICAL_ATTENTION="${ENABLE_TROPICAL_ATTENTION:-1}"
export ENABLE_UMA_COORDINATE_HEAD="${ENABLE_UMA_COORDINATE_HEAD:-1}"
export ENABLE_UMA_INTERNAL_COORDINATES="${ENABLE_UMA_INTERNAL_COORDINATES:-1}"
export SKIP_POLICY_CHECK="${SKIP_POLICY_CHECK:-1}"

# MHTA + coordinate-force + contact geometry is the heaviest supported 250M
# local path. Default to micro-batch 1 and recover effective batch with grad
# accumulation; callers can override these after checking local memory.
export FULL_TRAIN_BATCH_SIZE="${FULL_TRAIN_BATCH_SIZE:-1}"
export FULL_TRAIN_EVAL_BATCH_SIZE="${FULL_TRAIN_EVAL_BATCH_SIZE:-1}"
export FULL_TRAIN_GRAD_ACCUM="${FULL_TRAIN_GRAD_ACCUM:-36}"

UMA_CONTACT_CONFIG="${UMA_CONTACT_CONFIG:-config/train/overrides/uma_contact_geometry_loss.yaml}"
UMA_INTERNAL_CONFIG="${UMA_INTERNAL_CONFIG:-config/train/overrides/uma_internal_coordinates.yaml}"
UMA_ALL_ATOM_CONFIG="${UMA_ALL_ATOM_CONFIG:-config/train/overrides/uma_all_atom_cartesian_head_8192.yaml}"
ENABLE_UMA_CONTACT_GEOMETRY="${ENABLE_UMA_CONTACT_GEOMETRY:-1}"
ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD="${ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD:-0}"
if [[ "$ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD" == "1" ]]; then
  if [[ ! -f "$UMA_ALL_ATOM_CONFIG" ]]; then
    printf 'UMA all-atom Cartesian config not found: %s\n' "$UMA_ALL_ATOM_CONFIG" >&2
    exit 1
  fi
  export UMA_COORDINATE_HEAD_CONFIG="$UMA_ALL_ATOM_CONFIG"
  export ENABLE_UMA_COORDINATE_HEAD=1
fi
if [[ "$ENABLE_UMA_INTERNAL_COORDINATES" == "1" && "$ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD" != "1" ]]; then
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
  export WANDB_TAGS="${WANDB_TAGS:-direct,phase1,tokengt,4090,250m,hybrid-flash-mhta,mhta,uma-coordinate-head,uma-internal-coordinates,uma-force-dynamics,uma-contact-geometry,bioselfies}"
else
  export WANDB_TAGS="${WANDB_TAGS:-direct,phase1,tokengt,4090,250m,hybrid-flash-mhta,mhta,uma-coordinate-head,uma-internal-coordinates,uma-force-dynamics,bioselfies}"
fi

exec "$ROOT/scripts/train_full_selected_250m_direct.sh" "$@"
