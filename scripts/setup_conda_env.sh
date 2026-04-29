#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-iska-ugm}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists; updating from environment.yml"
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo "Creating conda env ${ENV_NAME} from environment.yml"
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

echo "Environment ready. Use:"
echo "  conda activate ${ENV_NAME}"

