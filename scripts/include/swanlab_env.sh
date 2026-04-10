#!/usr/bin/env bash
# Intended to be USED WITH `source`, not executed directly.
# Example:
#   source "${ROOT}/scripts/include/swanlab_env.sh"
#
# Set SwanLab env vars in one place for Linux/GPU runs without a .env file.

: "${SWANLAB_PROJECT:=minilm}"
: "${SWANLAB_MODE:=cloud}"
: "${SWANLAB_API_KEY:=L5t4Y3J6W3njoLWOjuKIu}"

export SWANLAB_PROJECT SWANLAB_MODE SWANLAB_API_KEY

