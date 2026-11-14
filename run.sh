#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${ISAAC_SIM_PATH:-}" ]]; then
	isaac_sim_path="$ISAAC_SIM_PATH"
else
	isaac_sim_path=""
	for candidate in "$HOME/isaacsim" "$HOME"/.local/share/ov/pkg/isaac-sim*; do
		if [[ -x "$candidate/python.sh" ]]; then
			isaac_sim_path="$candidate"
			break
		fi
	done
fi

if [[ -z "$isaac_sim_path" || ! -x "$isaac_sim_path/python.sh" ]]; then
	printf 'Isaac Sim python.sh not found. Set ISAAC_SIM_PATH to the Isaac Sim install directory.\n' >&2
	exit 1
fi

exec "${isaac_sim_path}/python.sh" "${project_dir}/main.py" "$@"