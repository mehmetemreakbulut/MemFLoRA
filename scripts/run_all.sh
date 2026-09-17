#!/usr/bin/env bash
# Run one paper experiment group, or all 21 experiments, in numerical order.
set -euo pipefail
cd "$(dirname "$0")/.."

group="${1:-all}"
case "$group" in
    main)        pattern='0[1-6]' ;;
    convergence) pattern='0[78]' ;;
    memory)      pattern='(09|1[0-2])' ;;
    ablations)   pattern='1[4-7]' ;;
    tinytl)      pattern='2[0-2]' ;;
    time_spec)   pattern='2[34]' ;;
    all)         pattern='[0-9][0-9]' ;;
    *) echo "use one of: main convergence memory ablations tinytl time_spec all" >&2; exit 2 ;;
esac

scripts=()
for script in scripts/[0-9][0-9]_*.sh; do
    [[ "$script" == scripts/00_* ]] && continue
    [[ "${script##*/}" =~ ^${pattern}_ ]] && scripts+=("$script")
done

echo "running ${#scripts[@]} experiment(s) in group '$group'"
failed=()
for script in "${scripts[@]}"; do
    echo ">>> $script"
    if bash "$script"; then
        echo "<<< OK $script"
    else
        failed+=("$script")
    fi
done
if ((${#failed[@]})); then
    printf 'FAILED: %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "all ${#scripts[@]} experiment(s) completed"
