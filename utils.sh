export RESULTS_DIR="$HOME/thesis/src/experiments/medical_dataset_gen/_results"

run() {
    uv run task pipeline "$@"
}

runall() {
  local find_args=""
  local pipeline_args=""
  local dry_run=0

  while (( $# > 0 )); do
    case "$1" in
      --find)
        [[ $# -ge 2 ]] || {
          print -u2 "runall: no value provided for --find"
          return 2
        }
        find_args=$2
        shift 2
        ;;

      --run)
        [[ $# -ge 2 ]] || {
          print -u2 "runall: no value provided for --run"
          return 2
        }
        pipeline_args=$2
        shift 2
        ;;

      --dry-run)
        dry_run=1
        shift
        ;;

      --)
        shift
        break
        ;;

      *)
        print -u2 "runall: unknown argument: $1"
        return 2
        ;;
    esac
  done

  eval "
    find -L ${(q)RESULTS_DIR} \
      -mindepth 2 -maxdepth 2 -type d \
      ! -path '*_reports*' \
      ! -path '*_embeddings*' \
      ! -path '*_shared*' \
      ${find_args} \
      -printf '%P\n'
  " |
    sort |
    while IFS= read -r EXP; do
      if (( dry_run )); then
        print -r -- "$EXP"
      else
        eval "run --exp ${(q)EXP} ${pipeline_args}"
      fi
    done
}
