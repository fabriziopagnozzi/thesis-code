export RESULTS_DIR="${HOME}/thesis-code/src/experiments/medical_dataset_gen/_results"
export REPORTS_DIR="${HOME}/thesis-documents/reports"

gen_query_geometry_for_report() {
    cells=(
        balanced_reference__qwen_unbiased_simple
        dominance_extreme__qwen_unbiased_simple
        sparse_two_severe__qwen_unbiased_simple
        near_miss_h96__qwen_unbiased_simple
        background_far_16x2__qwen_unbiased_simple
    )

    for cell in "${cells[@]}"; do
    uv run task pipeline \
        --suite thesis_v5 \
        --cell "$cell" \
        --no-log-tee \
        --run "geom_plots --plots candidate_pool_umap,query_cosine_heatmap --query-ids q1 --output-dir /home/pagnozzi/thesis-documents/reports/query_geometry --umap-neighbors 20 --umap-min-dist 0.20"
    done
}

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
