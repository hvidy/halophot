#!/bin/bash
#SBATCH --job-name=halophot-cpu
#SBATCH --account=oz440
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --tmp=10G
#SBATCH --time=01:00:00
#SBATCH --array=0-99%25
#SBATCH --output=/aphid/scratch-3month/%u/logs/%x-%A_%a.out
#SBATCH --error=/aphid/scratch-3month/%u/logs/%x-%A_%a.err
#
# Swinburne Ngarrgu Tindebeek (tooarrana) EPYC Milan CPU array template for halophot v2.0.
# For 20-second TESS sectors, override at submission with: --cpus-per-task=8 --mem=24G.
# Set PIPELINE_MODE=full_e2e (after Stage-1 `mundey prefetch`) to run mundey -> tessbkgd -> halophot
# in a single pass on raw MAST _tp.fits files; defaults to PIPELINE_MODE=halophot_only.
# Submit after creating log directory: mkdir -p "$SCRATCH/logs" && sbatch hpc/halophot_survey_cpu.sbatch

set -eo pipefail  # omit -u: Lmod reads unset variables (PS1) inside `module load`

: "${MYFRED:=/fred/${OZ_PROJECT:-oz440}/$USER}"
: "${SCRATCH:=/aphid/scratch-3month/$USER}"
SIF="${SIF:-$MYFRED/containers/halo.sif}"
MANIFEST="${MANIFEST:-$MYFRED/manifests/tess_targets.txt}"
CATALOG_DIR="${CATALOG_DIR:-$SCRATCH/catalogs}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/.cache/tess_bright}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$MYFRED/outputs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}}"
PIPELINE_MODE="${PIPELINE_MODE:-halophot_only}"
export MUNDEY_SMEAR="${MUNDEY_SMEAR:-automatic}"

case "$PIPELINE_MODE" in
    halophot_only|full_e2e) ;;
    *)
        echo "ERROR: Invalid PIPELINE_MODE='$PIPELINE_MODE' (expected 'halophot_only' or 'full_e2e')." >&2
        exit 2
        ;;
esac

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/.apptainer/cache}"
export APPTAINER_TMPDIR="${JOBFS:-$SCRATCH/.apptainer/tmp}"
export APPTAINER_BIND="/fred,/aphid/scratch-3month${JOBFS:+,$JOBFS}"

mkdir -p "$ARCHIVE_DIR" "$SCRATCH/jax_cache" "$CACHE_DIR" \
    "$SCRATCH/.cache/lightkurve" "$SCRATCH/.cache/astropy" \
    "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Stage all per-task FITS/Parquet products onto node-local SSD ($JOBFS) or $SCRATCH,
# and bundle into a single .tar file on EXIT so /fred/oz440 consumes 1 inode per task.
STAGE_OUT="${JOBFS:-$SCRATCH/stage/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}/out"
mkdir -p "$STAGE_OUT"

archive_outputs() {
    if [ -d "$STAGE_OUT" ] && [ -n "$(ls -A "$STAGE_OUT" 2>/dev/null || true)" ]; then
        tar -cf "${ARCHIVE_DIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tar" \
            -C "$STAGE_OUT" .
    fi
}
trap archive_outputs EXIT

# Pin CPU thread pools strictly to $SLURM_CPUS_PER_TASK to prevent 64/128-core Milan oversubscription.
CPUS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export VECLIB_MAXIMUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${CPUS}"

# Pure CPU backend (omit --nv on Milan nodes) and zero-/fred cache paths.
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/jax_cache}"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XDG_CACHE_HOME="$SCRATCH/.cache"
export LIGHTKURVE_CACHE_DIR="$SCRATCH/.cache/lightkurve"
export ASTROPY_CACHE_DIR="$SCRATCH/.cache/astropy"
export MPLBACKEND=Agg

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
TPF_PATH="$(sed -n "${LINE_NUM}p" "$MANIFEST")"
TPF_PATH="${TPF_PATH%$'\r'}"
TPF_PATH="${TPF_PATH#"${TPF_PATH%%[![:space:]]*}"}"
TPF_PATH="${TPF_PATH%"${TPF_PATH##*[![:space:]]}"}"
if [[ -z "$TPF_PATH" || "$TPF_PATH" == \#* ]]; then
    echo "Skipping empty or comment manifest entry at line ${LINE_NUM}; exiting cleanly."
    exit 0
fi

TARGET_STEM="$(basename "$TPF_PATH")"
TARGET_STEM="${TARGET_STEM%.gz}"
TARGET_STEM="${TARGET_STEM%.fits}"
CAT_FILE="${CATALOG_DIR}/${TARGET_STEM}_cat.parquet"

module -q load apptainer

if [ "$PIPELINE_MODE" = "full_e2e" ]; then
    # Note: no --nv flag on CPU Milan compute nodes.
    apptainer exec "$SIF" python - "$TPF_PATH" "$TARGET_STEM" "$STAGE_OUT" "$CACHE_DIR" "$CAT_FILE" <<'PY'
import os
import sys
from pathlib import Path

import halophot
import mundey
import tessbkgd

tpf_path = Path(sys.argv[1])
stem = sys.argv[2]
out_dir = Path(sys.argv[3])
cache_dir = halophot.resolve_cache_dir(sys.argv[4], strict_hpc=True)
cat_file = Path(sys.argv[5])
has_cat = cat_file.is_file()
smear_mode = os.environ.get("MUNDEY_SMEAR", "automatic")

cal_tpf = mundey.recalibrate_tpf(
    tpf_path,
    cache_dir=cache_dir,
    smear=smear_mode,
    offline=True,
    strict_hpc=True,
)
sub_tpf = tessbkgd.subtract_background(cal_tpf)
del cal_tpf

lc = halophot.extract_halo_lightcurve(
    sub_tpf,
    sigclip=True,
    remove_contaminants=has_cat,
    catalog_table=cat_file if has_cat else None,
    offline=True,
    compute_uncertainty=True,
)
out_fits = out_dir / f"{stem}_halo_lc_tv.fits"
halophot.write_fits(lc, out_fits, overwrite=True)
print(f"Saved HALOCVER=1 full_e2e halo light curve to {out_fits}")
PY
else
    EXTRA_ARGS=()
    if [ -f "$CAT_FILE" ]; then
        EXTRA_ARGS+=(--deathstar --catalog-table "$CAT_FILE")
    fi

    # Note: no --nv flag on CPU Milan compute nodes.
    apptainer exec "$SIF" halo "$TPF_PATH" \
        --name "$TARGET_STEM" \
        --hlsp-fits \
        --uncertainty \
        --offline \
        --strict-hpc \
        --cache-dir "$CACHE_DIR" \
        --save-dir "$STAGE_OUT" \
        "${EXTRA_ARGS[@]}"
fi
