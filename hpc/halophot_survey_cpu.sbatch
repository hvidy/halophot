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
# Submit after creating log directory: mkdir -p "$SCRATCH/logs" && sbatch hpc/halophot_survey_cpu.sbatch

set -euo pipefail

: "${MYFRED:=/fred/oz440/$USER}"
: "${SCRATCH:=/aphid/scratch-3month/$USER}"
SIF="${SIF:-$MYFRED/containers/virgil.sif}"
MANIFEST="${MANIFEST:-$MYFRED/manifests/tess_targets.txt}"
CATALOG_DIR="${CATALOG_DIR:-$SCRATCH/catalogs}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$MYFRED/outputs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}}"

mkdir -p "$ARCHIVE_DIR" "$SCRATCH/jax_cache" "$SCRATCH/.cache/tess_bright" \
    "$SCRATCH/.cache/lightkurve" "$SCRATCH/.cache/astropy"

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

# Pure CPU backend (omit --nv on Milan nodes) and zero-/fred cache paths.
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/jax_cache}"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XDG_CACHE_HOME="$SCRATCH/.cache"
export LIGHTKURVE_CACHE_DIR="$SCRATCH/.cache/lightkurve"
export ASTROPY_CACHE_DIR="$SCRATCH/.cache/astropy"
export MPLBACKEND=Agg
export APPTAINER_BINDPATH="/fred,/aphid${JOBFS:+,$JOBFS}"

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
TPF_PATH="$(sed -n "${LINE_NUM}p" "$MANIFEST")"
if [ -z "$TPF_PATH" ]; then
    echo "No manifest entry at line ${LINE_NUM}; exiting cleanly."
    exit 0
fi

TARGET_STEM="$(basename "$TPF_PATH" .fits)"
TARGET_STEM="${TARGET_STEM%.fits.gz}"
CAT_FILE="${CATALOG_DIR}/${TARGET_STEM}_cat.parquet"

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
    --cache-dir "$SCRATCH/.cache/tess_bright" \
    --save-dir "$STAGE_OUT" \
    "${EXTRA_ARGS[@]}"
