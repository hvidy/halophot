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
# Swinburne Ngarrgu Tindebeek (tooarrana) EPYC Milan CPU array template for halophot v2.0.1.
# For 20-second TESS sectors, override at submission with: --cpus-per-task=8 --mem=24G.
# For manifests > 100 lines, override --array=0-$(( $(wc -l < "$MANIFEST") - 1 ))%25 at submission.
# Set PIPELINE_MODE=full_e2e ONLY after running Stage-1 `mundey prefetch` so concurrent array tasks
# share prebuilt sector -blf.fits caches in $CACHE_DIR instead of rebuilding them redundantly;
# defaults to PIPELINE_MODE=halophot_only.
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
# and bundle atomically via .part -> mv into a single .tar file on EXIT so /fred/oz440
# consumes 1 inode per task and never leaves a truncated .tar on SIGKILL.
STAGE_BASE="${JOBFS:-$SCRATCH/stage/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
STAGE_OUT="${STAGE_BASE}/out"
mkdir -p "$STAGE_OUT"

archive_outputs() {
    local tar_name="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tar"
    local tar_part="${ARCHIVE_DIR}/.${tar_name}.part.$$"
    if [ -d "$STAGE_OUT" ] && [ -n "$(ls -A "$STAGE_OUT" 2>/dev/null || true)" ]; then
        if tar -cf "$tar_part" -C "$STAGE_OUT" .; then
            mv -f "$tar_part" "${ARCHIVE_DIR}/${tar_name}"
        else
            rm -f "$tar_part"
        fi
    fi
    if [ -z "${JOBFS:-}" ] && [ -n "${STAGE_BASE:-}" ]; then
        rm -rf "$STAGE_BASE"
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

if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ] && [ -n "${SLURM_ARRAY_TASK_MAX:-}" ]; then
    MANIFEST_LINES="$(wc -l < "$MANIFEST")"
    if [ "$MANIFEST_LINES" -gt $((SLURM_ARRAY_TASK_MAX + 1)) ]; then
        echo "WARNING: Manifest $MANIFEST has $MANIFEST_LINES lines, exceeding --array max ($((SLURM_ARRAY_TASK_MAX + 1))); resubmit with --array=0-$((MANIFEST_LINES - 1))%25 to cover all targets." >&2
    fi
fi

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
if [ ! -f "$CAT_FILE" ] && [[ "$TARGET_STEM" == *-sub ]]; then
    ALT_CAT_FILE="${CATALOG_DIR}/${TARGET_STEM%-sub}_cat.parquet"
    if [ -f "$ALT_CAT_FILE" ]; then
        CAT_FILE="$ALT_CAT_FILE"
    fi
fi

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
    cache_dir=cache_dir,
    strict_hpc=True,
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
