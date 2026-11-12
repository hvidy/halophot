#!/bin/bash
#SBATCH --job-name=halophot-gpu
#SBATCH --account=oz440
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=20G
#SBATCH --time=02:00:00
#SBATCH --output=/aphid/scratch-3month/%u/logs/%x-%j.out
#SBATCH --error=/aphid/scratch-3month/%u/logs/%x-%j.err
#
# Swinburne Ngarrgu Tindebeek (tooarrana) NVIDIA A100 80GB GPU survey template for halophot v2.0.
# Bucketed N_pad padding in HaloObservation reuses compiled FP64 XLA kernels across targets.
# Submit after creating log directory: mkdir -p "$SCRATCH/logs" && sbatch hpc/halophot_survey_gpu.sbatch

set -euo pipefail

: "${MYFRED:=/fred/oz440/$USER}"
: "${SCRATCH:=/aphid/scratch-3month/$USER}"
SIF="${SIF:-$MYFRED/containers/virgil.sif}"
MANIFEST="${MANIFEST:-$MYFRED/manifests/tess_20s_targets.txt}"
CATALOG_DIR="${CATALOG_DIR:-$SCRATCH/catalogs}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$MYFRED/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"

mkdir -p "$ARCHIVE_DIR" "$SCRATCH/jax_cache" "$SCRATCH/.cache/tess_bright" \
    "$SCRATCH/.cache/lightkurve" "$SCRATCH/.cache/astropy"

STAGE_OUT="${JOBFS:-$SCRATCH/stage/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}/out"
mkdir -p "$STAGE_OUT"

archive_outputs() {
    if [ -d "$STAGE_OUT" ] && [ -n "$(ls -A "$STAGE_OUT" 2>/dev/null || true)" ]; then
        tar -cf "${ARCHIVE_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.tar" -C "$STAGE_OUT" .
    fi
}
trap archive_outputs EXIT

# Pin host CPU thread pools alongside the A100 GPU context.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/jax_cache}"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XDG_CACHE_HOME="$SCRATCH/.cache"
export LIGHTKURVE_CACHE_DIR="$SCRATCH/.cache/lightkurve"
export ASTROPY_CACHE_DIR="$SCRATCH/.cache/astropy"
export MPLBACKEND=Agg
export APPTAINER_BINDPATH="/fred,/aphid${JOBFS:+,$JOBFS}"

# Run an entire target batch in one Python process on A100 (--nv) so XLA compilation is shared.
apptainer exec --nv "$SIF" python - "$MANIFEST" "$STAGE_OUT" "$CATALOG_DIR" <<'PY'
import sys
from pathlib import Path
from astropy.table import Table
import halophot

manifest_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
catalog_dir = Path(sys.argv[3])
halophot.resolve_cache_dir(strict_hpc=True)

rows = []
for raw_line in manifest_path.read_text().splitlines():
    tpf_str = raw_line.strip()
    if not tpf_str or tpf_str.startswith("#"):
        continue
    tpf_path = Path(tpf_str)
    stem = tpf_path.name.removesuffix(".gz").removesuffix(".fits")
    cat_file = catalog_dir / f"{stem}_cat.parquet"
    has_cat = cat_file.is_file()
    try:
        lc = halophot.extract_halo_lightcurve(
            tpf_path,
            sigclip=True,
            remove_contaminants=has_cat,
            catalog_table=cat_file if has_cat else None,
            offline=True,
            compute_uncertainty=True,
        )
        out_fits = out_dir / f"{stem}_halo_lc.fits"
        halophot.write_fits(lc, out_fits)
        rows.append(
            {
                "target": stem,
                "status": "ok",
                "n_cad": len(lc),
                "fits_file": out_fits.name,
                "error": "",
            }
        )
    except Exception as exc:
        print(f"ERROR [{stem}]: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        rows.append(
            {
                "target": stem,
                "status": "failed",
                "n_cad": 0,
                "fits_file": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

if rows:
    Table(rows=rows).write(out_dir / "summary.parquet", format="parquet", overwrite=True)
PY
