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
# Swinburne Ngarrgu Tindebeek (tooarrana) NVIDIA A100 80GB GPU survey template for halophot v2.0.1.
# Bucketed N_pad padding in HaloObservation reuses compiled FP64 XLA kernels across sigclip passes
# and jackknife folds within each target (and across targets sharing active pixel count M and N_pad).
# Set PIPELINE_MODE=full_e2e (after Stage-1 `mundey prefetch`) to run mundey -> tessbkgd -> halophot
# in a single pass on raw MAST _tp.fits files; defaults to PIPELINE_MODE=halophot_only.
# Set MAX_FAIL_FRAC (default 0.0) to control the maximum tolerated target failure fraction.
# Submit after creating log directory: mkdir -p "$SCRATCH/logs" && sbatch hpc/halophot_survey_gpu.sbatch

set -eo pipefail  # omit -u: Lmod reads unset variables (PS1) inside `module load`

: "${MYFRED:=/fred/${OZ_PROJECT:-oz440}/$USER}"
: "${SCRATCH:=/aphid/scratch-3month/$USER}"
SIF="${SIF:-$MYFRED/containers/halo.sif}"
MANIFEST="${MANIFEST:-$MYFRED/manifests/tess_20s_targets.txt}"
CATALOG_DIR="${CATALOG_DIR:-$SCRATCH/catalogs}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/.cache/tess_bright}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$MYFRED/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"
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

STAGE_BASE="${JOBFS:-$SCRATCH/stage/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"
STAGE_OUT="${STAGE_BASE}/out"
mkdir -p "$STAGE_OUT"

archive_outputs() {
    local tar_name="${SLURM_JOB_NAME}-${SLURM_JOB_ID}.tar"
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

# Pin host CPU thread pools alongside the A100 GPU context.
CPUS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export VECLIB_MAXIMUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/jax_cache}"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XDG_CACHE_HOME="$SCRATCH/.cache"
export LIGHTKURVE_CACHE_DIR="$SCRATCH/.cache/lightkurve"
export ASTROPY_CACHE_DIR="$SCRATCH/.cache/astropy"
export MPLBACKEND=Agg

module -q load apptainer

# Run an entire target batch in one Python process on A100 (--nv) so JAX startup and intra-target
# (sigclip / jackknife) XLA compilations are amortized across the batch.
apptainer exec --nv "$SIF" python - "$MANIFEST" "$STAGE_OUT" "$CATALOG_DIR" "$CACHE_DIR" "$PIPELINE_MODE" <<'PY'
from collections import Counter
import os
import sys
from pathlib import Path

from astropy.io import fits
from astropy.table import Table
import halophot

manifest_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
catalog_dir = Path(sys.argv[3])
cache_dir = halophot.resolve_cache_dir(sys.argv[4], strict_hpc=True)
pipeline_mode = sys.argv[5]

if pipeline_mode == "full_e2e":
    import mundey
    import tessbkgd
elif pipeline_mode != "halophot_only":
    raise ValueError(
        f"Unsupported PIPELINE_MODE={pipeline_mode!r}; expected 'halophot_only' or 'full_e2e'."
    )

smear_mode = os.environ.get("MUNDEY_SMEAR", "automatic")
max_fail_frac = float(os.environ.get("MAX_FAIL_FRAC", "0.0"))

manifest_entries = [
    Path(raw_line.strip())
    for raw_line in manifest_path.read_text().splitlines()
    if raw_line.strip() and not raw_line.strip().startswith("#")
]
raw_stems = [p.name.removesuffix(".gz").removesuffix(".fits") for p in manifest_entries]
stem_counts = Counter(raw_stems)
used_out_stems: set[str] = set()

rows = []
for entry_idx, (tpf_path, stem) in enumerate(zip(manifest_entries, raw_stems, strict=True)):
    cat_file = catalog_dir / f"{stem}_cat.parquet"
    if not cat_file.is_file() and stem.endswith("-sub"):
        alt_cat = catalog_dir / f"{stem.removesuffix('-sub')}_cat.parquet"
        if alt_cat.is_file():
            cat_file = alt_cat
    has_cat = cat_file.is_file()
    tpf_input = None
    lc = None
    out_stem = stem
    try:
        if stem_counts[stem] > 1:
            sector_val = None
            if tpf_path.is_file():
                with fits.open(tpf_path) as h_sec:
                    sector_val = h_sec[0].header.get("SECTOR")
            if isinstance(sector_val, int) and not isinstance(sector_val, bool):
                out_stem = f"{stem}_s{sector_val:04d}"
            if out_stem == stem or out_stem in used_out_stems:
                out_stem = f"{stem}_{entry_idx:04d}"
        elif out_stem in used_out_stems:
            out_stem = f"{stem}_{entry_idx:04d}"

        if pipeline_mode == "full_e2e":
            cal_tpf = mundey.recalibrate_tpf(
                tpf_path,
                cache_dir=cache_dir,
                smear=smear_mode,
                offline=True,
                strict_hpc=True,
            )
            tpf_input = tessbkgd.subtract_background(cal_tpf)
            del cal_tpf
        else:
            tpf_input = tpf_path

        lc = halophot.extract_halo_lightcurve(
            tpf_input,
            sigclip=True,
            remove_contaminants=has_cat,
            catalog_table=cat_file if has_cat else None,
            offline=True,
            cache_dir=cache_dir,
            strict_hpc=True,
            compute_uncertainty=True,
        )
        used_out_stems.add(out_stem)
        out_fits = out_dir / f"{out_stem}_halo_lc_tv.fits"
        halophot.write_fits(lc, out_fits, overwrite=True)
        rows.append(
            {
                "target": out_stem,
                "pipeline_mode": pipeline_mode,
                "status": "ok",
                "n_cad": len(lc),
                "fits_file": out_fits.name,
                "error": "",
            }
        )
    except Exception as exc:
        print(f"ERROR [{out_stem}]: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        rows.append(
            {
                "target": out_stem,
                "pipeline_mode": pipeline_mode,
                "status": "failed",
                "n_cad": 0,
                "fits_file": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        tpf_input = None
        lc = None

if rows:
    Table(rows=rows).write(out_dir / "summary.parquet", format="parquet", overwrite=True)

n_total = len(rows)
n_failed = sum(1 for r in rows if r["status"] == "failed")
n_ok = n_total - n_failed
if n_total == 0 or n_ok == 0 or (n_failed / n_total) > max_fail_frac:
    print(
        f"FATAL: {n_ok}/{n_total} targets succeeded ({n_failed} failed, "
        f"MAX_FAIL_FRAC={max_fail_frac}) in {manifest_path}; failing Slurm job.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)
PY
