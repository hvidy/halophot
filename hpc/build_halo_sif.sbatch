#!/bin/bash
# Build the single-inode halo.sif Apptainer image (mundey + tessbkgd + halophot + jax[cuda12]).
# Needs internet access (Docker Hub, PyPI, GitHub), so submit from a trevor login node:
#   mkdir -p "$SCRATCH/logs" && sbatch hpc/build_halo_sif.sbatch
# Measured on Swinburne NT trevor (job 20702665, incl. Sirius Sector 6 e2e smoke test):
#   elapsed 19m 03s, MaxRSS 2.05 GiB (2144368K); sized at ~2x time and 8G for /fred page cache.
#SBATCH --job-name=build-halo-sif
#SBATCH --account=oz440
#SBATCH --output=/aphid/scratch-3month/%u/logs/%x-%j.out
#SBATCH --error=/aphid/scratch-3month/%u/logs/%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -eo pipefail  # omit -u: Lmod reads unset variables (PS1) inside `module load`

: "${MYFRED:=/fred/${OZ_PROJECT:-oz440}/$USER}"
: "${SCRATCH:=/aphid/scratch-3month/$USER}"

# Default to audited `dev` branches; for post-release builds override via:
#   MUNDEY_REF=v2.0.0 TESSBKGD_REF=v2.0.0 HALOPHOT_REF=v2.0.0 sbatch hpc/build_halo_sif.sbatch
MUNDEY_REF="${MUNDEY_REF:-dev}"
TESSBKGD_REF="${TESSBKGD_REF:-dev}"
HALOPHOT_REF="${HALOPHOT_REF:-dev}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -n "${DEF_FILE:-}" ]; then
    :
elif [ -f "$SUBMIT_DIR/hpc/virgil_halophot.def" ]; then
    DEF_FILE="$SUBMIT_DIR/hpc/virgil_halophot.def"
elif [ -f "$SUBMIT_DIR/virgil_halophot.def" ]; then
    DEF_FILE="$SUBMIT_DIR/virgil_halophot.def"
else
    echo "ERROR: Cannot find virgil_halophot.def from SUBMIT_DIR=$SUBMIT_DIR (set DEF_FILE=...)." >&2
    exit 1
fi

OUTDIR="$MYFRED/containers"
SIF="$MYFRED/containers/halo-$(date +%Y%m%d).sif"
mkdir -p "$OUTDIR" "$SCRATCH/apptainer"

# Stage rootfs, layer cache, and partial .sif inside one per-job scratch directory (never /fred or /home).
BUILD_DIR="$SCRATCH/apptainer/build-$SLURM_JOB_ID"
export APPTAINER_TMPDIR="$BUILD_DIR/tmp"
export APPTAINER_CACHEDIR="$BUILD_DIR/cache"
export APPTAINER_BIND="/fred,/aphid/scratch-3month${JOBFS:+,$JOBFS}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
PART="$BUILD_DIR/$(basename "$SIF").part"
trap 'rm -rf "$BUILD_DIR"' EXIT
df -h "$BUILD_DIR" | tail -1

module -q load apptainer
echo "job=${SLURM_JOB_ID} host=$(hostname) def=${DEF_FILE} mundey_ref=${MUNDEY_REF} tessbkgd_ref=${TESSBKGD_REF} halophot_ref=${HALOPHOT_REF} sif=${SIF} start=$(date)"
apptainer --version
start=$(date +%s)

# OzSTAR/NT has no /etc/subuid entry and no fakeroot(1) binary on compute/build nodes, so
# --fakeroot --ignore-fakeroot-command selects Apptainer's root-mapped user namespace cleanly.
BUILD_LOG="$BUILD_DIR/build.log"
if ! apptainer build --fakeroot --ignore-fakeroot-command \
    --bind "$APPTAINER_TMPDIR:/tmp" \
    --build-arg mundey_ref="$MUNDEY_REF" \
    --build-arg tessbkgd_ref="$TESSBKGD_REF" \
    --build-arg halophot_ref="$HALOPHOT_REF" \
    "$PART" "$DEF_FILE" 2>&1 | tee "$BUILD_LOG"; then
    echo "BUILD FAILED ($(apptainer --version)); first error:"
    grep -m1 -E '^(E:|FATAL|ERROR)|Error|error:' "$BUILD_LOG" || true
    exit 1
fi
mv -f "$PART" "$SIF"

# Quick CPU import and CLI smoke test before updating the stable halo.sif symlink.
JAX_PLATFORMS=cpu apptainer exec "$SIF" python - <<'PY'
import importlib.metadata as md
import jax
import mundey
import tessbkgd
import halophot

print("jax", jax.__version__, jax.devices())
for pkg in ("jaxlib", "equinox", "optimistix", "optax", "zodiax", "lightkurve", "mundey", "tessbkgd", "halophot"):
    print(pkg, md.version(pkg))
PY
JAX_PLATFORMS=cpu apptainer exec "$SIF" halo --help >/dev/null

(cd "$OUTDIR" && sha256sum "$(basename "$SIF")" > "$SIF.sha256")
ln -sfn "$SIF" "$MYFRED/containers/halo.sif"

ls -la "$SIF" "$MYFRED/containers/halo.sif"
echo "elapsed=$(( $(date +%s) - start ))s end=$(date)"
