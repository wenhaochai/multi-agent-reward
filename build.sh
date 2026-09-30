#!/bin/bash
# Build radixark/miles:latest as a sif on della-vis1 (large-image recipe: sandbox + mksquashfs, see memory
# reference_della_apptainer_large_image_sif_truncated) and clone the matching miles source.
set -euo pipefail
P=/scratch/gpfs/GROUP/USER/project
B=$P/miles-q38-build
S=/dev/shm/USER_miles
export APPTAINER_CACHEDIR=$B/apptainer_cache APPTAINER_TMPDIR=$S/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
[ -d $P/miles-q38/.git ] || git clone -q https://github.com/radixark/miles $P/miles-q38
git -C $P/miles-q38 log --oneline -1
echo "[build] $(date -u +%FT%TZ) sandbox"
apptainer build --force --sandbox $S/sandbox docker://radixark/miles:latest
echo "[build] $(date -u +%FT%TZ) squashfs"
/usr/libexec/apptainer/bin/mksquashfs $S/sandbox $S/img.squashfs -noappend -comp gzip -mem 8G -processors 6 -no-progress
/usr/libexec/apptainer/bin/unsquashfs -s $S/img.squashfs | head -3
rm -f $S/img.sif
apptainer sif new $S/img.sif
apptainer sif add $S/img.sif $S/img.squashfs --datatype 4 --partfs 1 --parttype 2 --partarch 2 --groupid 1
cp $S/img.sif $B/miles.sif
apptainer exec $B/miles.sif python3 -c "import sglang, megatron.core, miles; print('sglang', sglang.__version__)" || true
rm -rf $S
echo "[build] $(date -u +%FT%TZ) DONE $B/miles.sif"
