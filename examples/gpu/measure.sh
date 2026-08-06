#!/bin/sh
# Capture everything the notes still record as "not yet run" on an NVIDIA node.
# Prints a compact summary; paste that back.
#
#   cd examples/gpu
#   sh measure.sh ../../build-cuda/genesis4
#
# Genesis resolves the lattice and writes its output relative to the working
# directory, not the deck, so the timing runs happen inside a scratch directory
# with the lattice copied in beside them.

set -e
HERE=$(pwd)
G=$HERE/${1:-../../build-cuda/genesis4}
case ${1:-} in /*) G=$1 ;; esac
OUT=${OUT:-/tmp/l4meas}

export UCX_LOG_LEVEL=error
export UCX_TLS=self,sm
unset FI_PROVIDER          # inert on a ch4:ucx build; do not let it mislead

rm -rf "$OUT"; mkdir -p "$OUT"
cp Aramis.lat sase_cpu.in sase_gpu.in validate.in "$OUT"/

say() { printf '\n=== %s ===\n' "$1"; }

ncore() { nproc 2>/dev/null || sysctl -n hw.perflevel0.physicalcpu 2>/dev/null \
          || sysctl -n hw.ncpu 2>/dev/null || echo 1; }

say "node"
hostname
echo "cores: $(ncore)"
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
               --format=csv,noheader
else
    echo "no nvidia-smi; assuming a single device"
fi

say "1. correctness, single card"
(cd "$OUT" && "$G" validate.in 2>&1 | grep -Ei "backend:|transfer check|vs CPU over|Error")

say "2. the matrix, 73 cases"
python3 sweep.py --genesis "$G" --workdir "$OUT/sweep" --keep-going 2>&1 | tail -18

say "3. GPU scaling, 500 slices at ngrid 256"
cd "$OUT"
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c . || echo 1)
for n in $(echo 1 2 4 8 | tr ' ' '\n' | awk -v m="$NGPU" '$1<=m||$1==1'); do
    sed "s/^rootname *=.*/rootname = g$n/" sase_gpu.in > "g$n.in"
    printf -- '-- %s rank(s) --\n' "$n"
    { /usr/bin/time -p mpirun -n $n "$G" "g$n.in" ; } 2>&1 \
        | grep -Ei "backend:|steps in|^real|Error" || true
done

say "4. CPU baseline, same deck"
NC=$(ncore)
for n in 1 $NC; do
    sed "s/^rootname *=.*/rootname = c$n/" sase_cpu.in > "c$n.in"
    printf -- '-- %s rank(s) --\n' "$n"
    { /usr/bin/time -p mpirun -n $n "$G" "c$n.in" ; } 2>&1 \
        | grep -Ei "Total Wall Clock|^real|Error" || true
done

# Both paths at the same rank count, because the shot-noise realisation follows
# the slice distribution and comparing across rank counts measures that instead.
R=$NGPU; [ "$R" -gt 8 ] && R=8
say "5. GPU against CPU, both at $R ranks"
sed "s/^rootname *=.*/rootname = a8_cpu/" sase_cpu.in > a8_cpu.in
sed "s/^rootname *=.*/rootname = a8_gpu/" sase_gpu.in > a8_gpu.in
mpirun -n $R "$G" a8_cpu.in > /dev/null 2>&1
mpirun -n $R "$G" a8_gpu.in > /dev/null 2>&1
python3 "$HERE/compare.py" a8_cpu.out.h5 a8_gpu.out.h5 2>&1 | tail -20 \
    || echo "compare.py needs h5py and numpy in this environment"

say "done"
