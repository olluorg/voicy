#!/usr/bin/env bash
# Build libvoicy-sa3: sa3.cpp (Stable Audio 3) with its ggml fork, statically,
# behind the C face in shim.cpp. Nothing but vsa_* is exported, so llama.cpp's
# ggml in the same process never meets this one.
#
#   SA3_SRC=~/src/sa3.cpp BACKEND=cuda rust/core/sa3/build.sh OUT_DIR
#
# BACKEND: cuda | vulkan | cpu. CUDA wants nvcc: CUDA_HOME, or /usr/local/cuda.
# The ggml submodule of sa3.cpp must be checked out.
set -euo pipefail

SRC=$(realpath "${SA3_SRC:?путь к sa3.cpp}")
BACKEND=${BACKEND:-cuda}
OUT=$(realpath -m "${1:?каталог для библиотеки}")
HERE=$(dirname "$(realpath "$0")")
BUILD=${BUILD_DIR:-$SRC/build-voicy-$BACKEND}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN)}

flags=(-DSA3_STATIC=ON -DSA3_BUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release
       -DGGML_BACKEND_DL=OFF -DGGML_NATIVE="${GGML_NATIVE:-OFF}")
libs=()
case $BACKEND in
    cuda)
        CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
        flags+=(-DSA3_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME")
        [ -n "${CUDA_ARCH:-}" ] && flags+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
        libs+=(-L"$CUDA_HOME/lib" -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib/stubs" -lcudart -lcublas -lcublasLt -lcuda)
        ;;
    vulkan) flags+=(-DSA3_VULKAN=ON); libs+=(-lvulkan) ;;
    cpu) ;;
    *) echo "BACKEND: cuda | vulkan | cpu" >&2; exit 2 ;;
esac

cmake -S "$SRC" -B "$BUILD" "${flags[@]}" >/dev/null
# статический libsa3 не тянет за собой сборку ggml — цели называются явно
ggml_targets=(ggml ggml-base ggml-cpu)
case $BACKEND in cuda) ggml_targets+=(ggml-cuda) ;; vulkan) ggml_targets+=(ggml-vulkan) ;; esac
cmake --build "$BUILD" --target sa3_shared "${ggml_targets[@]}" -j "$JOBS"

mkdir -p "$OUT"
mapfile -t archives < <(find "$BUILD" -name '*.a')
order=()
for pat in '^libsa3\.a' 'sa3' 'libggml\.a' 'ggml-(cuda|vulkan|metal|blas)' 'ggml-cpu' 'ggml-base'; do
    for a in "${archives[@]}"; do
        b=$(basename "$a")
        [[ $b =~ $pat ]] && [[ ! " ${order[*]} " =~ " $a " ]] && order+=("$a")
    done
done
c++ -std=c++17 -O2 -fPIC -shared -fvisibility=hidden -I"$SRC/src" \
    "$HERE/shim.cpp" -o "$OUT/libvoicy-sa3.so" \
    -Wl,--start-group "${order[@]}" -Wl,--end-group \
    -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,-z,defs -Wl,-rpath,'$ORIGIN' \
    "${libs[@]}" -fopenmp -lpthread -ldl
echo "$OUT/libvoicy-sa3.so"
