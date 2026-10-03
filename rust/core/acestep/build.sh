#!/usr/bin/env bash
# Build libvoicy-acestep: acestep.cpp with its ggml fork, statically, behind
# the C face in shim.cpp. Nothing but vac_* is exported, so llama.cpp's ggml in
# the same process never meets this one.
#
#   ACESTEP_SRC=~/src/acestep.cpp BACKEND=cuda rust/core/acestep/build.sh OUT_DIR
#
# BACKEND: cuda | vulkan | cpu (metal on macOS comes with cpu). CUDA wants
# nvcc: CUDA_HOME, or /usr/local/cuda. acestep.cpp is pinned below; its
# submodule (the ggml fork) must be checked out.
set -euo pipefail

ACESTEP_COMMIT=${ACESTEP_COMMIT:-}                  # пусто — то, что лежит в ACESTEP_SRC
SRC=$(realpath "${ACESTEP_SRC:?путь к acestep.cpp}")
BACKEND=${BACKEND:-cuda}
OUT=$(realpath -m "${1:?каталог для библиотеки}")
HERE=$(dirname "$(realpath "$0")")
BUILD=${BUILD_DIR:-$SRC/build-voicy-$BACKEND}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN)}

if [ -n "$ACESTEP_COMMIT" ]; then
    git -C "$SRC" checkout -q "$ACESTEP_COMMIT"
    git -C "$SRC" submodule update --init -q
fi

flags=(-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release
       -DGGML_BACKEND_DL=OFF -DGGML_NATIVE="${GGML_NATIVE:-OFF}")
libs=()
case $BACKEND in
    cuda)
        CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
        flags+=(-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME")
        [ -n "${CUDA_ARCH:-}" ] && flags+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
        libs+=(-L"$CUDA_HOME/lib" -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib/stubs" -lcudart -lcublas -lcublasLt -lcuda)
        ;;
    vulkan) flags+=(-DGGML_VULKAN=ON); libs+=(-lvulkan) ;;
    cpu) ;;
    *) echo "BACKEND: cuda | vulkan | cpu" >&2; exit 2 ;;
esac

cmake -S "$SRC" -B "$BUILD" "${flags[@]}" >/dev/null
cmake --build "$BUILD" --target acestep-core -j "$JOBS"

mkdir -p "$OUT"
# --whole-archive не нужен: обёртка тянет из архивов всё, что зовёт.
# --exclude-libs,ALL прячет символы статических библиотек, -Bsymbolic
# связывает их внутри себя, а не с ggml, загруженным раньше.
mapfile -t archives < <(find "$BUILD" -name '*.a' | grep -E 'acestep-core|yyjson|ggml')
# порядок: ядро, потом бэкенды ggml, потом ggml-base
order=()
for pat in acestep-core yyjson 'libggml\.a' 'ggml-(cuda|vulkan|metal|blas)' 'ggml-cpu' 'ggml-base'; do
    for a in "${archives[@]}"; do [[ $(basename "$a") =~ $pat ]] && order+=("$a"); done
done
c++ -std=c++17 -O2 -fPIC -shared -fvisibility=hidden \
    -I"$SRC/src" -I"$SRC" -I"$BUILD" -I"$SRC/ggml/include" -I"$SRC/vendor/yyjson" -DGGML_MAX_NAME=128 \
    "$HERE/shim.cpp" -o "$OUT/libvoicy-acestep.so" \
    -Wl,--start-group "${order[@]}" -Wl,--end-group \
    -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,-z,defs -Wl,-rpath,'$ORIGIN' \
    "${libs[@]}" -fopenmp -lpthread -ldl
echo "$OUT/libvoicy-acestep.so"
