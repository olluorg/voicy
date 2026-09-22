#!/bin/sh
# Builds libct2shim.so against the prebuilt libctranslate2 in the lib dir.
#   rust/ct2shim/build.sh <ct2 include dir> <lib dir>
# The headers are CTranslate2's own at the tag matching the library (v4.8.2);
# scripts/native_setup.py fetches both and calls this. RPATH, not RUNPATH:
# it also covers libctranslate2's own dependencies (libgomp, cuBLAS 12).
set -e
INC=$1
LIB=$2
here=$(dirname "$0")
${CXX:-g++} -std=c++17 -O2 -fPIC -shared "$here/ct2shim.cpp" -I"$INC" \
  "$LIB"/libctranslate2-*.so.4.8.2 -Wl,--disable-new-dtags,-rpath,'$ORIGIN' -o "$LIB/libct2shim.so"
