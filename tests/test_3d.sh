#!/bin/sh

BUILD_FOLDER=build

OUTPUT_FOLDER=3d_tests_outputs/out
mkdir -p $OUTPUT_FOLDER

ERR_FOLDER=3d_tests_outputs/err
mkdir -p $ERR_FOLDER

PROFILING_FOLDER=3d_tests_outputs/profiling
mkdir -p $PROFILING_FOLDER

KERNEL_TIMER_LIB=$BUILD_FOLDER/install/lib64/libkp_kernel_timer.so
KP_READER=./$BUILD_FOLDER/kokkostools-build/profiling/simple-kernel-timer/kp_reader

MEMORY_USAGE_LIB=$BUILD_FOLDER/install/lib64/libkp_memory_usage.so

test_3d () {
    SOURCE_MESH="$1"
    TARGET_MESH="$2"
    export KOKKOS_TOOLS_LIBS="$KERNEL_TIMER_LIB"
    ./$BUILD_FOLDER/main ./meshes/3d/$SOURCE_MESH ./meshes/3d/$TARGET_MESH 1> $OUTPUT_FOLDER/${SOURCE_MESH%.*}_${TARGET_MESH%.*} 2> $ERR_FOLDER/${SOURCE_MESH%.*}_${TARGET_MESH%.*}
    unset KOKKOS_TOOLS_LIBS
    $KP_READER $(hostname)-*.dat > "$PROFILING_FOLDER/${SOURCE_MESH%.*}_${TARGET_MESH%.*}.dat"
    rm -f $(hostname)-*.dat

    export KOKKOS_TOOLS_LIBS="$MEMORY_USAGE_LIB"
    ./$BUILD_FOLDER/main ./meshes/3d/$SOURCE_MESH ./meshes/3d/$TARGET_MESH > /dev/null 2>&1
    unset KOKKOS_TOOLS_LIBS

    for f in *.memspace_usage; do
    SUFFIX=$(echo "$f" | sed -E 's/.*-(Cuda|Host)\.memspace_usage/\1/')

    if [[ "$SUFFIX" == "Cuda" || "$SUFFIX" == "Host" ]]; then
        NEWNAME="${SOURCE_MESH%.*}_${TARGET_MESH%.*}.${SUFFIX}MemoryUsage"
        mv "$f" "$PROFILING_FOLDER/$NEWNAME"
    fi
done
}

test_3d 0.03.vtu 0.003.vtu && test_3d 0.02.vtu 0.002.vtu && test_3d 0.01.vtu 0.001.vtu && test_3d 0.004.vtu 0.0005.vtu && test_3d 0.0005.vtu 0.0005.vtu
test_3d 0.003.vtu 0.03.vtu && test_3d 0.002.vtu 0.02.vtu && test_3d 0.001.vtu 0.01.vtu && test_3d 0.0005.vtu 0.004.vtu