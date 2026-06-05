docker run --rm \
  -v /home/rawg/workspace/pacman:/workspace/pacman \
  -w /workspace/pacman \
  pacman:local \
  bash -c '
    set -e
    doxygen --version
    BUILD_DEBUG="build-debug"
    BUILD_RELEASE="build-release"
    COVERAGE_DIR="coverage"

    echo "========================================="
    echo "Stage 1 – test:debug-coverage"
    echo "========================================="
    cmake \
      -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_MODULE=ON \
      -DBUILD_TESTS=ON \
      -DENABLE_COVERAGE=ON \
      -S . -B "${BUILD_DEBUG}"
    cmake --build "${BUILD_DEBUG}" -- -j$(nproc)
    ctest --test-dir "${BUILD_DEBUG}" -j$(nproc) --output-on-failure
    mkdir -p "${COVERAGE_DIR}"
    gcovr --root . --object-directory "${BUILD_DEBUG}" \
      --exclude ".*tests/.*" --exclude ".*pybind11/.*" \
      --xml --output "${COVERAGE_DIR}/coverage.xml"
    gcovr --root . --object-directory "${BUILD_DEBUG}" \
      --exclude ".*tests/.*" --exclude ".*pybind11/.*" \
      --html --html-details --output "${COVERAGE_DIR}/index.html"
    gcovr --root . --object-directory "${BUILD_DEBUG}" \
      --exclude ".*tests/.*" --exclude ".*pybind11/.*"

    echo ""
    echo "========================================="
    echo "Stage 2 – build:release"
    echo "========================================="
    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_MODULE=ON \
      -DBUILD_TESTS=ON \
      -S . -B "${BUILD_RELEASE}"
    cmake --build "${BUILD_RELEASE}" -- -j$(nproc)

    echo ""
    echo "========================================="
    echo "Stage 3 – test:release"
    echo "========================================="
    ctest --test-dir "${BUILD_RELEASE}" -j$(nproc) --output-on-failure \
      --output-junit "${BUILD_RELEASE}/Testing/junit.xml"

    echo ""
    echo "========================================="
    echo "Stage 4 – docs:build"
    echo "========================================="
    if [ -f Doxyfile ]; then
      doxygen Doxyfile
    else
      echo "SKIP: no Doxyfile found"
    fi
  '