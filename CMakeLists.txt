cmake_minimum_required(VERSION 3.26)
project(takumi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Locate Python and nanobind ------------------------------------------
# scikit-build-core sets SKBUILD and puts the right Python on the path.
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

# Ask the installed nanobind package where its CMake config lives, then load it.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# --- Build the packing extension -----------------------------------------
# Produces takumi/packing/_core.<abi>.so
#   STABLE_ABI: build against the CPython Stable ABI so one wheel works across
#               Python versions (3.12+). NB_STATIC links nanobind statically.
nanobind_add_module(
  packing_core
  STABLE_ABI
  NB_STATIC
  src/packing/packing.cpp
)

# The compiled module must be importable as `takumi.packing._core`, so name the
# output file `_core` and install it into the takumi/packing package directory.
set_target_properties(packing_core PROPERTIES OUTPUT_NAME "_core")
install(TARGETS packing_core LIBRARY DESTINATION takumi/packing)
