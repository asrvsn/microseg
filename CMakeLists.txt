cmake_minimum_required(VERSION 3.5)

project(microseg LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
if(DEFINED ENV{CONDA_PREFIX})
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
    message(STATUS "Setting Python hints for conda environment: ${Python_ROOT_DIR}")
    
    # Set Python executable based on platform
    if(WIN32)
        set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/python.exe")
        # Windows conda environments need explicit library and include paths
        set(Python_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
        set(Python_LIBRARY "$ENV{CONDA_PREFIX}/libs/python310.lib")
    else()
        set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    endif()
    
    # For conda environments, let CMake find the Python installation automatically
    # by setting the root directory - this is more reliable than manually setting paths
endif()

find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  nanobind_example_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  src/nanobind_example_ext.cpp
)

# Install directive for scikit-build-core
install(TARGETS nanobind_example_ext LIBRARY DESTINATION microseg)
