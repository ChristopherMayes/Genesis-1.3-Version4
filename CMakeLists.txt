cmake_minimum_required(VERSION 3.12)
project(Genesis_1_3_Version4 CXX C)

option(USE_DPI "Use diagnostic plugin interface" OFF)
option(ENABLE_METAL "Build the Apple Silicon GPU backend (macOS only)" OFF)
option(ENABLE_CUDA "Build the NVIDIA GPU backend" OFF)

# Which NVIDIA architectures the CUDA backend is compiled for. The default
# covers the three cards it is written against -- A100 (80), L4 (89) and the
# consumer Blackwell parts such as the RTX 5080 (120) -- and adds PTX for the
# last of them so that a newer card still runs, by JIT, rather than failing to
# launch. Set it to "native" to build only for the card in the machine, which
# is quicker and is what to use while developing.
set(CUDA_ARCHS "80;89;120;120-virtual" CACHE STRING
    "CMAKE_CUDA_ARCHITECTURES for the CUDA backend (e.g. native, or 80;89;120)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENTIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
# set(CMAKE_INSTALL_RPATH "/usr/local/lib")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# developer options
set(CMAKE_VERBOSE_MAKEFILE ON)
add_compile_options(-g)

# OpenMPI: preprocessor macro to disable C++ interface (then we don't have to link libmpi_cxx)
add_compile_definitions(OMPI_SKIP_MPICXX)

list(APPEND CMAKE_PREFIX_PATH $ENV{FFTW_ROOT})

# container for needed librarires
list(APPEND _libraries)

# MacPorts installs a parallel HDF5 whose compiler wrapper supplies HDF5 and MPI
# itself, which makes it the easiest way to build on a Mac that has MacPorts and
# nothing else. It is only the easiest way when it is the whole toolchain. This
# runs after project(), so CMake has already detected a C and a C++ compiler and
# derived every flag from them; replacing just the C++ one leaves the C compiler
# and the flags belonging to somewhere else, and the link then goes out against
# two HDF5 and MPI installations at once. What that looks like is a page of
# undefined _ompi_mpi_* symbols after everything has compiled, which says nothing
# about its cause.
#
# It therefore stays the default where it is present, since that is what a
# MacPorts-only machine wants, but it can be switched off, and it says both that
# it happened and how to decline it.
option(USE_HDF5_COMPILER_WRAPPER
       "Use an HDF5 compiler wrapper when present, instead of finding MPI and \
HDF5 with find_package" ON)
set(HDF5_COMPILER_WRAPPER "/opt/local/bin/h5pcc" CACHE FILEPATH
    "Path to the HDF5 compiler wrapper to prefer, if it exists")

set(_hdf5_wrapper_in_use FALSE)
if(USE_HDF5_COMPILER_WRAPPER AND EXISTS "${HDF5_COMPILER_WRAPPER}")
  set(CMAKE_CXX_COMPILER "${HDF5_COMPILER_WRAPPER}")
  set(_hdf5_wrapper_in_use TRUE)
  message(STATUS "Using ${HDF5_COMPILER_WRAPPER} as the C++ compiler "
                 "(-DUSE_HDF5_COMPILER_WRAPPER=OFF to use your own)")

  # The mix is worth naming before it becomes a link error rather than after.
  get_filename_component(_h5pcc_dir "${HDF5_COMPILER_WRAPPER}" DIRECTORY)
  get_filename_component(_cc_dir "${CMAKE_C_COMPILER}" DIRECTORY)
  if(NOT _cc_dir STREQUAL _h5pcc_dir)
    message(WARNING
      "The C++ compiler is now ${HDF5_COMPILER_WRAPPER} but the C compiler is still "
      "${CMAKE_C_COMPILER}, so this build mixes two toolchains and will link "
      "against two HDF5 and MPI installations. If you meant to build with a "
      "conda environment or any other self-contained toolchain, configure with "
      "-DUSE_HDF5_COMPILER_WRAPPER=OFF.")
  endif()
endif()


# The name match is kept as well as the flag, because passing the wrapper
# directly as CMAKE_CXX_COMPILER has always worked and should go on working.
if (_hdf5_wrapper_in_use OR CMAKE_CXX_COMPILER MATCHES "h5pcc")
   Message(STATUS "HDF5 compiler wrapper in use - No additional libraries needed")
else()
   Message(STATUS "No HDF5 compiler wrapper in use - looking for compiler, MPI and HDF5")
   Message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
   Message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")

   # check for MPI
   find_package(MPI REQUIRED)
   include_directories(${MPI_INCLUDE_PATH})
   Message(STATUS "MPI Found")

   # and check for HDF5
   SET(HDF5_PREFER_PARALLEL TRUE)
   find_package(HDF5 REQUIRED)
   Message(STATUS "HDF5 Found")
   Message(STATUS "Version: ${HDF5_VERSION}")
   if (TARGET HDF5::HDF5)
#       list(APPEND _libraries MPI::MPI_C hdf5_mpich)  # needed for my local computer
       list(APPEND _libraries MPI::MPI_C HDF5::HDF5)
   else()
       list(APPEND _libraries MPI::MPI_C)
   endif()
   include_directories(${HDF5_C_INCLUDE_DIRS})
   if (HDF5_IS_PARALLEL)   # this should be required - needs a better check
        Message(STATUS "Parallel HDF5 supported")
   else()
        Message(STATUS "No parallel HDF5 Support")
   endif()
endif()

# check for FFTW libraries. Cray implementation are not found but have the environmental variable FFTW_ROOT defined.
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{FFTW_ROOT}/lib/pkgconfig/")
pkg_check_modules(FFTW QUIET fftw3 IMPORTED_TARGET)
if (TARGET PkgConfig::FFTW)
    message(STATUS "FFTW found")
    list(APPEND _libraries PkgConfig::FFTW)
    link_libraries(PkgConfig::FFTW)
else()
        message(STATUS "FFTW not found")
        message(STATUS "Excluding FFTW specific source code")
endif()


if(USE_DPI)
   set(SRC_DPI src/Main/RegPlugin.cpp src/Core/DiagnosticHook.cpp src/Core/DiagnosticHookBeam.cpp src/Core/DiagnosticHookLI.cpp)
endif()

# Apple Silicon GPU backend. Objective-C++, so it needs its own language and the
# Metal frameworks. The device-specific code is behind the G4_METAL macro, so a
# build without this option takes exactly the paths it always took: the tracking
# loop talks to GPUEngine (always compiled, see src/Core/GPUEngine.cpp), which
# then has no backend to offer it and says so.
if(ENABLE_METAL)
   if(NOT APPLE)
      message(FATAL_ERROR "ENABLE_METAL requires macOS")
   endif()
   enable_language(OBJCXX)
   set(CMAKE_OBJCXX_STANDARD 17)
   set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
   find_library(METAL_LIBRARY Metal REQUIRED)
   find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
   list(APPEND _libraries ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
   set(SRC_METAL src/Core/MetalEngine.mm)
   message(STATUS "Metal GPU backend enabled")
endif()

# NVIDIA GPU backend. One translation unit of CUDA C++, behind the G4_CUDA
# macro, so a build without this option takes exactly the paths it always took:
# the tracking loop talks to GPUEngine (always compiled, see
# src/Core/GPUEngine.cpp), which then has no backend to offer it and says so.
#
# The kernels are specialised on the transverse grid size at compile time, so
# there is no runtime compilation and nothing to ship beside the binary.
if(ENABLE_CUDA)
   if(CMAKE_VERSION VERSION_LESS 3.18)
      message(FATAL_ERROR "ENABLE_CUDA needs CMake 3.18 or newer for "
                          "CMAKE_CUDA_ARCHITECTURES; this is ${CMAKE_VERSION}")
   endif()
   # nvcc drives a host compiler of its own choosing unless told otherwise,
   # which in a conda environment is the system gcc rather than the one the
   # rest of the build uses. The two then disagree about the C++ ABI and the
   # link fails in a way that is hard to read, so pin it here.
   if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
      set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
   endif()
   set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})
   enable_language(CUDA)
   set(CMAKE_CUDA_STANDARD 17)
   set(CMAKE_CUDA_STANDARD_REQUIRED ON)
   # Keep IEEE semantics. The single-precision reformulations in CudaEngine.cu
   # are chosen against a known rounding behaviour, and -use_fast_math would
   # quietly replace the transcendentals under them.
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
   set(SRC_CUDA src/Core/CudaEngine.cu)
   message(STATUS "CUDA GPU backend enabled, architectures ${CMAKE_CUDA_ARCHITECTURES}")
endif()


add_custom_command(
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/include/version.h
    ALL
    COMMAND
    ${CMAKE_COMMAND} -D TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/git-hash.cmake
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
    get_git_hash
    ALL
    DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/include/version.h
   
)


include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


# some status print out before compilation
message(STATUS "Build Info:")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
MESSAGE(STATUS "CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS} )
MESSAGE(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )
Message(STATUS "Used libraries: ${_libraries}")

# objects for the main part - maybe needs a better name
add_library(genesis13 STATIC
        src/Util/BesselJ.cpp
        src/Util/GaussHermite.cpp
        src/Util/Hammerslay.cpp
        src/Util/Inverfc.cpp
        src/Util/RandomU.cpp
	    src/Util/SimpleHandshake.cpp
        src/Util/Sorting.cpp
        src/Util/StringProcessing.cpp
        src/Util/SemaFile.cpp
        src/Util/FFTObj.cpp
        src/Lattice/AlterLattice.cpp
        src/Lattice/Lattice.cpp
        src/Lattice/LatticeElements.cpp
        src/Lattice/LatticeParser.cpp
        src/Lattice/Optics.cpp
        src/Core/Beam.cpp
        src/Core/BeamSolver.cpp
        src/Core/Collective.cpp
        src/Core/Control.cpp
        src/Core/EFieldSolver.cpp
        src/Core/Field.cpp
        src/Core/FieldSolverADI.cpp
        src/Core/FieldSolverFFT.cpp
        src/Core/Gencore.cpp
        src/Core/GPUEngine.cpp
        src/Core/Incoherent.cpp
        src/Core/TrackBeam.cpp
        src/Core/Undulator.cpp
        src/Core/Diagnostic.cpp
        src/Core/DiagnosticUser.cpp
        src/IO/HDF5base.cpp
        src/IO/Output.cpp
        src/IO/readBeamHDF5.cpp
        src/IO/readFieldHDF5.cpp
        src/IO/SDDSBeam.cpp
        src/IO/writeBeamHDF5.cpp
        src/IO/writeFieldHDF5.cpp
        src/IO/writeWakeHDF5.cpp
        src/IO/readMapHDF5.cpp
        src/Loading/ImportBeam.cpp
        src/Loading/ImportField.cpp
        src/Loading/LoadBeam.cpp
        src/Loading/LoadField.cpp
        src/Loading/GenProfile.cpp
        src/Loading/QuietLoading.cpp
        src/Loading/Series.cpp
        src/Loading/ShotNoise.cpp
        src/Loading/SeriesManager.cpp
        src/Loading/SeriesParser.cpp
        src/Loading/Sequence.cpp
        src/Loading/ImportTransformation.cpp
        src/Main/AlterSetup.cpp
        src/Main/Dump.cpp
        src/Main/AlterBeam.cpp
        src/Main/EField.cpp
        src/Main/FieldManipulator.cpp
        src/Main/GenMain.cpp
        src/Main/mainwrap.cpp
        src/Main/Parser.cpp
        src/Main/Setup.cpp
        src/Main/SponRad.cpp
        src/Main/GenTime.cpp
        src/Main/Track.cpp
        src/Main/Wake.cpp
        ${SRC_DPI}
        ${SRC_METAL}
        ${SRC_CUDA}
)



target_include_directories(genesis13
	PRIVATE
	${CMAKE_CURRENT_BINARY_DIR}/include
)


# add optional macro definitions
if (TARGET PkgConfig::FFTW)
    message(STATUS "Defining the macro FFTW in the source code")
    target_compile_definitions(genesis13 PRIVATE FFTW=1)
endif()

if(USE_DPI)
   target_compile_definitions(genesis13 PRIVATE USE_DPI)
endif()

if(ENABLE_METAL)
   target_compile_definitions(genesis13 PRIVATE G4_METAL)
   # Automatic reference counting for the Objective-C objects held by the engine.
   set_source_files_properties(src/Core/MetalEngine.mm PROPERTIES
           COMPILE_OPTIONS "-fobjc-arc")
endif()

if(ENABLE_CUDA)
   target_compile_definitions(genesis13 PRIVATE G4_CUDA)
   # The engine chooses its device from the rank's position within its node, so
   # the CUDA translation unit needs the MPI headers as well.
   target_include_directories(genesis13 PRIVATE ${MPI_C_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
   find_package(CUDAToolkit REQUIRED)
   target_link_libraries(genesis13 PUBLIC CUDA::cudart)
endif()


add_dependencies(genesis13 get_git_hash)

add_executable(genesis4 src/Main/mainwrap.cpp)

target_link_libraries(genesis4 genesis13 stdc++ ${_libraries})


# Compilation of plugin testbench only when plugin system is requested
#if(USE_DPI)
#   add_subdirectory(misc/plugin_testbench)
#endif()
