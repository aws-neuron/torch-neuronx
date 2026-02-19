cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_neuronx)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CMake Build Type - default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Get PyTorch path manually
set(TORCH_PATH $ENV{LOCAL_TORCH_PATH})
if (NOT DEFINED ${TORCH_PATH})
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
        OUTPUT_VARIABLE TORCH_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

# Get NRT path
set(NRT_PATH $ENV{NRT_LOCAL_PATH})
if (NOT DEFINED ${NRT_PATH})
    set(NRT_PATH "/opt/aws/neuron")
endif()

# Set PyTorch include and library paths
set(TORCH_INCLUDE_DIRS "${TORCH_PATH}/include" "${TORCH_PATH}/include/torch/csrc/api/include")
set(TORCH_LIBRARY_DIRS "${TORCH_PATH}/lib")
set(TORCH_LIBRARIES "torch" "torch_cpu" "c10" "torch_python")

# Kineto headers for profiler integration (libkineto.h, ActivityType.h, IActivityProfiler.h)
set(KINETO_INCLUDE_DIRS "${TORCH_PATH}/include/kineto")

# Set NRT include and library paths
set(NRT_INCLUDE_DIRS "${NRT_PATH}/include")
set(NRT_LIBRARY_DIRS "${NRT_PATH}/lib")
set(NRT_LIBRARIES "nrt")

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-error=attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_API_INCLUDE_EXTENSION_H")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_EXTENSION_NAME=_C")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} '-DPYBIND11_COMPILER_TYPE=\"_gcc\"' '-DPYBIND11_STDLIB=\"_libstdcpp\"' '-DPYBIND11_BUILD_ABI=\"_cxxabi1018\"'")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fwrapv -O2")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/torch_neuronx/csrc)
include_directories(${NRT_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

# Link directories
link_directories(${NRT_LIBRARY_DIRS})
link_directories(${TORCH_LIBRARY_DIRS})

# Compilation subsystem
set(COMPILATION_SOURCES
    torch_neuronx/csrc/core/compilation/CacheEntry.cpp
    torch_neuronx/csrc/core/compilation/CompilationCache.cpp
    torch_neuronx/csrc/core/compilation/NeuronCompiler.cpp
)

set(COMPILATION_HEADERS
    torch_neuronx/csrc/core/compilation/CacheEntry.h
    torch_neuronx/csrc/core/compilation/CompilationCache.h
    torch_neuronx/csrc/core/compilation/NeuronCompiler.h
)

# Runtime subsystem
set(RUNTIME_SOURCES
    torch_neuronx/csrc/core/runtime/ModelHandleCache.cpp
    torch_neuronx/csrc/core/runtime/NRTHandler.cpp
    torch_neuronx/csrc/core/runtime/NRTUtils.cpp
)

set(RUNTIME_HEADERS
    torch_neuronx/csrc/core/runtime/ModelHandleCache.h
    torch_neuronx/csrc/core/runtime/NRTHandler.h
    torch_neuronx/csrc/core/runtime/NRTUtils.h
)

# Core library
set(CORE_SOURCES
    torch_neuronx/csrc/core/KernelExecution.cpp
    torch_neuronx/csrc/core/NeuronDevice.cpp
    torch_neuronx/csrc/core/OperationExecutionEngine.cpp
    torch_neuronx/csrc/core/utils/CPUFallbackExecutor.cpp
    torch_neuronx/csrc/core/utils/ErrorHandler.cpp
    torch_neuronx/csrc/core/utils/ErrorRecovery.cpp
    torch_neuronx/csrc/core/utils/NeuronResourceManager.cpp
    torch_neuronx/csrc/core/utils/PlatformUtils.cpp
    ${COMPILATION_SOURCES}
    ${RUNTIME_SOURCES}
)

set(CORE_HEADERS
    torch_neuronx/csrc/core/KernelExecution.h
    torch_neuronx/csrc/core/NeuronDevice.h
    torch_neuronx/csrc/core/NeuronLogging.h
    torch_neuronx/csrc/core/OperationContext.h
    torch_neuronx/csrc/core/OperationExecutionEngine.h
    torch_neuronx/csrc/core/utils/CPUFallbackExecutor.h
    torch_neuronx/csrc/core/utils/ErrorHandler.h
    torch_neuronx/csrc/core/utils/ErrorRecovery.h
    torch_neuronx/csrc/core/utils/NeuronResourceManager.h
    torch_neuronx/csrc/core/utils/PlatformUtils.h
    ${COMPILATION_HEADERS}
    ${RUNTIME_HEADERS}
)

add_library(core STATIC ${CORE_SOURCES})
target_link_libraries(core ${TORCH_LIBRARIES} ${NRT_LIBRARIES})
target_include_directories(core PRIVATE ${TORCH_INCLUDE_DIRS} ${NRT_INCLUDE_DIRS})

set_target_properties(core PROPERTIES
    COMPILE_FLAGS "-fvisibility=default"
    LINK_FLAGS "-Wl,--whole-archive"
)

# Internal ops library
set(INTERNAL_OPS_SOURCES
    torch_neuronx/csrc/ops/InternalOps.cpp
)

set(INTERNAL_OPS_HEADERS
    torch_neuronx/csrc/ops/InternalOps.h
    torch_neuronx/csrc/utils/CopyUtils.h
)

add_library(internal_ops STATIC ${INTERNAL_OPS_SOURCES})
target_link_libraries(internal_ops ${TORCH_LIBRARIES} ${NRT_LIBRARIES})
target_include_directories(internal_ops PRIVATE ${TORCH_INCLUDE_DIRS} ${NRT_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

set_target_properties(internal_ops PROPERTIES
    COMPILE_FLAGS "-fvisibility=default"
    LINK_FLAGS "-Wl,--whole-archive"
)

# Stream library
set(STREAM_SOURCES
    torch_neuronx/csrc/Stream.cpp
    torch_neuronx/csrc/c10/neuron/NeuronStream.cpp
    torch_neuronx/csrc/c10/neuron/NeuronEvent.cpp
    torch_neuronx/csrc/core/streams/StreamManager.cpp
    torch_neuronx/csrc/core/streams/StreamImpl.cpp
)

set(STREAM_HEADERS
    torch_neuronx/csrc/Stream.h
    torch_neuronx/csrc/c10/neuron/NeuronStream.h
    torch_neuronx/csrc/c10/neuron/NeuronEvent.h
    torch_neuronx/csrc/core/streams/StreamManager.h
    torch_neuronx/csrc/core/streams/StreamImpl.h
)

add_library(stream STATIC ${STREAM_SOURCES})
target_link_libraries(stream core internal_ops ${TORCH_LIBRARIES} ${NRT_LIBRARIES})
target_include_directories(stream PRIVATE ${TORCH_INCLUDE_DIRS} ${NRT_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

set_target_properties(stream PROPERTIES
    COMPILE_FLAGS "-fvisibility=default"
    LINK_FLAGS "-Wl,--whole-archive"
)

# Distributed library
set(DISTRIBUTED_SOURCES
    torch_neuronx/csrc/distributed/NeuronWork.cpp
    torch_neuronx/csrc/distributed/NeuronWatchdog.cpp
)

set(DISTRIBUTED_HEADERS
    torch_neuronx/csrc/distributed/NeuronWork.h
    torch_neuronx/csrc/distributed/NeuronWatchdog.h
)

add_library(distributed STATIC ${DISTRIBUTED_SOURCES})
target_link_libraries(distributed stream ${TORCH_LIBRARIES} ${NRT_LIBRARIES})
target_include_directories(distributed PRIVATE ${TORCH_INCLUDE_DIRS} ${NRT_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

set_target_properties(distributed PROPERTIES
    COMPILE_FLAGS "-fvisibility=default"
    LINK_FLAGS "-Wl,--whole-archive"
)

# ATen Neuron library
set(ATEN_NEURON_SOURCES
    torch_neuronx/csrc/aten/CopyKernel.cpp
    torch_neuronx/csrc/aten/NeuronNativeFunctions.cpp
    torch_neuronx/csrc/aten/RecordStream.cpp
    torch_neuronx/csrc/aten/TensorShape.cpp
    torch_neuronx/csrc/core/NeuronTensorImpl.cpp
    torch_neuronx/csrc/core/NeuronStorageImpl.cpp
    torch_neuronx/csrc/core/NeuronOpTracking.cpp
    torch_neuronx/csrc/core/NeuronHooksInterface.cpp
    torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.cpp
    torch_neuronx/csrc/core/allocator/NeuronHostAllocator.cpp
    torch_neuronx/csrc/utils/CopyUtils.cpp
)

set(ATEN_NEURON_HEADERS
    torch_neuronx/csrc/aten/NeuronNativeFunctions.h
    torch_neuronx/csrc/core/NeuronTensorImpl.h
    torch_neuronx/csrc/core/NeuronStorageImpl.h
    torch_neuronx/csrc/core/NeuronOpTracking.h
    torch_neuronx/csrc/core/NeuronGuardImpl.h
    torch_neuronx/csrc/core/NeuronHooksInterface.h
    torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h
    torch_neuronx/csrc/core/allocator/NeuronHostAllocator.h
    torch_neuronx/csrc/utils/CopyUtils.h
    torch_neuronx/csrc/utils/SymIntUtils.h
)

add_library(aten_neuron STATIC ${ATEN_NEURON_SOURCES})
target_link_libraries(aten_neuron core internal_ops stream ${TORCH_LIBRARIES} ${NRT_LIBRARIES})
target_include_directories(aten_neuron PRIVATE ${TORCH_INCLUDE_DIRS} ${NRT_INCLUDE_DIRS})

set_target_properties(aten_neuron PROPERTIES
    COMPILE_FLAGS "-fvisibility=default"
    LINK_FLAGS "-Wl,--whole-archive"
)

# Main extension library
# Note: Profiler sources are included directly in EXTENSION_SOURCES below
# to avoid static library linking issues
set(EXTENSION_SOURCES
    torch_neuronx/csrc/core/NeuronBindings.cpp
    torch_neuronx/csrc/core/NeuronGuardImpl.cpp
    torch_neuronx/csrc/aten/SdpBackendSelection.cpp
    torch_neuronx/csrc/amp/autocast_mode.cpp
    # Profiler sources (compiled directly to ensure proper linking)
    torch_neuronx/csrc/profiler/Profiler.cpp
    torch_neuronx/csrc/profiler/Session.cpp
    torch_neuronx/csrc/profiler/Registration.cpp
    torch_neuronx/csrc/profiler/TorchStubs.cpp
    torch_neuronx/csrc/profiler/NConfigParser.cpp
)

add_library(_C SHARED ${EXTENSION_SOURCES})
target_link_libraries(_C
    -Wl,--whole-archive
    core
    internal_ops
    stream
    distributed
    aten_neuron
    -Wl,--no-whole-archive
    ${TORCH_LIBRARIES}
    ${NRT_LIBRARIES}
    stdc++fs
)

target_include_directories(_C PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${KINETO_INCLUDE_DIRS}
    ${NRT_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_neuronx/csrc/profiler
)

set_target_properties(_C PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LINKER_LANGUAGE CXX
)

target_link_options(_C PRIVATE
    -Wl,--no-as-needed
    -Wl,-soname,_C.so
    -Wl,--enable-new-dtags,-rpath,"${NRT_LIBRARY_DIRS}"
    -Wl,--enable-new-dtags,-rpath,"${TORCH_LIBRARY_DIRS}"
)

# Installation
install(TARGETS _C DESTINATION .)

# ============================================================================
# Testing Configuration
# ============================================================================

# Enable testing. As of now, it is disabled by default.
# See tests/csrc/CMakeLists.txt for test build configuration.
option(BUILD_TESTS "Build C++ unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/csrc)
    message(STATUS "C++ unit tests enabled")
endif()
