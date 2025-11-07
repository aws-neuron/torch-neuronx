cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_neuron)

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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/torch_neuron/csrc)
include_directories(${NRT_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

# Link directories
link_directories(${NRT_LIBRARY_DIRS})
link_directories(${TORCH_LIBRARY_DIRS})

# Compilation subsystem
set(COMPILATION_SOURCES
    torch_neuron/csrc/core/compilation/CacheEntry.cpp
    torch_neuron/csrc/core/compilation/CompilationCache.cpp
    torch_neuron/csrc/core/compilation/NeuronCompiler.cpp
)

set(COMPILATION_HEADERS
    torch_neuron/csrc/core/compilation/CacheEntry.h
    torch_neuron/csrc/core/compilation/CompilationCache.h
    torch_neuron/csrc/core/compilation/NeuronCompiler.h
)

# Runtime subsystem
set(RUNTIME_SOURCES
    torch_neuron/csrc/core/runtime/ModelHandleCache.cpp
    torch_neuron/csrc/core/runtime/NRTHandler.cpp
    torch_neuron/csrc/core/runtime/NRTUtils.cpp
)

set(RUNTIME_HEADERS
    torch_neuron/csrc/core/runtime/ModelHandleCache.h
    torch_neuron/csrc/core/runtime/NRTHandler.h
    torch_neuron/csrc/core/runtime/NRTUtils.h
)

# Core library
set(CORE_SOURCES
    torch_neuron/csrc/core/KernelExecution.cpp
    torch_neuron/csrc/core/NeuronDevice.cpp
    torch_neuron/csrc/core/OperationExecutionEngine.cpp
    torch_neuron/csrc/core/utils/CPUFallbackExecutor.cpp
    torch_neuron/csrc/core/utils/ErrorHandler.cpp
    torch_neuron/csrc/core/utils/ErrorRecovery.cpp
    torch_neuron/csrc/core/utils/NeuronResourceManager.cpp
    torch_neuron/csrc/core/utils/PlatformUtils.cpp
    ${COMPILATION_SOURCES}
    ${RUNTIME_SOURCES}
)

set(CORE_HEADERS
    torch_neuron/csrc/core/KernelExecution.h
    torch_neuron/csrc/core/NeuronDevice.h
    torch_neuron/csrc/core/NeuronLogging.h
    torch_neuron/csrc/core/OperationContext.h
    torch_neuron/csrc/core/OperationExecutionEngine.h
    torch_neuron/csrc/core/utils/CPUFallbackExecutor.h
    torch_neuron/csrc/core/utils/ErrorHandler.h
    torch_neuron/csrc/core/utils/ErrorRecovery.h
    torch_neuron/csrc/core/utils/NeuronResourceManager.h
    torch_neuron/csrc/core/utils/PlatformUtils.h
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
    torch_neuron/csrc/ops/InternalOps.cpp
)

set(INTERNAL_OPS_HEADERS
    torch_neuron/csrc/ops/InternalOps.h
    torch_neuron/csrc/utils/CopyUtils.h
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
    torch_neuron/csrc/Stream.cpp
    torch_neuron/csrc/c10/neuron/NeuronStream.cpp
    torch_neuron/csrc/c10/neuron/NeuronEvent.cpp
    torch_neuron/csrc/core/streams/StreamManager.cpp
    torch_neuron/csrc/core/streams/StreamImpl.cpp
)

set(STREAM_HEADERS
    torch_neuron/csrc/Stream.h
    torch_neuron/csrc/c10/neuron/NeuronStream.h
    torch_neuron/csrc/c10/neuron/NeuronEvent.h
    torch_neuron/csrc/core/streams/StreamManager.h
    torch_neuron/csrc/core/streams/StreamImpl.h
)

add_library(stream STATIC ${STREAM_SOURCES})
target_link_libraries(stream core internal_ops ${TORCH_LIBRARIES} ${NRT_LIBRARIES})
target_include_directories(stream PRIVATE ${TORCH_INCLUDE_DIRS} ${NRT_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

set_target_properties(stream PROPERTIES
    COMPILE_FLAGS "-fvisibility=default"
    LINK_FLAGS "-Wl,--whole-archive"
)

# ATen Neuron library
set(ATEN_NEURON_SOURCES
    torch_neuron/csrc/aten/CopyKernel.cpp
    torch_neuron/csrc/aten/NeuronNativeFunctions.cpp
    torch_neuron/csrc/aten/RecordStream.cpp
    torch_neuron/csrc/aten/TensorShape.cpp
    torch_neuron/csrc/core/NeuronTensorImpl.cpp
    torch_neuron/csrc/core/NeuronStorageImpl.cpp
    torch_neuron/csrc/core/NeuronOpTracking.cpp
    torch_neuron/csrc/core/NeuronHooksInterface.cpp
    torch_neuron/csrc/core/allocator/NeuronCachingAllocator.cpp
    torch_neuron/csrc/core/allocator/NeuronHostAllocator.cpp
    torch_neuron/csrc/utils/CopyUtils.cpp
)

set(ATEN_NEURON_HEADERS
    torch_neuron/csrc/aten/NeuronNativeFunctions.h
    torch_neuron/csrc/core/NeuronTensorImpl.h
    torch_neuron/csrc/core/NeuronStorageImpl.h
    torch_neuron/csrc/core/NeuronOpTracking.h
    torch_neuron/csrc/core/NeuronGuardImpl.h
    torch_neuron/csrc/core/NeuronHooksInterface.h
    torch_neuron/csrc/core/allocator/NeuronCachingAllocator.h
    torch_neuron/csrc/core/allocator/NeuronHostAllocator.h
    torch_neuron/csrc/utils/CopyUtils.h
    torch_neuron/csrc/utils/SymIntUtils.h
)

add_library(aten_neuron STATIC ${ATEN_NEURON_SOURCES})
target_link_libraries(aten_neuron core internal_ops stream ${TORCH_LIBRARIES} ${NRT_LIBRARIES})
target_include_directories(aten_neuron PRIVATE ${TORCH_INCLUDE_DIRS} ${NRT_INCLUDE_DIRS})

set_target_properties(aten_neuron PROPERTIES
    COMPILE_FLAGS "-fvisibility=default"
    LINK_FLAGS "-Wl,--whole-archive"
)

# Main extension library
set(EXTENSION_SOURCES
    torch_neuron/csrc/core/NeuronBindings.cpp
    torch_neuron/csrc/core/NeuronGuardImpl.cpp
    torch_neuron/csrc/aten/SdpBackendSelection.cpp
    torch_neuron/csrc/amp/autocast_mode.cpp
)

add_library(_C SHARED ${EXTENSION_SOURCES})
target_link_libraries(_C
    -Wl,--whole-archive
    core
    internal_ops
    stream
    aten_neuron
    -Wl,--no-whole-archive
    ${TORCH_LIBRARIES}
    ${NRT_LIBRARIES}
    stdc++fs
)

target_include_directories(_C PRIVATE ${TORCH_INCLUDE_DIRS} ${NRT_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

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
option(BUILD_TESTS "Build C++ unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()

    # Find Google Test
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})

    # Mock library for testing
    add_library(test_mocks STATIC
        tests/csrc/mocks/MockNeuronDevice.cpp
        tests/csrc/mocks/MockNRT.cpp
        tests/csrc/mocks/MockPythonUtils.cpp
        tests/csrc/mocks/MockCopyUtils.cpp
    )
    target_include_directories(test_mocks PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/csrc/mocks
        ${NRT_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(test_mocks INTERFACE
        ${TORCH_LIBRARIES}
        gmock
    )

    # KernelExecution test
    add_executable(KernelExecutionTest
        tests/csrc/core/KernelExecutionTest.cpp
    )
    target_link_libraries(KernelExecutionTest
        core
        internal_ops
        stream
        aten_neuron
        test_mocks
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(KernelExecutionTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME KernelExecutionTest COMMAND KernelExecutionTest)

    # OperationContext test
    add_executable(OperationContextTest
        tests/csrc/core/OperationContextTest.cpp
    )
    target_link_libraries(OperationContextTest
        core
        internal_ops
        stream
        aten_neuron
        test_mocks
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(OperationContextTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME OperationContextTest COMMAND OperationContextTest)

    # CacheEntry test
    add_executable(CacheEntryTest
        tests/csrc/core/compilation/CacheEntryTest.cpp
    )
    target_link_libraries(CacheEntryTest
        core
        internal_ops
        stream
        aten_neuron
        test_mocks
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(CacheEntryTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME CacheEntryTest COMMAND CacheEntryTest)

    # CompilationCache test
    add_executable(CompilationCacheTest
        tests/csrc/core/compilation/CompilationCacheTest.cpp
    )
    target_link_libraries(CompilationCacheTest
        core
        internal_ops
        stream
        aten_neuron
        test_mocks
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(CompilationCacheTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME CompilationCacheTest COMMAND CompilationCacheTest)

    # NeuronCompiler test
    add_executable(NeuronCompilerTest
        tests/csrc/core/compilation/NeuronCompilerTest.cpp
    )
    target_link_libraries(NeuronCompilerTest
        core
        internal_ops
        stream
        aten_neuron
        test_mocks
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(NeuronCompilerTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME NeuronCompilerTest COMMAND NeuronCompilerTest)

    # NRTUtils test
    add_executable(NRTUtilsTest
        tests/csrc/core/runtime/NRTUtilsTest.cpp
    )
    target_link_libraries(NRTUtilsTest
        -Wl,--start-group
        test_mocks
        core
        internal_ops
        stream
        aten_neuron
        -Wl,--end-group
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(NRTUtilsTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME NRTUtilsTest COMMAND NRTUtilsTest)

    # ModelHandleCache test
    add_executable(ModelHandleCacheTest
        tests/csrc/core/runtime/ModelHandleCacheTest.cpp
    )
    target_link_libraries(ModelHandleCacheTest
        -Wl,--start-group
        test_mocks
        core
        internal_ops
        stream
        aten_neuron
        -Wl,--end-group
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(ModelHandleCacheTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME ModelHandleCacheTest COMMAND ModelHandleCacheTest)

    # NRTHandler test
    add_executable(NRTHandlerTest
        tests/csrc/core/runtime/NRTHandlerTest.cpp
    )
    target_link_libraries(NRTHandlerTest
        -Wl,--start-group
        test_mocks
        core
        internal_ops
        stream
        aten_neuron
        -Wl,--end-group
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(NRTHandlerTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME NRTHandlerTest COMMAND NRTHandlerTest)

    # OperationExecutionEngine test
    add_executable(OperationExecutionEngineTest
        tests/csrc/core/OperationExecutionEngineTest.cpp
    )
    target_link_libraries(OperationExecutionEngineTest
        -Wl,--start-group
        test_mocks
        core
        internal_ops
        stream
        aten_neuron
        -Wl,--end-group
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(OperationExecutionEngineTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME OperationExecutionEngineTest COMMAND OperationExecutionEngineTest)

    # NeuronResourceManager test
    add_executable(NeuronResourceManagerTest
        tests/csrc/core/utils/NeuronResourceManagerTest.cpp
    )
    target_link_libraries(NeuronResourceManagerTest
        -Wl,--start-group
        test_mocks
        core
        internal_ops
        stream
        aten_neuron
        -Wl,--end-group
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(NeuronResourceManagerTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME NeuronResourceManagerTest COMMAND NeuronResourceManagerTest)

    # StreamImpl test
    add_executable(StreamImplTest
        tests/csrc/core/streams/StreamImplTest.cpp
        tests/csrc/mocks/MockNRTHandler.cpp
    )
    target_link_libraries(StreamImplTest
        -Wl,--start-group
        test_mocks
        core
        internal_ops
        stream
        aten_neuron
        -Wl,--end-group
        -Wl,--allow-multiple-definition
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(StreamImplTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME StreamImplTest COMMAND StreamImplTest)

    # StreamManager test
    add_executable(StreamManagerTest
        tests/csrc/core/streams/StreamManagerTest.cpp
        tests/csrc/mocks/MockNRTHandler.cpp
    )
    target_link_libraries(StreamManagerTest
        -Wl,--start-group
        test_mocks
        core
        internal_ops
        stream
        aten_neuron
        -Wl,--end-group
        -Wl,--allow-multiple-definition
        ${TORCH_LIBRARIES}
        ${NRT_LIBRARIES}
        ${Python3_LIBRARIES}
        GTest::GTest
        GTest::Main
        gmock
        pthread
    )
    target_include_directories(StreamManagerTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${NRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    add_test(NAME StreamManagerTest COMMAND StreamManagerTest)

    message(STATUS "C++ unit tests enabled")
endif()
