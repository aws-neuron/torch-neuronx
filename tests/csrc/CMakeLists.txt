# CMakeLists.txt for TorchNeuronEager C++ unit tests

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${NRT_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})

# Create test_mocks static library with shared mock sources
add_library(test_mocks STATIC
    mocks/MockNRT.cpp
    mocks/MockNRT.h
    mocks/MockNeuronDevice.cpp
    mocks/MockNeuronDevice.h
    mocks/MockKernelExecution.h
    mocks/MockCopyUtils.cpp
    mocks/MockCopyUtils.h
    mocks/MockNRTHandler.cpp
    mocks/MockPythonUtils.cpp
    mocks/MockOperationExecutionEngine.h
    mocks/MockStreamImpl.h
)

target_include_directories(test_mocks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mocks
    ${NRT_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(test_mocks
    ${NRT_LIBRARIES}
    ${TORCH_LIBRARIES}
    gmock
    gtest
)

# Helper function to create test executables
function(add_neuron_test TEST_NAME)
    # Parse additional arguments as source files
    set(TEST_SOURCES ${ARGN})

    # Create test executable
    add_executable(${TEST_NAME} ${TEST_SOURCES})

    # Set up include directories
    target_include_directories(${TEST_NAME} PRIVATE
        /usr/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/mocks
        ${NRT_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )

    # Link libraries
    target_link_libraries(${TEST_NAME}
        core
        internal_ops
        stream
        aten_neuron
        test_mocks
        gtest
        gtest_main
        gmock
        ${NRT_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
        pthread
    )

    # Add to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # Set test properties
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 60
        LABELS "unit"
    )
endfunction()

# Enable testing
enable_testing()

# Create tests
add_neuron_test(CacheEntryTest core/compilation/CacheEntryTest.cpp)
add_neuron_test(CompilationCacheTest core/compilation/CompilationCacheTest.cpp)
add_neuron_test(NeuronCompilerTest core/compilation/NeuronCompilerTest.cpp)
add_neuron_test(ModelHandleCacheTest core/runtime/ModelHandleCacheTest.cpp)
add_neuron_test(NRTHandlerTest core/runtime/NRTHandlerTest.cpp)
add_neuron_test(NRTUtilsTest core/runtime/NRTUtilsTest.cpp)
add_neuron_test(StreamImplTest core/streams/StreamImplTest.cpp)
add_neuron_test(NeuronResourceManagerTest core/utils/NeuronResourceManagerTest.cpp)
add_neuron_test(KernelExecutionTest core/KernelExecutionTest.cpp)
add_neuron_test(NeuronDeviceTest core/NeuronDeviceTest.cpp)
add_neuron_test(NeuronTensorImplTest core/NeuronTensorImplTest.cpp)
add_neuron_test(StorageImplTest core/StorageImplTest.cpp)
add_neuron_test(OperationContextTest core/OperationContextTest.cpp)
add_neuron_test(OperationExecutionEngineTest core/OperationExecutionEngineTest.cpp)
add_neuron_test(ErrorHandlerTest core/utils/ErrorHandlerTest.cpp)
add_neuron_test(ErrorRecoveryTest core/utils/ErrorRecoveryTest.cpp)
add_neuron_test(CPUFallbackExecutorTest core/utils/CPUFallbackExecutorTest.cpp)
add_neuron_test(NeuronBindingsTest core/utils/NeuronBindingsTest.cpp)
