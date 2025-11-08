cmake_minimum_required(VERSION 3.12)
project(torch_alt LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# OpenCL
find_library(OpenCL_LIBRARY OpenCL)
find_path(OpenCL_INCLUDE_DIR CL/cl.h)
if(NOT OpenCL_LIBRARY)
    message(FATAL_ERROR "OpenCL not found")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include ${OpenCL_INCLUDE_DIR})

# Source files
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.c)

# Build main binary
add_executable(torch_alt ${SRC_FILES})
target_link_libraries(torch_alt PRIVATE ${OpenCL_LIBRARY})

# Custom target: make run
add_custom_target(run
    COMMAND torch_alt
    DEPENDS torch_alt
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

# --- Auto-create folders and a dummy main.cpp if missing ---
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/kernels)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)

set(MAIN_CPP ${PROJECT_SOURCE_DIR}/src/main.cpp)
if(NOT EXISTS ${MAIN_CPP})
    file(WRITE ${MAIN_CPP}
"#include <iostream>\nint main(){ std::cout << \"Torch Alt OK\\n\"; return 0; }\n")
endif()

