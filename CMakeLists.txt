cmake_minimum_required(VERSION 3.12)

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(torch_alt LANGUAGES C CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Output directory ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- OpenCL setup ---
find_library(OpenCL_LIBRARY OpenCL)
find_path(OpenCL_INCLUDE_DIR CL/cl.h)
if(NOT OpenCL_LIBRARY)
    message(FATAL_ERROR "OpenCL not found")
endif()

# --- Include directories ---
include_directories(
    ${PROJECT_SOURCE_DIR}           # project root, allows ../ includes
    ${PROJECT_SOURCE_DIR}/include   # main headers
    ${OpenCL_INCLUDE_DIR}
)

# --- Gather all source files recursively ---
file(GLOB_RECURSE SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.c
)

# --- Build executable ---
add_executable(torch_alt ${SRC_FILES})
target_link_libraries(torch_alt PRIVATE ${OpenCL_LIBRARY})

# --- Auto-create directories if missing ---
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/include/layers)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/include/losses)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/include/optimizers)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/include/data)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/include/utils)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src/layers)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src/losses)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src/optimizers)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src/data)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src/utils)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/kernels)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)

# --- Create dummy main.cpp if missing ---
set(MAIN_CPP ${PROJECT_SOURCE_DIR}/src/main.cpp)
if(NOT EXISTS ${MAIN_CPP})
    file(WRITE ${MAIN_CPP}
"#include <iostream>\nint main(){ std::cout << \"Torch Alt OK\\n\"; return 0; }\n")
endif()

# --- Custom run target ---
# add_custom_target(run
#     COMMAND torch_alt
#     DEPENDS torch_alt
#     WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
# )


add_custom_target(run
    COMMAND ./torch_alt
    DEPENDS torch_alt
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

