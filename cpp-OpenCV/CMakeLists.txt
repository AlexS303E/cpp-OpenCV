cmake_minimum_required(VERSION 3.20)
project(MyOpenCVProject VERSION 1.0.0 LANGUAGES CXX)

# Указываем стандарт C++ для всего проекта
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Включаем поддержку папок в IDE (Visual Studio)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Настройка для vcpkg (если используется)
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# Поиск OpenCV (в корневом CMakeLists.txt, чтобы не искать в каждом подпроекте)
find_package(OpenCV REQUIRED COMPONENTS
    core
    highgui
    videoio
)

if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    
    # Экспортируем переменные OpenCV для использования в поддиректориях
    set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} CACHE INTERNAL "")
    set(OpenCV_LIBS ${OpenCV_LIBS} CACHE INTERNAL "")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or set OpenCV_DIR")
endif()

# Добавляем поддиректорию с проектом Camera
add_subdirectory(Camera)

# Дополнительные настройки (опционально)
option(BUILD_EXAMPLES "Build example programs" ON)

# Установка (опционально)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND NOT CMAKE_SKIP_INSTALL_RULES)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Camera/
            DESTINATION bin
            FILES_MATCHING PATTERN "*.exe")
endif()