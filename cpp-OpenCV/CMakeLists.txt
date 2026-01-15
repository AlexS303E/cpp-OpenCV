cmake_minimum_required(VERSION 3.20)
project(MyOpenCVProject VERSION 1.0.0 LANGUAGES CXX)

# === Общие настройки ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# === Поддержка vcpkg ===
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# === Поиск OpenCV (добавлены imgproc и dnn) ===
find_package(OpenCV REQUIRED COMPONENTS 
    core 
    highgui 
    videoio 
    imgproc  # Для обработки изображений
    dnn      # Для YOLO
)

if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or set OpenCV_DIR")
endif()

# === Создание исполняемого файла ===

add_executable(WebcamViewer
    src/main.cpp
    src/WebcamViewer.h
    src/WebcamViewer.cpp
    src/yolo.h
    src/yolo.cpp
    src/global.cpp
)

# === Настройки цели ===
set_target_properties(WebcamViewer PROPERTIES
    FOLDER "Examples"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    DEBUG_POSTFIX "_d"
)

# === Подключение библиотек ===
target_link_libraries(WebcamViewer PRIVATE ${OpenCV_LIBS})
target_include_directories(WebcamViewer PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# === Специфичные настройки для Windows ===
if(WIN32)
    target_compile_definitions(WebcamViewer PRIVATE 
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

# === Установка (опционально) ===
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND NOT CMAKE_SKIP_INSTALL_RULES)
    install(TARGETS WebcamViewer
            RUNTIME DESTINATION bin
            CONFIGURATIONS Release RelWithDebInfo)
endif()

add_custom_command(
    TARGET WebcamViewer PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/models
    COMMENT "Создание каталога для моделей..."
)

if(EXISTS ${CMAKE_SOURCE_DIR}/models AND IS_DIRECTORY ${CMAKE_SOURCE_DIR}/models)
    add_custom_command(
        TARGET WebcamViewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/models
            ${CMAKE_BINARY_DIR}/models
        COMMENT "Копирование всей папки models..."
    )
else()
    message(WARNING "Папка models не найдена: ${CMAKE_SOURCE_DIR}/models")
    
    # Проверяем наличие отдельных файлов для обратной совместимости
    if(NOT EXISTS ${CMAKE_BINARY_DIR}/bin/models/coco.names)
        # Альтернатива: скачать coco.names при сборке
        add_custom_command(
            TARGET WebcamViewer POST_BUILD
            COMMAND powershell -Command "
                if (!(Test-Path '${CMAKE_BINARY_DIR}/models/coco.names')) {
                    Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names' -OutFile '${CMAKE_BINARY_DIR}/models/coco.names'
                    Write-Host 'Файл классов загружен'
                } else {
                    Write-Host 'Файл классов уже существует'
                }
            "
            COMMENT "Скачивание файла классов COCO..."
        )
    endif()
endif()