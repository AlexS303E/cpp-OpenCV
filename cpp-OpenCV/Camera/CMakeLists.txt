# Подпроект Camera
project(WebcamViewer LANGUAGES CXX)

# Используем стандарт C++ из родительского CMakeLists.txt
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Исходные файлы
set(SOURCES
    main.cpp
    # WebcamViewer.cpp # если будет дополнительный файл
)

# Заголовочные файлы
set(HEADERS
    # WebcamViewer.h # если будет заголовочный файл
)

# Создаем исполняемый файл
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Устанавливаем свойства для отображения в IDE
set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "Examples"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    DEBUG_POSTFIX "_d"
)

# Подключаем OpenCV
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Добавляем include директории
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Настройки для Windows
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

# Установка (опционально)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND NOT CMAKE_SKIP_INSTALL_RULES)
    install(TARGETS ${PROJECT_NAME}
            RUNTIME DESTINATION bin
            CONFIGURATIONS Release RelWithDebInfo)
endif()