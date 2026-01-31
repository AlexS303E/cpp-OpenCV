#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "EdgeDetector.h"
#include "BitGrid.h"

// Вспомогательная функция для отображения информации о сжатии
std::string getCompressionMethodName(CompressionMethod method) {
    switch (method) {
    case COMPRESSION_NONE: return "NONE";
    case COMPRESSION_RLE: return "RLE";
    case COMPRESSION_LZ4: return "LZ4";
    case COMPRESSION_HUFFMAN: return "HUFFMAN";
    default: return "UNKNOWN";
    }
}

int main() {
    setlocale(LC_ALL, "Russian");

    try {
        // === Инициализация камеры ===
        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);

        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera!" << std::endl;
            return -1;
        }

        // === Создание детекторов ===
        CannyEdgeDetector cannyDetector(50.0, 150.0);
        CombinedEdgeDetector combinedDetector(50.0, 150.0, 2, 2);

        // Параметры по умолчанию
        bool useCombinedDetector = false;
        bool showOnlyEdges = false;
        bool useBitGridMode = false;
        bool useCompressedMode = false;
        CompressionMethod compressionMethod = COMPRESSION_RLE;

        double cannyThresh1 = 50.0, cannyThresh2 = 150.0;
        double combinedThresh1 = 50.0, combinedThresh2 = 150.0;
        int dilateSize = 2, erodeSize = 2;

        // Для измерения FPS
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        float fps = 0.0f;

        cv::namedWindow("Edge Detector", cv::WINDOW_AUTOSIZE);

        cv::Mat frame;

        std::cout << "\n═══════════════════════════════════════════════════\n";
        std::cout << "       Детекция границ с битовой сеткой\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        std::cout << "Управление:\n";
        std::cout << "  [1/2] - Переключить детектор (Canny/Combined)\n";
        std::cout << "  [+/-] - Изменить пороги Канни\n";
        std::cout << "  [c/C] - Переключить режим отображения (overlay/edges only)\n";
        std::cout << "  [b/B] - Включить/выключить режим BitGrid\n";
        std::cout << "  [z/Z] - Включить/выключить сжатие BitGrid\n";
        std::cout << "  [m/M] - Сменить метод сжатия\n";
        std::cout << "  [d/D] - Увеличить/уменьшить дилатацию (Combined)\n";
        std::cout << "  [e/E] - Увеличить/уменьшить эрозию (Combined)\n";
        std::cout << "  [r/R] - Сбросить параметры\n";
        std::cout << "  [s/S] - Сохранить текущий кадр/битовую сетку\n";
        std::cout << "  [ESC/Q] - Выход\n";
        std::cout << "═══════════════════════════════════════════════════\n\n";

        // === Главный цикл обработки ===
        while (true) {
            if (!cap.read(frame)) {
                std::cerr << "Failed to grab frame!" << std::endl;
                break;
            }

            // Измерение FPS
            frameCount++;
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);

            if (elapsedTime.count() >= 1000) {
                fps = frameCount * 1000.0f / elapsedTime.count();
                frameCount = 0;
                lastTime = currentTime;
            }

            cv::Mat originalFrame = frame.clone();

            // Обработка в зависимости от режима
            if (useBitGridMode) {
                // Получаем битовую сетку в зависимости от выбранного детектора
                BitGrid edgeGrid;

                if (useCombinedDetector) {
                    edgeGrid = combinedDetector.getEdgeBitGrid(frame);
                }
                else {
                    edgeGrid = cannyDetector.getEdgeBitGrid(frame);
                }

                if (useCompressedMode) {
                    // Режим сжатой битовой сетки
                    // Сжимаем битовую сетку
                    auto compressedData = edgeGrid.compress(compressionMethod);
                    auto compInfo = edgeGrid.getCompressionInfo(compressedData);

                    // Распаковываем для отображения
                    BitGrid decompressedGrid;
                    decompressedGrid.decompress(compressedData);

                    // Конвертируем в изображение
                    cv::Mat edgeImage = decompressedGrid.toImage();
                    cv::cvtColor(edgeImage, frame, cv::COLOR_GRAY2BGR);

                    // Отображаем информацию о сжатии
                    std::string compressionInfo = "COMPRESSED BITGRID [" +
                        getCompressionMethodName(compressionMethod) + "]";

                    cv::putText(frame, compressionInfo, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

                    cv::putText(frame, "Original: " + std::to_string(compInfo.originalSize) + " B",
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 1);

                    cv::putText(frame, "Compressed: " + std::to_string(compInfo.compressedSize) + " B",
                        cv::Point(10, 85), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 1);

                    cv::putText(frame, "Ratio: " + std::to_string(compInfo.ratio * 100.0f).substr(0, 4) + "%",
                        cv::Point(10, 110), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 1);

                    // Статистика границ
                    int edgesCount = edgeGrid.countTrue();
                    float edgesDensity = edgeGrid.density() * 100.0f;

                    cv::putText(frame, "Edges: " + std::to_string(edgesCount),
                        cv::Point(10, 135), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 1);

                    cv::putText(frame, "Density: " + std::to_string(edgesDensity).substr(0, 4) + "%",
                        cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 1);

                }
                else {
                    // Режим обычной битовой сетки (без сжатия)
                    // Конвертируем в изображение
                    cv::Mat edgeImage = edgeGrid.toImage();
                    cv::cvtColor(edgeImage, frame, cv::COLOR_GRAY2BGR);

                    // Отображаем информацию
                    cv::putText(frame, "BITGRID MODE", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

                    cv::putText(frame, "Memory: " + std::to_string(edgeGrid.byteSize()) + " bytes",
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 1);

                    // Статистика границ
                    int edgesCount = edgeGrid.countTrue();
                    float edgesDensity = edgeGrid.density() * 100.0f;

                    cv::putText(frame, "Edges: " + std::to_string(edgesCount),
                        cv::Point(10, 85), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 1);

                    cv::putText(frame, "Density: " + std::to_string(edgesDensity).substr(0, 4) + "%",
                        cv::Point(10, 110), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 1);
                }

            }
            else {
                // Обычный режим (без BitGrid)
                if (useCombinedDetector) {
                    if (showOnlyEdges) {
                        combinedDetector.detectOnlyEdges(frame);
                        cv::putText(frame, "Mode: Edges Only (Combined Method)", cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                    }
                    else {
                        combinedDetector.detectAndDraw(frame);
                    }
                }
                else {
                    if (showOnlyEdges) {
                        cannyDetector.detectOnlyEdges(frame);
                        cv::putText(frame, "Mode: Edges Only (Canny)", cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                    }
                    else {
                        cannyDetector.detectAndDraw(frame);
                    }
                }

                // Отображаем информацию о детекторе
                std::string detectorName = useCombinedDetector ?
                    "Combined Edge Detector" : "Canny Edge Detector";

                cv::putText(frame, "Detector: " + detectorName,
                    cv::Point(10, frame.rows - 100), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 200, 0), 2);
            }

            // Отображение информации о режиме и FPS
            std::string modeInfo;
            if (useBitGridMode) {
                modeInfo = useCompressedMode ? "[Compressed BitGrid]" : "[BitGrid]";
            }
            else {
                modeInfo = showOnlyEdges ? "[Edges Only]" : "[Overlay]";
            }

            cv::putText(frame, modeInfo, cv::Point(frame.cols - 200, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 100, 0), 2);

            // Отображение FPS
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
            cv::putText(frame, fpsText, cv::Point(frame.cols - 150, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            // Отображение подсказок управления
            cv::putText(frame, "[ESC/Q] - Exit", cv::Point(10, frame.rows - 70),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            cv::putText(frame, "[1/2] - Switch Detector", cv::Point(10, frame.rows - 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
            cv::putText(frame, "[b] - BitGrid, [z] - Compress", cv::Point(10, frame.rows - 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

            // Если включен режим сжатия, показываем текущий метод
            if (useCompressedMode) {
                cv::putText(frame, "Compression: " + getCompressionMethodName(compressionMethod),
                    cv::Point(frame.cols - 200, 90), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(200, 200, 0), 1);
            }

            cv::imshow("Edge Detector", frame);

            // Обработка нажатий клавиш
            int key = cv::waitKey(1);

            // Выход
            if (key == 27 || key == 'q' || key == 'Q') break;

            // Переключение детекторов
            if (key == '1') {
                useCombinedDetector = false;
                std::cout << "Switched to Canny Edge Detector" << std::endl;
            }

            if (key == '2') {
                useCombinedDetector = true;
                std::cout << "Switched to Combined Edge Detector" << std::endl;
            }

            // Переключение режима отображения
            if (key == 'c' || key == 'C') {
                showOnlyEdges = !showOnlyEdges;
                std::cout << "Mode switched: "
                    << (showOnlyEdges ? "edges only" : "overlay mode")
                    << std::endl;
            }

            // Включение/выключение режима BitGrid
            if (key == 'b' || key == 'B') {
                useBitGridMode = !useBitGridMode;
                std::cout << "BitGrid mode: " << (useBitGridMode ? "ON" : "OFF") << std::endl;
            }

            // Включение/выключение режима сжатия
            if (key == 'z' || key == 'Z') {
                useCompressedMode = !useCompressedMode;
                std::cout << "Compressed BitGrid mode: " <<
                    (useCompressedMode ? "ON" : "OFF") << std::endl;
            }

            // Смена метода сжатия
            if (key == 'm' || key == 'M') {
                int currentMethod = static_cast<int>(compressionMethod);
                currentMethod = (currentMethod + 1) % 4;
                compressionMethod = static_cast<CompressionMethod>(currentMethod);
                std::cout << "Compression method: " <<
                    getCompressionMethodName(compressionMethod) << std::endl;
            }

            // Сброс параметров для текущего детектора
            if (key == 'r' || key == 'R') {
                if (useCombinedDetector) {
                    combinedThresh1 = 50.0; combinedThresh2 = 150.0;
                    dilateSize = 2; erodeSize = 2;
                    combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
                    std::cout << "Combined detector parameters reset: thresholds " << combinedThresh1 << ", " << combinedThresh2
                        << ", dilation " << dilateSize << ", erosion " << erodeSize << std::endl;
                }
                else {
                    cannyThresh1 = 50.0; cannyThresh2 = 150.0;
                    cannyDetector = CannyEdgeDetector(cannyThresh1, cannyThresh2);
                    std::cout << "Canny detector parameters reset: " << cannyThresh1 << ", " << cannyThresh2 << std::endl;
                }
            }

            // Изменение порогов Канни для текущего детектора
            if (key == '+' || key == '=') {
                if (useCombinedDetector) {
                    combinedThresh1 += 10; combinedThresh2 += 20;
                    combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
                    std::cout << "Combined Canny thresholds increased: " << combinedThresh1 << ", " << combinedThresh2 << std::endl;
                }
                else {
                    cannyThresh1 += 10; cannyThresh2 += 20;
                    cannyDetector = CannyEdgeDetector(cannyThresh1, cannyThresh2);
                    std::cout << "Canny thresholds increased: " << cannyThresh1 << ", " << cannyThresh2 << std::endl;
                }
            }

            if (key == '-' || key == '_') {
                if (useCombinedDetector) {
                    combinedThresh1 = std::max(10.0, combinedThresh1 - 10);
                    combinedThresh2 = std::max(30.0, combinedThresh2 - 20);
                    combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
                    std::cout << "Combined Canny thresholds decreased: " << combinedThresh1 << ", " << combinedThresh2 << std::endl;
                }
                else {
                    cannyThresh1 = std::max(10.0, cannyThresh1 - 10);
                    cannyThresh2 = std::max(30.0, cannyThresh2 - 20);
                    cannyDetector = CannyEdgeDetector(cannyThresh1, cannyThresh2);
                    std::cout << "Canny thresholds decreased: " << cannyThresh1 << ", " << cannyThresh2 << std::endl;
                }
            }

            // Изменение параметров дилатации/эрозии только для Combined детектора
            if (useCombinedDetector) {
                if (key == 'd') {
                    dilateSize = std::min(10, dilateSize + 1);
                    combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
                    std::cout << "Dilation size increased: " << dilateSize << std::endl;
                }

                if (key == 'D') {
                    dilateSize = std::max(1, dilateSize - 1);
                    combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
                    std::cout << "Dilation size decreased: " << dilateSize << std::endl;
                }

                if (key == 'e') {
                    erodeSize = std::min(10, erodeSize + 1);
                    combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
                    std::cout << "Erosion size increased: " << erodeSize << std::endl;
                }

                if (key == 'E') {
                    erodeSize = std::max(1, erodeSize - 1);
                    combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
                    std::cout << "Erosion size decreased: " << erodeSize << std::endl;
                }
            }

            // Сохранение текущего кадра
            if (key == 's' || key == 'S') {
                std::string filename;
                if (useBitGridMode) {
                    BitGrid edgeGrid;
                    if (useCombinedDetector) {
                        edgeGrid = combinedDetector.getEdgeBitGrid(originalFrame);
                    }
                    else {
                        edgeGrid = cannyDetector.getEdgeBitGrid(originalFrame);
                    }

                    if (useCompressedMode) {
                        filename = "compressed_bitgrid_" + std::to_string(time(nullptr)) + ".bgrid";
                        edgeGrid.save(filename, compressionMethod);
                        std::cout << "Saved compressed bitgrid to: " << filename << std::endl;
                    }
                    else {
                        filename = "bitgrid_" + std::to_string(time(nullptr)) + ".bgrid";
                        edgeGrid.save(filename);
                        std::cout << "Saved bitgrid to: " << filename << std::endl;
                    }
                }
                else {
                    filename = "frame_" + std::to_string(time(nullptr)) + ".jpg";
                    cv::imwrite(filename, frame);
                    std::cout << "Saved frame to: " << filename << std::endl;
                }
            }

            // Проверка, закрыто ли окно
            if (cv::getWindowProperty("Edge Detector", cv::WND_PROP_VISIBLE) < 1) {
                std::cout << "Window closed. Terminating program." << std::endl;
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

        std::cout << "\n═══════════════════════════════════════════════════\n";
        std::cout << "           Программа завершена\n";
        std::cout << "═══════════════════════════════════════════════════\n";
    }
    catch (const std::exception& e) {
        std::cerr << "\nCritical error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}