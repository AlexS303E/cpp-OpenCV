#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>

#include "yolo.h"

int main() {
    setlocale(LC_ALL, "Russian");

    try {
        // === Инициализация камеры ===
        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);  // 640x480 для лучшей производительности
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);

        if (!cap.isOpened()) {
            std::cerr << "Ошибка: не удалось открыть камеру!" << std::endl;
            return -1;
        }

        std::cout << "Камера успешно открыта." << std::endl;

        // === Загрузка YOLO ===
        // ВАЖНО: Скачайте эти файлы и разместите в папке models рядом с exe!
        std::string modelCfg = "models/yolov3.cfg";
        std::string modelWeights = "models/yolov3.weights";
        std::string classesFile = "models/coco.names";

        YOLODetector detector(modelCfg, modelWeights, classesFile, 0.5f, 0.4f, 416, 416);

        cv::namedWindow("YOLOv3 Детекция", cv::WINDOW_AUTOSIZE);

        cv::Mat frame;
        bool running = true;

        std::cout << "Детекция запущена. Нажмите ESC или Q для выхода" << std::endl;

        // === Основной цикл обработки ===
        while (running) {
            if (!cap.read(frame)) {
                std::cerr << "Не удалось получить кадр с камеры!" << std::endl;
                break;
            }

            // Детекция объектов
            detector.detectAndDraw(frame);

            // Информация о выходе
            cv::putText(frame, "ESC/Q - выход", cv::Point(10, frame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

            cv::imshow("YOLOv3 Детекция", frame);

            // Обработка событий окна
            int key = cv::waitKey(1);

            // Проверка закрытия окна
            if (cv::getWindowProperty("YOLOv3 Детекция", cv::WND_PROP_VISIBLE) <= 0) {
                std::cout << "Окно закрыто пользователем." << std::endl;
                break;
            }

            // Проверка нажатия клавиш
            if (key == 27 || key == 'q' || key == 'Q') {
                std::cout << "Выход по нажатию клавиши." << std::endl;
                break;
            }
        }

        // Освобождение ресурсов
        cap.release();
        cv::destroyAllWindows();

    }
    catch (const std::exception& e) {
        std::cerr << "Критическая ошибка: " << e.what() << std::endl;
        std::cerr << "Убедитесь, что все файлы YOLOv3 загружены в папку models!" << std::endl;
        std::cerr << "См. README для инструкций по загрузке." << std::endl;
        return -1;
    }

    std::cout << "Программа завершена !" << std::endl;
    return 0;
}