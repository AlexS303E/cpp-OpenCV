#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "yolo.h"



int main() {
    setlocale(LC_ALL, "Russian");

    try {
        // === Инициализация камеры ===
        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);

        if (!cap.isOpened()) {
            std::cerr << "Ошибка: не удалось открыть камеру!" << std::endl;
            return -1;
        }
        

        // === Создание детектора ===
        YOLOv3Detector detector(0.5f, 0.4f, 416, 416);
        //YOLOv8Detector detector(0.5f, 0.45f, 640, 640);
        //YOLOv26Detector detector(0.5f, 0.45f, 640, 640);

        cv::namedWindow("YOLO Детекция", cv::WINDOW_AUTOSIZE);

        cv::Mat frame;
        std::cout << "\nДетекция запущена. Нажмите ESC или Q для выхода\n" << std::endl;

        // === Основной цикл ===
        while (true) {
            if (!cap.read(frame)) {
                std::cerr << "Не удалось получить кадр!" << std::endl;
                break;
            }

            detector.detectAndDraw(frame);

            cv::putText(frame, "[ESC/Q] - выход", cv::Point(10, frame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

            cv::imshow("YOLO Детекция", frame);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;

            if (cv::getWindowProperty("YOLO Детекция", cv::WND_PROP_VISIBLE) < 1) {
                std::cout << "Окно закрыто. Завершение работы программы." << std::endl;
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

    }
    catch (const std::exception& e) {
        std::cerr << "\nКритическая ошибка: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}