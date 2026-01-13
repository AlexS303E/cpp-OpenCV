#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    setlocale(LC_ALL, "Russian");
    cv::VideoCapture cap(0);

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 60);

    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть камеру!" << std::endl;
        return -1;
    }

    std::cout << "Камера успешно открыта!" << std::endl;
    std::cout << "Для выхода нажмите ESC или закройте окно" << std::endl;

    cv::namedWindow("Веб-камера", cv::WINDOW_AUTOSIZE);

    // Дадим окну время на инициализацию
    cv::waitKey(100);

    cv::Mat frame;
    bool running = true;

    while (running) {
        // Пробуем получить кадр
        if (!cap.read(frame)) {
            std::cerr << "Не удалось получить кадр с камеры!" << std::endl;
            break;
        }

        cv::putText(frame,
            "Press ESC or Q to quit",
            cv::Point(10, frame.rows - 10),
            cv::FONT_HERSHEY_SIMPLEX,
            0.9,
            cv::Scalar(0, 0, 255),
            2);

        // Показываем кадр
        cv::imshow("Веб-камера", frame);

        // Обработка событий окна
        int key = cv::waitKey(30);

        // Проверяем, было ли закрыто окно
        // (в Windows это можно проверить через состояние окна)
        if (cv::getWindowProperty("Веб-камера", cv::WND_PROP_VISIBLE) <= 0) {
            std::cout << "Окно закрыто пользователем." << std::endl;
            return 0;
        }

        // Проверяем нажатие ESC
        if (key == 27) {
            std::cout << "Выход по нажатию ESC." << std::endl;
            break;
        }
        if (key == 'q' || key == 'Q') {
            std::cout << "Выход по нажатию Q." << std::endl;
            break;
        }

        // Если waitKey вернул -1 и окно все еще видимо, продолжаем цикл
        if (key == -1) {
            continue;
        }
    }

    // Освобождаем ресурсы
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Программа завершена." << std::endl;

    // Дадим время прочитать сообщение перед закрытием консоли
    std::cout << "Нажмите Enter для выхода..." << std::endl;
    std::cin.get();

    return 0;
}