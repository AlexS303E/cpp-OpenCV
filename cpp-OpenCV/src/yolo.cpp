#include "yolo.h"

void YOLODetector::loadClasses(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Ошибка: не удалось открыть файл классов: " + filename);
    }

    std::string className;
    while (std::getline(file, className)) {
        if (!className.empty()) {
            classNames.push_back(className);
        }
    }
}

void YOLODetector::generateColors() {
    // Генерация уникальных цветов для каждого класса
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);

    for (size_t i = 0; i < classNames.size(); ++i) {
        colors.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
    }
}

void YOLODetector::detectAndDraw(cv::Mat& frame) {
    // Создание blob из изображения
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false);

    // Установка blob как входа сети
    net.setInput(blob);

    // Получение имен выходных слоев
    std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();

    // Прямой проход
    std::vector<cv::Mat> outs;
    net.forward(outs, outNames);

    // Обработка выходов
    postprocess(frame, outs);

    // Отображение FPS
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time: %.2f ms", t);
    cv::putText(frame, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(0, 255, 0), 2);
}

void YOLODetector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        const float* data = reinterpret_cast<const float*>(outs[i].data);

        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;

            // Получение класса с максимальной уверенностью
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confidenceThreshold) {
                int centerX = static_cast<int>(data[0] * frame.cols);
                int centerY = static_cast<int>(data[1] * frame.rows);
                int width = static_cast<int>(data[2] * frame.cols);
                int height = static_cast<int>(data[3] * frame.rows);

                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(confidence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Применение Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    // Отрисовка результатов
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box, frame);
    }
}

void YOLODetector::drawPred(int classId, float conf, const cv::Rect& box, cv::Mat& frame) {
    // Отрисовка рамки
    cv::rectangle(frame, box, colors[classId % colors.size()], 2);

    // Формирование текста
    std::string label = cv::format("%.2f", conf);
    if (classId < static_cast<int>(classNames.size())) {
        label = classNames[classId] + ": " + label;
    }

    // Фон для текста
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = std::max(box.y, labelSize.height);
    cv::rectangle(frame, cv::Point(box.x, top - labelSize.height),
        cv::Point(box.x + labelSize.width, top + baseLine),
        colors[classId % colors.size()], cv::FILLED);

    // Текст
    cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(0, 0, 0), 1);
}