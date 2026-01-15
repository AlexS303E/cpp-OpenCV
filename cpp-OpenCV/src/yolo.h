
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>



class YOLODetector {
private:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    float confidenceThreshold;
    float nmsThreshold;
    std::vector<cv::Scalar> colors;
    int inputWidth;
    int inputHeight;

public:
    YOLODetector(const std::string& modelCfg, const std::string& modelWeights,
        const std::string& classesFile, float confThreshold = 0.5f,
        float nmsThresh = 0.4f, int width = 416, int height = 416)
        : confidenceThreshold(confThreshold), nmsThreshold(nmsThresh),
        inputWidth(width), inputHeight(height) {

        // Загрузка классов
        loadClasses(classesFile);

        // Генерация цветов для каждого класса
        generateColors();

        // Загрузка сети YOLO
        net = cv::dnn::readNetFromDarknet(modelCfg, modelWeights);

        if (net.empty()) {
            throw std::runtime_error("Ошибка: не удалось загрузить модель YOLO!");
        }

        // Настройка вычислений (CPU/OpenCV, можно заменить на CUDA)
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::cout << "YOLOv3 модель загружена успешно!" << std::endl;
        std::cout << "Классов загружено: " << classNames.size() << std::endl;
    }

    void loadClasses(const std::string& filename);

    void generateColors();

    void detectAndDraw(cv::Mat& frame);

    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);

    void drawPred(int classId, float conf, const cv::Rect& box, cv::Mat& frame);
};