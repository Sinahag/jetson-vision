#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
using namespace dnn;

// Load names of classes
vector<string> getOutputNames(const Net &net)
{
    static vector<string> names;
    if (names.empty())
    {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

// Draw bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame, const vector<string> &classes)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
}

// Stereo camera calibration parameters (placeholders)
Mat cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR;
Mat R, T, E, F;
Mat R1, R2, P1, P2, Q;

void initStereoCalibration()
{
    // Load calibration parameters (this example assumes they are stored in a file)
    FileStorage fs("stereo_calib.yml", FileStorage::READ);
    fs["cameraMatrixL"] >> cameraMatrixL;
    fs["distCoeffsL"] >> distCoeffsL;
    fs["cameraMatrixR"] >> cameraMatrixR;
    fs["distCoeffsR"] >> distCoeffsR;
    fs["R"] >> R;
    fs["T"] >> T;
    fs["R1"] >> R1;
    fs["R2"] >> R2;
    fs["P1"] >> P1;
    fs["P2"] >> P2;
    fs["Q"] >> Q;
    fs.release();
}

int main(int argc, char **argv)
{
    // Load class names and YOLOv4-tiny model
    vector<string> classes;
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line))
        classes.push_back(line);

    String modelConfiguration = "yolov4-tiny.cfg";
    String modelWeights = "yolov4-tiny.weights";
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    // Open video capture for two cameras
    VideoCapture capL("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
    VideoCapture capR("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");

    if (!capL.isOpened() || !capR.isOpened())
    {
        cerr << "Error: Could not open cameras." << endl;
        return -1;
    }

    namedWindow("Object Detection Left", WINDOW_AUTOSIZE);
    namedWindow("Object Detection Right", WINDOW_AUTOSIZE);

    initStereoCalibration();

    while (true)
    {
        Mat frameL, frameR;
        capL >> frameL;
        capR >> frameR;
        if (frameL.empty() || frameR.empty())
        {
            cerr << "Error: Could not read frame." << endl;
            break;
        }

        Mat blobL, blobR;
        blobFromImage(frameL, blobL, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
        blobFromImage(frameR, blobR, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
        net.setInput(blobL);
        vector<Mat> outsL;
        net.forward(outsL, getOutputNames(net));
        net.setInput(blobR);
        vector<Mat> outsR;
        net.forward(outsR, getOutputNames(net));

        float confThreshold = 0.5;
        vector<int> classIdsL, classIdsR;
        vector<float> confidencesL, confidencesR;
        vector<Rect> boxesL, boxesR;

        for (size_t i = 0; i < outsL.size(); ++i)
        {
            float *data = (float *)outsL[i].data;
            for (int j = 0; j < outsL[i].rows; ++j, data += outsL[i].cols)
            {
                Mat scores = outsL[i].row(j).colRange(5, outsL[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frameL.cols);
                    int centerY = (int)(data[1] * frameL.rows);
                    int width = (int)(data[2] * frameL.cols);
                    int height = (int)(data[3] * frameL.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIdsL.push_back(classIdPoint.x);
                    confidencesL.push_back((float)confidence);
                    boxesL.push_back(Rect(left, top, width, height));
                }
            }
        }

        for (size_t i = 0; i < outsR.size(); ++i)
        {
            float *data = (float *)outsR[i].data;
            for (int j = 0; j < outsR[i].rows; ++j, data += outsR[i].cols)
            {
                Mat scores = outsR[i].row(j).colRange(5, outsR[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frameR.cols);
                    int centerY = (int)(data[1] * frameR.rows);
                    int width = (int)(data[2] * frameR.cols);
                    int height = (int)(data[3] * frameR.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIdsR.push_back(classIdPoint.x);
                    confidencesR.push_back((float)confidence);
                    boxesR.push_back(Rect(left, top, width, height));
                }
            }
        }

        // Perform non-maxima suppression to eliminate redundant overlapping boxes with lower confidences
        vector<int> indicesL;
        vector<int> indicesR;
        NMSBoxes(boxesL, confidencesL, confThreshold, 0.4, indicesL);
        NMSBoxes(boxesR, confidencesR, confThreshold, 0.4, indicesR);

        for (int idx : indicesL)
        {
            Rect box = boxesL[idx];
            drawPred(classIdsL[idx], confidencesL[idx], box.x, box.y, box.x + box.width, box.y + box.height, frameL, classes);
        }

        for (int idx : indicesR)
        {
            Rect box = boxesR[idx];
            drawPred(classIdsR[idx], confidencesR[idx], box.x, box.y, box.x + box.width, box.y + box.height, frameR, classes);
        }

        imshow("Object Detection Left", frameL);
        imshow("Object Detection Right", frameR);

        if (waitKey(1) >= 0)
            break;
    }

    capL.release();
    capR.release();
    destroyAllWindows();
    return 0;
}
