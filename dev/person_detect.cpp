#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

using namespace cv;
using namespace std;

// Class representing a CSI Camera
class CSI_Camera {
public:
    VideoCapture cap; // VideoCapture object

    // Constructor to initialize the camera pipeline
    CSI_Camera(const std::string& pipeline) {
        cap.open(pipeline, CAP_GSTREAMER); // Open camera with GStreamer pipeline
        if (!cap.isOpened()) {
            cerr << "Error: Could not open camera." << endl;
        }
    }

    // Check if the camera is opened
    bool isOpened() const {
        return cap.isOpened();
    }

    // Read frame from camera
    bool read(Mat& frame) {
        return cap.read(frame); // Read frame
    }

    // Release camera resources
    void release() {
        if (cap.isOpened()) {
            cap.release();
        }
    }

    // Destructor to release camera resources
    ~CSI_Camera() {
        release();
    }
};

// Function to process frames from a camera
void process_frame(VideoCapture& cap, CascadeClassifier& body_cascade, vector<pair<Mat, vector<Rect>>>& frame_queue, mutex& mtx, bool& running) {
    while (running) {
        Mat frame, gray;
        if (!cap.read(frame)) {
            cerr << "Error: Could not read frame." << endl;
            break;
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY); // Convert frame to grayscale
        vector<Rect> bodies;
        body_cascade.detectMultiScale(gray, bodies, 1.3, 5); // Detect bodies in frame

        mtx.lock(); // Lock mutex before accessing shared data
        frame_queue.emplace_back(frame.clone(), bodies); // Store frame and detected bodies
        if (frame_queue.size() > 2) {
            frame_queue.erase(frame_queue.begin()); // Keep only latest frames
        }
        mtx.unlock(); // Unlock mutex after accessing shared data
    }
}

// Function for person detection using two synchronized cameras
void person_detect() {
    // Load upper body cascade classifier
    String cascade_name = "/usr/share/opencv4/haarcascades/haarcascade_upperbody.xml";
    CascadeClassifier body_cascade;
    if (!body_cascade.load(cascade_name)) {
        cerr << "Error: Could not load cascade classifier." << endl;
        return;
    }

    // Create CSI_Camera objects for two cameras with GStreamer pipelines
    CSI_Camera camera0("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
    CSI_Camera camera1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");

    // Check if cameras are opened successfully
    if (!camera0.isOpened() || !camera1.isOpened()) {
        cerr << "Error: Could not open CSI cameras." << endl;
        return;
    }

    // Vector to store frames and detected bodies for each camera
    vector<pair<Mat, vector<Rect>>> frame_queue0, frame_queue1;
    mutex mtx0, mtx1; // Mutexes for synchronization

    bool running = true; // Flag to indicate if threads are running

    // Threads for processing frames from each camera
    thread thread0(process_frame, std::ref(camera0.cap), std::ref(body_cascade), std::ref(frame_queue0), std::ref(mtx0), std::ref(running));
    thread thread1(process_frame, std::ref(camera1.cap), std::ref(body_cascade), std::ref(frame_queue1), std::ref(mtx1), std::ref(running));

    // Create windows for displaying frames
    namedWindow("Person Detection_Left", WINDOW_AUTOSIZE);
    namedWindow("Person Detection_Right", WINDOW_AUTOSIZE);

    try {
        while (true) {
            mtx0.lock(); // Lock mutex before accessing frame_queue0
            mtx1.lock(); // Lock mutex before accessing frame_queue1

            // Check if both frame queues have frames
            if (!frame_queue0.empty() && !frame_queue1.empty()) {
                // Clone latest frames from each queue
                Mat frame0 = frame_queue0.back().first.clone();
                Mat frame1 = frame_queue1.back().first.clone();

                // Draw rectangles around detected bodies
                vector<Rect> bodies0 = frame_queue0.back().second;
                vector<Rect> bodies1 = frame_queue1.back().second;
                vector<pair<int, int>> pairs;

                for (const auto& rect : bodies0) {
                    pairs.push_back({rect.x, rect.width});
                    rectangle(frame0, rect, Scalar(255, 0, 0), 2);
                }
                for (const auto& rect : bodies1) {
                    pairs.push_back({rect.x, rect.width});
                    rectangle(frame1, rect, Scalar(255, 0, 0), 2);
                }

                if (pairs.size() > 1 && pairs.size() % 2 == 0) {
                    int x_diff = pairs[1].first - pairs[0].first;
                    int w_diff = pairs[1].second - pairs[0].second;
                    int x_loc = x_diff / 2 + pairs[1].first;
                    int depth = int((270.0 / x_diff) * 10); // in 1/10 centimeters
                    int x_mean = x_diff / 2 + pairs[1].first;
                    int x_offset = x_mean - frame0.cols / 2;
                    int angle = int(x_offset / 8);
                    cout << "Angle: " << angle << endl;
                    cout << "Distance: " << depth * 10 << " mm" << endl;
                }

                // Display frames in respective windows
                imshow("Person Detection_Left", frame0);
                imshow("Person Detection_Right", frame1);
            }

            mtx0.unlock(); // Unlock mutex after accessing frame_queue0
            mtx1.unlock(); // Unlock mutex after accessing frame_queue1

            // Exit loop if ESC key is pressed
            if (waitKey(30) == 27) {
                break;
            }
        }
    } catch (...) {
        cerr << "Exception occurred during execution." << endl;
    }

    // Stop threads and release resources
    running = false;
    camera0.release();
    camera1.release();
    destroyAllWindows();
    thread0.join();
    thread1.join();
}

// Main function
int main() {
    person_detect(); // Call person_detect function
    return 0; // Return 0 to indicate successful execution
}

