// dual_csi_camera.cpp
// MIT License
// Copyright (c) 2019-2022 JetsonHacks
// Using two CSI cameras (such as the Raspberry Pi Version 2) connected to a 
// NVIDIA Jetson Nano Developer Kit with two CSI ports (Jetson Nano, Jetson Xavier NX) via OpenCV
// Drivers for the camera and OpenCV are included in the base image

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <iostream>

class CSI_Camera {
public:
    CSI_Camera() : grabbed(false), running(false) {}

    bool open(const std::string &gstreamer_pipeline_string) {
        video_capture.open(gstreamer_pipeline_string, cv::CAP_GSTREAMER);
        if (video_capture.isOpened()) {
            video_capture.read(frame);
            grabbed = !frame.empty();
            return true;
        } else {
            std::cerr << "Unable to open camera" << std::endl;
            std::cerr << "Pipeline: " << gstreamer_pipeline_string << std::endl;
            return false;
        }
    }

    void start() {
        if (running) {
            std::cout << "Video capturing is already running" << std::endl;
            return;
        }
        running = true;
        read_thread = std::thread(&CSI_Camera::updateCamera, this);
    }

    void stop() {
        running = false;
        if (read_thread.joinable()) {
            read_thread.join();
        }
    }

    void release() {
        if (video_capture.isOpened()) {
            video_capture.release();
        }
        if (read_thread.joinable()) {
            read_thread.join();
        }
    }

    bool read(cv::Mat &output_frame) {
        std::lock_guard<std::mutex> lock(read_lock);
        if (grabbed) {
            output_frame = frame.clone();
        }
        return grabbed;
    }

private:
    void updateCamera() {
        while (running) {
            cv::Mat new_frame;
            bool new_grabbed = video_capture.read(new_frame);
            std::lock_guard<std::mutex> lock(read_lock);
            grabbed = new_grabbed;
            if (grabbed) {
                frame = new_frame;
            }
        }
    }

    cv::VideoCapture video_capture;
    cv::Mat frame;
    bool grabbed;
    std::thread read_thread;
    std::mutex read_lock;
    bool running;
};

std::string gstreamer_pipeline(int sensor_id, int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    std::string pipeline = "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) + " ! "
           "video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" + std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! "
           "video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" + std::to_string(display_height) + ", format=(string)BGRx ! "
           "videoconvert ! video/x-raw, format=(string)BGR ! appsink";
    
    std::cout << "GStreamer pipeline: " << pipeline << std::endl;
    return pipeline;
}

void run_cameras() {
    const std::string window_title = "Dual CSI Cameras";
    CSI_Camera left_camera, right_camera;

    if (left_camera.open(gstreamer_pipeline(0, 1920, 1080, 960, 540, 30, 0)) &&
        right_camera.open(gstreamer_pipeline(1, 1920, 1080, 960, 540, 30, 0))) {

        left_camera.start();
        right_camera.start();

        cv::namedWindow(window_title, cv::WINDOW_AUTOSIZE);

        while (true) {
            cv::Mat left_image, right_image;
            if (left_camera.read(left_image) && right_camera.read(right_image)) {
                cv::Mat combined_image;
                cv::hconcat(left_image, right_image, combined_image);

                if (cv::getWindowProperty(window_title, cv::WND_PROP_AUTOSIZE) >= 0) {
                    cv::imshow(window_title, combined_image);
                } else {
                    break;
                }

                int keycode = cv::waitKey(30) & 0xFF;
                if (keycode == 27) {
                    break;
                }
            } else {
                std::cerr << "Error: Unable to read from one of the cameras" << std::endl;
                break;
            }
        }

        left_camera.stop();
        right_camera.stop();
        left_camera.release();
        right_camera.release();
        cv::destroyAllWindows();
    } else {
        std::cerr << "Error: Unable to open both cameras" << std::endl;
        left_camera.stop();
        right_camera.stop();
        left_camera.release();
        right_camera.release();
    }
}

int main() {
    run_cameras();
    return 0;
}

