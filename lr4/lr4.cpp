#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stack>

using namespace cv;
using namespace std;

Mat computeGradient(const Mat& gray, Mat& Gx, Mat& Gy) {
    Mat gx, gy;
    Mat kx = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat ky = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    filter2D(gray, gx, CV_32F, kx);
    filter2D(gray, gy, CV_32F, ky);
    Gx = gx;
    Gy = gy;
    Mat magnitude;
    magnitude.create(gray.size(), CV_32F);
    for (int y = 0; y < gray.rows; ++y) {
        const float* px = gx.ptr<float>(y);
        const float* py = gy.ptr<float>(y);
        float* pm = magnitude.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x) {
            pm[x] = std::hypot(px[x], py[x]);
        }
    }
    return magnitude;
}

Mat computeAngle(const Mat& Gx, const Mat& Gy) {
    Mat angle(Gx.size(), CV_32F);
    for (int y = 0; y < Gx.rows; ++y) {
        const float* px = Gx.ptr<float>(y);
        const float* py = Gy.ptr<float>(y);
        float* pa = angle.ptr<float>(y);
        for (int x = 0; x < Gx.cols; ++x) {
            pa[x] = atan2(py[x], px[x]) * 180.0f / CV_PI;
            if (pa[x] < 0) pa[x] += 360.0f;
        }
    }
    return angle;
}

Mat nonMaximumSuppression(const Mat& magnitude, const Mat& angle) {
    Mat nms = Mat::zeros(magnitude.size(), CV_32F);
    int rows = magnitude.rows, cols = magnitude.cols;
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            float ang = angle.at<float>(y, x);
            float mag = magnitude.at<float>(y, x);
            float m1 = 0.0f, m2 = 0.0f;

            if ((ang >= 0 && ang < 22.5) || (ang >= 337.5 && ang <= 360) || (ang >= 157.5 && ang < 202.5)) {
                m1 = magnitude.at<float>(y, x - 1);
                m2 = magnitude.at<float>(y, x + 1);
            }
            else if ((ang >= 22.5 && ang < 67.5) || (ang >= 202.5 && ang < 247.5)) {
                m1 = magnitude.at<float>(y - 1, x + 1);
                m2 = magnitude.at<float>(y + 1, x - 1);
            }
            else if ((ang >= 67.5 && ang < 112.5) || (ang >= 247.5 && ang < 292.5)) {
                m1 = magnitude.at<float>(y - 1, x);
                m2 = magnitude.at<float>(y + 1, x);
            }
            else {
                m1 = magnitude.at<float>(y - 1, x - 1);
                m2 = magnitude.at<float>(y + 1, x + 1);
            }

            if (mag >= m1 && mag >= m2) nms.at<float>(y, x) = mag;
            else nms.at<float>(y, x) = 0.0f;
        }
    }
    return nms;
}

Mat doubleThresholdAndHysteresis(const Mat& nms, float lowRatio = 0.5f, float highRatio = 0.2f) {
    double maxVal;
    minMaxLoc(nms, nullptr, &maxVal);
    float high = static_cast<float>(maxVal * highRatio);
    float low = high * lowRatio;

    Mat res = Mat::zeros(nms.size(), CV_8U);
    const uchar STRONG = 255;
    const uchar WEAK = 75;

    int rows = nms.rows, cols = nms.cols;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float v = nms.at<float>(y, x);
            if (v >= high) res.at<uchar>(y, x) = STRONG;
            else if (v >= low) res.at<uchar>(y, x) = WEAK;
            else res.at<uchar>(y, x) = 0;
        }
    }

    std::stack<Point> st;
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            if (res.at<uchar>(y, x) == STRONG) st.push(Point(x, y));
        }
    }

    while (!st.empty()) {
        Point p = st.top(); st.pop();
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = p.x + dx, ny = p.y + dy;
                if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
                if (res.at<uchar>(ny, nx) == WEAK) {
                    res.at<uchar>(ny, nx) = STRONG;
                    st.push(Point(nx, ny));
                }
            }
        }
    }

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            if (res.at<uchar>(y, x) != STRONG) res.at<uchar>(y, x) = 0;

    return res;
}

int main(int argc, char** argv) {
    string input;
    if (argc > 1) input = argv[1];
    VideoCapture cap;
    bool useCamera = false;

    if (input.empty()) {
        cap.open(0);
        if (!cap.isOpened()) {
            cout << "Cannot open camera. Or provide image path as argument." << endl;
            return -1;
        }
        useCamera = true;
    }

    Mat frame;
    if (!useCamera) {
        Mat img = imread(input, IMREAD_COLOR);
        if (img.empty()) {
            cout << "Failed to read image: " << input << endl;
            return -1;
        }
        frame = img;
    }

    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Blurred", WINDOW_NORMAL);
    namedWindow("Gradient Magnitude (norm)", WINDOW_NORMAL);
    namedWindow("Non-Maximum Suppression", WINDOW_NORMAL);
    namedWindow("Manual Canny", WINDOW_NORMAL);
    namedWindow("OpenCV Canny", WINDOW_NORMAL);

    while (true) {
        if (useCamera) {
            if (!cap.read(frame)) break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        Mat blurred;
        GaussianBlur(gray, blurred, Size(5, 5), 1.4);

        Mat Gx, Gy;
        Mat magnitude = computeGradient(blurred, Gx, Gy);
        Mat angle = computeAngle(Gx, Gy);
        Mat nms = nonMaximumSuppression(magnitude, angle);

        Mat manualEdges = doubleThresholdAndHysteresis(nms, 0.5f, 0.2f);

        Mat magnitudeDisplay;
        normalize(magnitude, magnitudeDisplay, 0, 255, NORM_MINMAX, CV_8U);

        Mat nmsDisplay;
        normalize(nms, nmsDisplay, 0, 255, NORM_MINMAX, CV_8U);

        Mat cvCanny;
        double t1 = 100.0, t2 = 200.0;
        Canny(gray, cvCanny, t1, t2);

        imshow("Original", frame);
        imshow("Blurred", blurred);
        imshow("Gradient Magnitude (norm)", magnitudeDisplay);
        imshow("Non-Maximum Suppression", nmsDisplay);
        imshow("Manual Canny", manualEdges);
        imshow("OpenCV Canny", cvCanny);

        char key = (char)waitKey(useCamera ? 1 : 0);
        if (key == 27) break; // ESC
        if (!useCamera && key == 'q') break;
    }

    destroyAllWindows();
    if (useCamera) cap.release();
    return 0;
}
