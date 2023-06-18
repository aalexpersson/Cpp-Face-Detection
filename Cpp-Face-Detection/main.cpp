#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cam(0);

    // Check if camera opened successfully
    if (!cam.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while (1) {

        Mat frame;
        // Capture frame-by-frame
        cam >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Display the video capture
        imshow("Video capture", frame);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    // When everything done, release the video capture object
    cam.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}