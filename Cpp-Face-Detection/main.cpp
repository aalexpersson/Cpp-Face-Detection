#include <opencv2/objdetect/objdetect.hpp>
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

    string trained_classifier_location = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";//Defining the location our XML Trained Classifier in a string
    CascadeClassifier faceDetector;//Declaring an object named 'face detector' of CascadeClassifier class//
    faceDetector.load(trained_classifier_location);//loading the XML trained classifier in the object//
    vector<Rect>faces;

    while (true) {

        Mat frame;
        // Capture frame-by-frame
        cam >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        faceDetector.detectMultiScale(frame, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(70, 70));//Detecting the faces in 'image_with_humanfaces' matrix//
        cam.read(frame);// reading frames from camera and loading them in 'frame' Matrix//
        for (int i = 0; i < faces.size(); i++) { //for locating the face
            Mat faceROI = frame(faces[i]);//Storing face in the matrix//
            int x = faces[i].x;//Getting the initial row value of face rectangle's starting point//
            int y = faces[i].y;//Getting the initial column value of face rectangle's starting point//
            int h = y + faces[i].height;//Calculating the height of the rectangle//
            int w = x + faces[i].width;//Calculating the width of the rectangle//
            rectangle(frame, Point(x, y), Point(w, h), Scalar(255, 0, 255), 2, 8, 0);//Drawing a rectangle using around the faces//
        }

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