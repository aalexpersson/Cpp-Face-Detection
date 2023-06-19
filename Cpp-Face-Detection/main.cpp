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

    string trained_faceClassifier_location = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";//Defining the location our XML Trained Classifier in a string
    string trained_eyesClassifier_location = "C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

    CascadeClassifier faceDetector, eyesDetector;
    
    //loading the XMLs trained classifier in the objects//
    faceDetector.load(trained_faceClassifier_location);
    eyesDetector.load(trained_eyesClassifier_location);
    
    vector<Rect>faces;

    while (true) {

        Mat frame;
        // Capture frame-by-frame
        cam >> frame;   

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        faceDetector.detectMultiScale(frame, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(100, 100));//Detecting the faces in 'image_with_humanfaces' matrix//
        cam.read(frame);// reading frames from camera and loading them in 'frame' Matrix//
        for (int i = 0; i < faces.size(); i++) { //for locating the face
            Mat faceROI = frame(faces[i]);//Storing face in the matrix//
            int x = faces[i].x;//Getting the initial row value of face rectangle's starting point//
            int y = faces[i].y;//Getting the initial column value of face rectangle's starting point//
            int h = y + faces[i].height;//Calculating the height of the rectangle//
            int w = x + faces[i].width;//Calculating the width of the rectangle//
            rectangle(frame, Point(x, y), Point(w, h), Scalar(255, 0, 255), 2, 8, 0);//Drawing a rectangle using around the faces//
           

            //Eye detection
            vector<Rect> eyes;
            eyesDetector.detectMultiScale(faceROI, eyes);
            for (size_t j = 0; j < eyes.size(); j++)
            {
                Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
            }
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