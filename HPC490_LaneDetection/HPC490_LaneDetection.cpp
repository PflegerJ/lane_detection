// HPC490_LaneDetection.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void drawTwoCirclesDemo();
void videoPlaybackDemo();

int main()
{
    cout << "Press any key to end a demo.\n";

	videoPlaybackDemo();

	drawTwoCirclesDemo();

	return 0;

}

void drawTwoCirclesDemo()
{
	Mat image = Mat::zeros(300, 600, CV_8UC3);
	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
	imshow("Two Cool Circles!", image);
	waitKey(0);
}

void videoPlaybackDemo()
{
	Mat frame;
	VideoCapture cap;
	cap.open("laneTest1.mp4");

	// Check if we succeeded
	if (!cap.isOpened()) 
	{
		cerr << "ERROR! Unable to open camera\n";
		return;
	}

	cout << "Start grabbing" << endl << "Press any key to terminate" << endl;
	
	for (;;)
	{
		// Wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// Check if we succeeded
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		// Show live and wait for a key with timeout long enough to show images
		imshow("Live", frame);
		if (waitKey(33) >= 0)  // 30 fps
			break;
	}

	// The camera is deinitialized automatically in VideoCapture destructor
}
