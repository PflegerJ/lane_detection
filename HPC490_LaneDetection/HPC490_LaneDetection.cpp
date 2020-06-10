// HPC490_LaneDetection.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

void videoPlaybackDemo();
void cannyFrame(Mat);
void CannyThreshold(int, void*, Mat, Mat&);
void hough(Mat);

int main()
{
    cout << "Press any key to end a demo.\n";

	videoPlaybackDemo();

	return 0;

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
		
		cannyFrame(frame);

		if (waitKey(33) >= 0)  // 30 fps
			break;
	}

	// The camera is deinitialized automatically in VideoCapture destructor
}

void cannyFrame(Mat frame)
{
	Mat src_gray;
	Mat dst;

	/// Create a matrix of the same type and size as src (for dst)
	dst.create(frame.size(), frame.type());

	/// Convert the image to grayscale
	cvtColor(frame, src_gray, CV_BGR2GRAY);

	/// Create a window
	//namedWindow("Canny Frame Test", CV_WINDOW_AUTOSIZE);

	/// Show the image
	CannyThreshold(0, 0, src_gray, dst);
}

int lowThreshold = 80;  // The bigger the number, the less edges detected
int const max_lowThreshold = 100;
int iratio = 3;
int kernel_size = 3;

void CannyThreshold(int, void*, Mat input, Mat& output)
{
	Mat detected_edges; 

	/// Reduce noise with a kernel 3x3
	blur(input, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * iratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	output = Scalar::all(0);

	input.copyTo(output, detected_edges);

	hough(output);

	//imshow("Canny Threshold Test", output);

}

void hough(Mat frame)
{
	// Declare the output variables
	Mat output;

	// DST = FRAME
	// CDST = output

	// Check if image is loaded fine
	if (frame.empty()) {
		printf(" Error opening image\n");
		return;
	}

	// Copy edges to the images that will display the results in BGR
	cvtColor(frame, output, COLOR_GRAY2BGR);

	// Standard Hough Line Transform
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(frame, lines, 1, CV_PI / 180, 250, 0, 0); // runs the actual detection

	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(output, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}

	// Show results
	imshow("Hough Line Transform", output);
}