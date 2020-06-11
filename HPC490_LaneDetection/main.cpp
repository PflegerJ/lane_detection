// HPC490_LaneDetection.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>

#include "edge_detect_cuda.h"

using namespace cv;
using namespace std;

void VideoDemo();
void ImageDemo();
void EdgeDetect(Mat, const Mat);
void CudaEdgeDetect(Mat, const Mat);
void CannyEdgeDetect(Mat, const Mat);
void HoughTransform(Mat, const Mat);

enum EdgeDetection { ed_Canny, ed_CUDA };

const EdgeDetection edgeDetectionMode = EdgeDetection::ed_Canny;  // Adjust this to do either the Canny or CUDA Edge Detection

bool isImageDemo = false;

int main()
{
    cout << "Press any key to end a demo.\n";

	VideoDemo();
	
	ImageDemo();

	return 0;

}

void ImageDemo()
{
	Mat src = imread(samples::findFile("image.jpg"), IMREAD_COLOR);
	isImageDemo = true;

	Mat croppedFrame = src(Rect(0, src.rows / 2, src.cols, src.rows / 2));

	EdgeDetect(croppedFrame, src);
}

void VideoDemo()
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

		Mat croppedFrame = frame(Rect(0, frame.rows / 2, frame.cols, frame.rows / 2));

		EdgeDetect(croppedFrame, frame);

		if (waitKey(33) >= 0)  // 30 fps
			break;
	}

	// The camera is deinitialized automatically in VideoCapture destructor
}

// Run either Canny or CUDA edge detection
void EdgeDetect(Mat frame, const Mat orig)
{
	if (edgeDetectionMode == EdgeDetection::ed_Canny)
	{
		CannyEdgeDetect(frame, orig);
	}
	else
	{
		CudaEdgeDetect(frame, orig);
	}
}

void CudaEdgeDetect(Mat frame, const Mat orig)
{
	pixel_t* orig_pixels = (pixel_t*)frame.data;

	unsigned input_pixel_length = frame.rows * frame.cols;
	int rows = frame.rows;
	int cols = frame.cols;

	Mat output(frame.rows, frame.cols, CV_8UC1);

	cu_detect_edges((pixel_channel_t*)output.data, orig_pixels, rows, cols);

	imshow("CUDA Edge Detection", output);

	HoughTransform(output, orig);
}

void CannyEdgeDetect(Mat frame, const Mat orig)
{
	Mat input_gray;
	Mat output;

	/// Create a matrix of the same type and size as src (for dst)
	output.create(frame.size(), frame.type());

	/// Convert the image to grayscale
	cvtColor(frame, input_gray, CV_BGR2GRAY);

	Mat detected_edges; 

	/// Reduce noise with a kernel 3x3
	blur(input_gray, detected_edges, Size(3, 3));

	int lowThreshold = 80;  // The bigger the number, the less edges detected
	int ratio = 3;
	int kernel_size = 3;

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	output = Scalar::all(0);

	input_gray.copyTo(output, detected_edges);

	imshow("Canny Edge Detection", output);

	HoughTransform(output, orig);
}

void HoughTransform(Mat frame, const Mat orig)
{
	// Declare the output variables
	Mat output;

	// Check if image is loaded fine
	if (frame.empty()) {
		printf(" Error opening image\n");
		return;
	}

	// Copy edges to the images that will display the results in BGR
	cvtColor(frame, output, COLOR_GRAY2BGR);

	// Standard Hough Line Transform
	vector<Vec2f> lines; // will hold the results of the detection
	double rho = 2.5;  // Larger rho = two values might end up in the same bucket = more lines (because more buckets have a large vote count)
	double theta = CV_PI / 180;  // Larger theta = fewer calculations = fewer accumulator columns/buckets = fewer lines found
	int threshold = 250;
	HoughLines(frame, lines, rho, theta, threshold, 0, 0); // runs the actual detection

	Point avgLeftTop;
	Point avgLeftBot;
	Point avgRightTop;
	Point avgRightBot;

	int numLeft = 0;
	int sumLeftTopx = 0;
	int sumLeftTopy = 0;
	int sumLeftBotx = 0;
	int sumLeftBoty = 0;

	int numRight = 0;
	int sumRightTopx = 0;
	int sumRightTopy = 0;
	int sumRightBotx = 0;
	int sumRightBoty = 0;

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

		double slope = (pt2.y - pt1.y) / (double)(pt2.x - pt1.x);

		// Ensure the lines are within a reasonable bound
		if (abs(slope) > 0.5 && abs(slope) < 5)
		{
			if (slope < 0)
			{
				numLeft++;
				sumLeftTopx += pt1.x;
				sumLeftTopy += pt1.y;
				sumLeftBotx += pt2.x;
				sumLeftBoty += pt2.y;
			}
			else
			{
				numRight++;
				sumRightTopx += pt1.x;
				sumRightTopy += pt1.y;
				sumRightBotx += pt2.x;
				sumRightBoty += pt2.y;
			}
		}
	}

	Mat overlay(orig, Rect(0, orig.rows / 2, orig.cols, orig.rows / 2));

	if (numLeft > 0)
	{
		avgLeftTop.x = sumLeftTopx / numLeft;
		avgLeftTop.y = sumLeftTopy / numLeft;
		avgLeftBot.x = sumLeftBotx / numLeft;
		avgLeftBot.y = sumLeftBoty / numLeft;
		line(overlay, avgLeftTop, avgLeftBot, Scalar(0, 0, 255), 10, LINE_AA);
	}

	if (numRight > 0)
	{
		avgRightTop.x = sumRightTopx / numRight;
		avgRightTop.y = sumRightTopy / numRight;
		avgRightBot.x = sumRightBotx / numRight;
		avgRightBot.y = sumRightBoty / numRight;
		line(overlay, avgRightTop, avgRightBot, Scalar(255, 0, 255), 10, LINE_AA);
	}

	// Show results
	imshow("Hough Line Transform", orig);

	if (isImageDemo)
	{
		waitKey(0);
	}
}