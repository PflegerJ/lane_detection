


#include <iostream>
#include "cannyEdgeDetector.hpp"
#include "canny.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

void VideoDemo();
void ImageDemo();
void CannyEdgeDetect(Mat, const Mat);
void HoughTransform(Mat, const Mat);

void para();
pixel_t* mat_To_Pixel(Mat sourceImage);
bool isImageDemo = false;

int main()
{
	cout << "Press any key to end a demo.\n";

	//VideoDemo();

	//ImageDemo();


	para();
	return 0;

}

void ImageDemo()
{
	Mat src = imread(samples::findFile("image.jpg"), IMREAD_COLOR);
	isImageDemo = true;
	CannyEdgeDetect(src, src);
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
		// Show live and wait for a key with timeout long enough to show images
		//imshow("Live", frame);

		//CannyEdgeDetect(frame);

		Mat croppedFrame = frame(Rect(0, frame.rows / 2, frame.cols, frame.rows / 2));

		CannyEdgeDetect(croppedFrame, frame);

		if (waitKey(33) >= 0)  // 30 fps
			break;
	}

	// The camera is deinitialized automatically in VideoCapture destructor
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



void para()
{
	Mat src = imread(samples::findFile("yes.jpg"), IMREAD_COLOR);
	//std::shared_ptr<ImgMgr> img_mgr = std::make_shared<ImgMgr>(*argv);

	pixel_t* pixel_array = mat_To_Pixel(src);


	CannyEdgeDetector ced(pixel_array, src.rows, src.cols);
	ced.detect_edges(false);
}


pixel_t* mat_To_Pixel(Mat sourceImage)
{
	int cn = sourceImage.channels();
	pixel_t* pixel_array;

	for (int i = 0; i < sourceImage.rows; i++)
	{
		uint8_t* rowPtr = sourceImage.row(i).data;
		//uint8_t* rowPtr = sourceImage.row(i);

		for (int j = 0; j < sourceImage.cols; j++)
		{
			pixel_array[i + j].red = rowPtr[j * cn + 2]; // R
			pixel_array[i + j].green = rowPtr[j * cn + 1]; // G
			pixel_array[i + j].blue = rowPtr[j * cn + 0]; // B
		}
	}

	return pixel_array;
	/*
	
	// We iterate over all pixels of the image
    for(int r = 0; r < image.rows; r++) {
        // We obtain a pointer to the beginning of row r
        cv::Vec3b* ptr = image.ptr<cv::Vec3b>(r);

        for(int c = 0; c < image.cols; c++) {
            // We invert the blue and red values of the pixel
            ptr[c] = cv::Vec3b(ptr[c][2], ptr[c][1], ptr[c][0]);
        }
    }
	
	*/

}




/*



int main(int argc, char** argv)
{
    /* storage and defaults for command line arguments 
   // struct arguments args;
  //  args.inFile = DEFAULT_INFILE;
  //  args.outFile = DEFAULT_OUTFILE;
 //   args.serial = false;

    /*
    int rc = argp_parse(&argp, argc, argv, 0, 0, &args);
    if (rc) {
        std::cerr << "Failed to parse command line arguments." << std::endl;
        exit(rc);
    }

    if (0 == args.inFile.compare(args.outFile)) {
        std::cerr << "Input and output file names must be different!" << std::endl;
        exit(ED_PARSE_ERR);
    }

    

  //  std::cout << "Canny Edge Detection" << std::endl;
  //  if (true == args.serial) {
  //      std::cout << "Executing serially on CPU" << std::endl;
  //  }
  ///  else {
    //    std::cout << "Executing in parallel on GPU" << std::endl;
 //   }

    /* Instantiate our image manager 
    std::shared_ptr<ImgMgr> img_mgr = std::make_shared<ImgMgr>(*argv);

    /* read input file 
    img_mgr->read_image(args.inFile);

    /* Instantiate our edge detector 
    CannyEdgeDetector ced(img_mgr);

    /* run edge detection algorithm 
    ced.detect_edges(args.serial);

    /* write results 
    img_mgr->write_image(args.outFile);
    std::cout << "Edge detection complete" << std::endl;

  //  return ED_SUCCESS;
}


*/