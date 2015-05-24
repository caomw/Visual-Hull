/// Using 4 cameras to locate the positions of people
/// @input: binary image sequences from 4 views
/// @output: positions of moving people

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include "parse_parameters.cuh"
#include "visual_hull.cuh"
#include "data.cuh"

using namespace std;
using namespace cv;

/// true: store but not display
/// false: display but not store
bool store = true;

int main(void){
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	/// 1. parameters
	string para_file_name = "parameters_Fview2_small.dat";
	Parameters para;
	para.parse(para_file_name);

	/// 2. io
	Data data(para);
	Mat output = Mat:: zeros(para.getOutputRows(), para.getOutputColumns(), CV_8UC1);

	/// 3. visual hull
	VisualHull visual_hull(para);
	cudaStatus = visual_hull.init(para);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "visual hull initialization error\n");
		return 1;
	}

	int k = 0; /// start from the 2nd frame
	if(store == false) namedWindow("color", 0);
	ofstream out("wpositions.txt");
	for(;;){
		cout<<"processing "<<k<<"th frames... ..."<<endl;

		/// start 3D reconstruction
		vector<Mat> bw_img = data.getBWImageVector(k, para_file_name);
		cudaStatus = visual_hull.construct(bw_img, para, output);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "visual hull construction error\n");
			return 1;
		}

		/// localization
		vector<Point2f> positions = visual_hull.locate(output);

		/// show
		if(store == false){
			Mat color_output;
			cvtColor(255/60*output, color_output, CV_GRAY2BGR);
			for(int i = 0; i < positions.size(); ++i)
				circle(color_output, positions[i], 2, Scalar(0, 0, 255), 2, 8, 0);

			imshow("color", color_output);
			char c = waitKey(1);
			if(c == 'q') break;
		}
		else{
			vector<Point2f> wpositions = visual_hull.transform2world(positions);
			out<<k<<" "<<wpositions.size();
			for(int i = 0; i < wpositions.size(); ++i) out<<" "<<wpositions[i].x<<" "<<wpositions[i].y;
			out<<endl;
		}

		++k;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}