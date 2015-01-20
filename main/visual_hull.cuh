#ifndef VISUAL_HULL_H
#define VISUAL_HULL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "parse_parameters.cuh"

using namespace std;
using namespace cv;

class VisualHull{
public:
	cudaError_t init(Parameters);
	cudaError_t construct(vector<Mat>, Parameters, Mat&);
	vector<Point2f> locate(Mat);
	vector<Point2f> transform2world(vector<Point2f>);

private:
	/// GPU parameters
	float* gpu_P0;
	float* gpu_P1;
	float* gpu_P2;
	float* gpu_P3;
	float* gpu_T;	
	float* gpu_ratiox;
	float* gpu_ratioy;
	float* gpu_ratioz;
	int* gpu_row;
	int* gpu_column;
	int* gpu_threadz;

	/// GPU images
	uchar* gpu_bw_img0;
	uchar* gpu_bw_img1;
	uchar* gpu_bw_img2;
	uchar* gpu_bw_img3;
	uchar* point_cloud;
	uchar* point_cloud_without_noise;
	uchar* ground_plane;

	/// localization
	int gaussian_kernel;
	int threshold_img;
	float ratio_x;
	float ratio_y;
	float* T;

	cudaError_t cudaStatus;

public:
	VisualHull(Parameters _p): gaussian_kernel(_p.gaussian_kernel), threshold_img(_p.threshold_img),
	ratio_x(_p.ratio_x), ratio_y(_p.ratio_y){
		T = new float[_p.tnbr];
		for(int i = 0; i < _p.tnbr; ++i) T[i] = _p.T[i];
	}
	~VisualHull(){
		delete[] T;
		cudaFree(gpu_bw_img0);
		cudaFree(gpu_P0);

		cudaFree(gpu_bw_img1);
		cudaFree(gpu_P1);

		cudaFree(gpu_bw_img2);
		cudaFree(gpu_P2);

		cudaFree(gpu_bw_img3);
		cudaFree(gpu_P3);

		cudaFree(gpu_ratiox);
		cudaFree(gpu_ratioy);
		cudaFree(gpu_ratioz);
		cudaFree(gpu_T);
		cudaFree(gpu_threadz);
		cudaFree(gpu_row);
		cudaFree(gpu_column);
	}
};

#endif