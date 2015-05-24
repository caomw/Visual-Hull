#ifndef __DISPLAY__
#define __DISPLAY__

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <cstdio>
#include "parse_parameters.cuh"

using namespace std;
using namespace cv;

class Data{
private:
	/// basic
	int cam_nbr;
	string image_name;
	char temp[200];

public:
	Data(Parameters para): 
		cam_nbr(para.cam_nbr),
		image_name(para.image_name){}

	vector<Mat> getBWImageVector(int k, string param){	
		vector<Mat> bw_imgs;
		Mat img;
		for(int i = 0; i < cam_nbr; ++i){
			sprintf(temp, image_name.c_str(), i, k);
			img = imread(temp, 0);
			if(img.empty()) {cerr<<"There are no images ... ...\n"; exit(1);}
			bw_imgs.push_back(img);
		}
		return bw_imgs;
	}
};

#endif