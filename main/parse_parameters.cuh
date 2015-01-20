#ifndef PARSE_PARAMETERS_H
#define PARSE_PARAMETERS_H

#include <string>
#include <fstream>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <iostream>
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

class Parameters{
public:
	void parse(string);
	int getOutputRows(){return block_y*thread_y;}
	int getOutputColumns(){return block_x*thread_x;}

private:
	string image_name;

	float roomsize_x;
	float roomsize_y;
	float roomsize_z;

	int thread_x;
	int thread_y;
	int thread_z;
	int block_x;
	int block_y;

	float ratio_x;
	float ratio_y;
	float ratio_z;

	int rows;
	int columns;

	/// camera info
	int pnbr;
	int tnbr;
	int cam_nbr;
	vector<float*> P;
	float* T;

	/// localization
	int gaussian_kernel;
	int threshold_img;

public:
	Parameters(): pnbr(12), tnbr(16), cam_nbr(4){
		for(int i = 0; i < cam_nbr; ++i){
			P.push_back(new float[pnbr]);
		}
		T = new float[tnbr];
	}
	Parameters(const Parameters& _p){
		/// additional
		image_name = _p.image_name;
		gaussian_kernel = _p.gaussian_kernel;
		threshold_img = _p.threshold_img;

		//room info
		roomsize_x = _p.roomsize_x;
		roomsize_y = _p.roomsize_y;
		roomsize_z = _p.roomsize_z;

		// virtual world info
		thread_x = _p.thread_x;
		thread_y = _p.thread_y;
		thread_z = _p.thread_z;
		block_x = _p.block_x;
		block_y = _p.block_y;
		ratio_x = _p.ratio_x;
		ratio_y = _p.ratio_y;
		ratio_z = _p.ratio_z;

		//camera info
		rows = _p.rows;
		columns = _p.columns;
		pnbr = _p.pnbr;
		tnbr = _p.tnbr;
		cam_nbr = _p.cam_nbr;

		for(int i = 0; i < cam_nbr; ++i){
			P.push_back(new float[pnbr]);
			for(int k = 0; k < pnbr; ++k){
				P[i][k] = _p.P[i][k];
			}
		}

		T = new float[tnbr];
		for(int k = 0; k < tnbr; ++k){
			T[k] = _p.T[k];
		}
	}
	~Parameters(){
		for(int i = 0; i < cam_nbr; ++i){
			delete[] P[i];
		}
		P.clear();
		delete T;
	}

	friend class Data;
	friend class VisualHull;
};

#endif