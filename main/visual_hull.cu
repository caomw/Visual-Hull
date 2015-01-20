#include "visual_hull.cuh"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

/// 3d mapping
__global__ void mapping(uchar* point_cloud, 
	const uchar* img0, const uchar* img1, const uchar* img2, const uchar* img3, 
	const float* T,
	const float* P0, const float* P1, const float* P2, const float* P3,
	const float* ratio_x, const float* ratio_y, const float* ratio_z,
	const int* img_row, const int* img_column){
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		int idy = blockIdx.y*blockDim.y + threadIdx.y;
		int idz = blockIdx.z*blockDim.z + threadIdx.z;
		int rows_m = blockDim.y*gridDim.y;
		int columns_m = blockDim.x*gridDim.x;
		int index = idz*rows_m*columns_m + idy*columns_m + idx;

		//grid to coordinates
		float mx = ratio_x[0]*idx;
		float my = ratio_y[0]*idy;
		float mz = ratio_z[0]*idz;

		//world coordinates
		float x = T[0]*mx + T[1]*my + T[2]*mz + T[3];
		float y = T[4]*mx + T[5]*my + T[6]*mz + T[7];
		float z = mz;

		//image coordinates
		float u[4], v[4], norm[4];
		u[0] = P0[0]*x + P0[1]*y + P0[2]*z + P0[3];
		v[0] = P0[4]*x + P0[5]*y + P0[6]*z + P0[7];
		norm[0] = P0[8]*x + P0[9]*y + P0[10]*z + P0[11];

		u[1] = P1[0]*x + P1[1]*y + P1[2]*z + P1[3];
		v[1] = P1[4]*x + P1[5]*y + P1[6]*z + P1[7];
		norm[1] = P1[8]*x + P1[9]*y + P1[10]*z + P1[11];

		u[2] = P2[0]*x + P2[1]*y + P2[2]*z + P2[3];
		v[2] = P2[4]*x + P2[5]*y + P2[6]*z + P2[7];
		norm[2] = P2[8]*x + P2[9]*y + P2[10]*z + P2[11];

		u[3] = P3[0]*x + P3[1]*y + P3[2]*z + P3[3];
		v[3] = P3[4]*x + P3[5]*y + P3[6]*z + P3[7];
		norm[3] = P3[8]*x + P3[9]*y + P3[10]*z + P3[11];

		int u00 = u[0]/norm[0];
		int u11 = u[1]/norm[1];
		int u22 = u[2]/norm[2];
		int u33 = u[3]/norm[3];

		int v00 = v[0]/norm[0];
		int v11 = v[1]/norm[1];
		int v22 = v[2]/norm[2];
		int v33 = v[3]/norm[3];

		int sum = 0;
		int final_width = img_column[0];
		int final_height = img_row[0];

		//decide the point cloud
		if((u00>0)&&(u00<final_width)&&(v00>0)&&(v00<final_height)){
			if(img0[v00*final_width + u00] != 0 )
				++sum;
		}

		if((u11>0)&&(u11<final_width)&&(v11>0)&&(v11<final_height)){
			if(img1[v11*final_width + u11] != 0 )
				++sum;
		}

		if((u22>0)&&(u22<final_width)&&(v22>0)&&(v22<final_height)){
			if(img2[v22*final_width + u22] != 0 )
				++sum;
		}

		if((u33>0)&&(u33<final_width)&&(v33>0)&&(v33<final_height)){
			if(img3[v33*final_width + u33] != 0 )
				++sum;
		}
		point_cloud[index] = 0;
		if(sum >= 3)
			point_cloud[index] = 255;

}

__global__ void denoise(uchar* new_cloud, uchar* old_cloud){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int rows_m = blockDim.y*gridDim.y;
	int columns_m = blockDim.x*gridDim.x;
	int height_m = blockDim.z;// TODO
	int index = idz*rows_m*columns_m + idy*columns_m + idx;

	if( (idz-1 >= 0) && (idz+1 < height_m) ){
		int neighbor1 = (idz-1)*rows_m*columns_m + idy*columns_m + idx;
		int neighbor2 = (idz+1)*rows_m*columns_m + idy*columns_m + idx;

		if(old_cloud[neighbor1] && old_cloud[neighbor2]){
			new_cloud[index] = 255;
		}
		else{
			new_cloud[index] = 0;
		}
	}
}

/// point_cloud 2 image
__global__ void point_cloud2image(uchar* image, const uchar* point_cloud, const int* gpu_thread_z){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int rows = gridDim.y*blockDim.y;
	int columns = gridDim.x*blockDim.x;
	int index = idy*columns + idx;

	image[index] = 0;
	int temp_z = gpu_thread_z[0];
	for(int i = 0; i < temp_z; ++i)
		image[index] += point_cloud[i*rows*columns + index]/255;
}

cudaError_t VisualHull:: init(Parameters _para){
	int rows = _para.rows;
	int columns = _para.columns;
	int pnbr = _para.pnbr;
	int tnbr = _para.tnbr;

	/// malloc GPU memory and copy parameters
	/// P[0...3]
	cudaStatus = cudaMalloc((void**)&gpu_P0, pnbr*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_P0, _para.P[0], pnbr*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_P1, pnbr*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_P1, _para.P[1], pnbr*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_P2, pnbr*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_P2, _para.P[2], pnbr*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_P3, pnbr*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_P3, _para.P[3], pnbr*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/// T
	cudaStatus = cudaMalloc((void**)&gpu_T, tnbr*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_T, _para.T, tnbr*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/// thread_z
	cudaStatus = cudaMalloc((void**)&gpu_threadz, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_threadz, &_para.thread_z, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/// ratio_x/y/z
	cudaStatus = cudaMalloc((void**)&gpu_ratiox, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_ratioy, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_ratioz, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_ratiox, &_para.ratio_x, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_ratioy, &_para.ratio_y, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_ratioz, &_para.ratio_z, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/// input binary images
	cudaStatus = cudaMalloc((void**)&gpu_bw_img0, rows*columns*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_bw_img1, rows*columns*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_bw_img2, rows*columns*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_bw_img3, rows*columns*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	/// point cloud, grand plane
	int size1 = _para.block_x*_para.block_y*_para.thread_x*_para.thread_y;
	int size2 = _para.thread_z*size1;
	cudaStatus = cudaMalloc((void**)&point_cloud, size2*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&point_cloud_without_noise, size2*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&ground_plane, size1*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	/// row, column
	cudaStatus = cudaMalloc((void**)&gpu_row, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_column, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_row, &_para.rows, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_column, &_para.columns, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	return cudaStatus;
}

cudaError_t VisualHull:: construct(vector<Mat> masks, Parameters para, Mat& output){
	int rows = para.rows;
	int columns = para.columns;
	int cam_nbr = para.cam_nbr;

	cudaStatus = cudaMemcpy(gpu_bw_img0, masks[0].data, rows*columns*sizeof(uchar), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_bw_img1, masks[1].data, rows*columns*sizeof(uchar), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_bw_img2, masks[2].data, rows*columns*sizeof(uchar), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gpu_bw_img3, masks[3].data, rows*columns*sizeof(uchar), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/// CUDA configuration
	dim3 grid(para.block_x, para.block_y, 1);
	dim3 block_3d(para.thread_x, para.thread_y, para.thread_z);
	dim3 block_2d(para.thread_x, para.thread_y, 1);

	/// mapping
	mapping<<<grid, block_3d>>>(point_cloud, 
		gpu_bw_img0, gpu_bw_img1, gpu_bw_img2, gpu_bw_img3, 
		gpu_T, 
		gpu_P0, gpu_P1, gpu_P2, gpu_P3, 
		gpu_ratiox, gpu_ratioy, gpu_ratioz,
		gpu_row, gpu_column);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	denoise<<<grid, block_3d>>>(point_cloud_without_noise, point_cloud);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	point_cloud2image<<<grid, block_2d>>>(ground_plane, point_cloud_without_noise, gpu_threadz);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(output.data, ground_plane, 
		para.block_x*para.thread_x*para.block_y*para.thread_y*sizeof(uchar), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

Error:
	return cudaStatus;
}

vector<Point2f> VisualHull::locate(Mat output){
	GaussianBlur(output, output, Size(gaussian_kernel, gaussian_kernel), 0, 0); // note, kernel size should be odd
	threshold(output, output, threshold_img, 255, THRESH_BINARY);
	GaussianBlur(output, output, Size(gaussian_kernel, gaussian_kernel), 0, 0); // note, kernel size should be odd

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat temp = output.clone();
	findContours(temp, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<Point2f> positions;
	for( int idx = 0; idx < contours.size(); ++idx){
		Moments m = moments(contours[idx], false);
		Point2f mc = Point2f(m.m10/m.m00 , m.m01/m.m00);
		positions.push_back(mc);
	}

	return positions;
}

vector<Point2f> VisualHull::transform2world(vector<Point2f> vpositions){
	vector<Point2f> wpositions;
	float mx, my, mz = 0;
	float x, y, z;
	for(int i = 0; i < vpositions.size(); ++i){
		mx = vpositions[i].x*ratio_x;
		my = vpositions[i].y*ratio_y;
		mz = 0;
		x = T[0]*mx + T[1]*my + T[2]*mz + T[3];
		y = T[4]*mx + T[5]*my + T[6]*mz + T[7];
		z = mz;
		wpositions.push_back(Point2f(x, y));
	}
	return wpositions;
}