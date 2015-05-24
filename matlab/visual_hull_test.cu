
__global__ void mapping(double *point_cloud, 
const double *img1, const double *img2, 
const double *img3, const double *img4, 
const double *T,
const double *P1, const double *P2, const double *P3, const double *P4, 
const double *ratiox, const double *ratioy, const double *ratioz,
const int *img_width, const int *img_height)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = threadIdx.z;
    int width = gridDim.x*blockDim.x;
    int height = gridDim.y*blockDim.y;
    
    int index = idz*width*height + idx*height + idy;
	
	//grid to coordinates
	double mx = idx*ratiox[0];
	double my = idy*ratioy[0];
	double mz = idz*ratioz[0];
	
	
	//world coordinates
	double x = T[0]*mx + T[4]*my + T[8]*mz + T[12];
	double y = T[1]*mx + T[5]*my + T[9]*mz + T[13];
	double z = mz;
	
	
	//image coordinates
	double u1 = P1[0]*x + P1[3]*y + P1[6]*z + P1[9];
	double v1 = P1[1]*x + P1[4]*y + P1[7]*z + P1[10];
	double norm1 = P1[2]*x + P1[5]*y + P1[8]*z + P1[11];
	u1/=norm1;
	v1/=norm1;
	
	double u2 = P2[0]*x + P2[3]*y + P2[6]*z + P2[9];
	double v2 = P2[1]*x + P2[4]*y + P2[7]*z + P2[10];
	double norm2 = P2[2]*x + P2[5]*y + P2[8]*z + P2[11];
	u2/=norm2;
	v2/=norm2;
	
	double u3 = P3[0]*x + P3[3]*y + P3[6]*z + P3[9];
	double v3 = P3[1]*x + P3[4]*y + P3[7]*z + P3[10];
	double norm3 = P3[2]*x + P3[5]*y + P3[8]*z + P3[11];
	u3/=norm3;
	v3/=norm3;
	
	double u4 = P4[0]*x + P4[3]*y + P4[6]*z + P4[9];
	double v4 = P4[1]*x + P4[4]*y + P4[7]*z + P4[10];
	double norm4 = P4[2]*x + P4[5]*y + P4[8]*z + P4[11];
	u4/=norm4;
	v4/=norm4;

	int u11 = (u1);
	int u22 = (u2);
	int u33 = (u3);
	int u44 = (u4);
	
	int v11 = (v1);
	int v22 = (v2);
	int v33 = (v3);
	int v44 = (v4);
	
	int final_width = img_width[0];
	int final_height = img_height[0];
	
	int seen_record[4] = {0};
	
	//decide the point cloud
	if((u11>0)&&(u11<final_width)&&(v11>0)&&(v11<final_height))
		seen_record[0] = 1;
	
	if((u22>0)&&(u22<final_width)&&(v22>0)&&(v22<final_height))
		seen_record[1] = 1;
		
	if((u33>0)&&(u33<final_width)&&(v33>0)&&(v33<final_height))
		seen_record[2] = 1;
	
	if((u44>0)&&(u44<final_width)&&(v44>0)&&(v44<final_height))
		seen_record[3] = 1;	
		
	int sum = 0;
		
	if((seen_record[0]==1)&&(img1[u11*final_height+v11]!=0))
		++sum;
				
	if((seen_record[1]==1)&&(img2[u22*final_height+v22]!=0))
		++sum;
			
	if((seen_record[2]==1)&&(img3[u33*final_height+v33]!=0))
		++sum;
			
	if((seen_record[3]==1)&&(img4[u44*final_height+v44]!=0))
		++sum;	
	
	point_cloud[index] = 0;
	if(sum>=3)
	{
		point_cloud[index] = 1;
	}
}