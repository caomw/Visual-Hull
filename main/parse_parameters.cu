#include "parse_parameters.cuh"

void Parameters::parse(string file_name){
	ifstream infile(file_name.c_str()); //, ios:: in | ios:: binary);

	char line[100];
	while(infile.getline(line, 100, '\n')){
		/// comment
		if(line[0] == '#' || line[0] == '\0') continue;

		if(strcmp(line, "gaussian_kernel") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%d", &gaussian_kernel);
			cout<<"gaussian kernel is:"<<endl<<gaussian_kernel<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(strcmp(line, "threshold") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%d", &threshold_img);
			cout<<"threshold is:"<<endl<<threshold_img<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		///
		if(strcmp(line, "input_images") == 0){
			infile.getline(line, 100, '\n');
			image_name = line;
			cout<<"input image name is:"<<endl<<image_name<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		/// P
		if(line[0] == 'P'){
			int nbr = line[1] - 48; // '0', '1', '2', '3' --> 0, 1, 2, 3
			cout<<"P"<<nbr<<" is: "<<endl;
			for(int i = 0; i < 3; ++i){
				infile.getline(line, 100, '\n');
				sscanf(line, "%g %g %g %g", &P[nbr][0+4*i], &P[nbr][1+4*i], &P[nbr][2+4*i], &P[nbr][3+4*i]);
				cout<<"["<<P[nbr][0+4*i]<<", "<<P[nbr][1+4*i]<<", "<<P[nbr][2+4*i]<<", "<<P[nbr][3+4*i]<<"]"<<endl;
			}
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(line[0] == 'T'){
			cout<<"T is:"<<endl;
			for(int i = 0; i < 4; ++i){
				infile.getline(line, 100, '\n');
				sscanf(line, "%f, %f, %f, %f", &T[0+4*i], &T[1+4*i], &T[2+4*i], &T[3+4*i]);
				cout<<"["<<T[0+4*i]<<", "<<T[1+4*i]<<", "<<T[2+4*i]<<", "<<T[3+4*i]<<"]"<<endl;
			}
			cout<<">---------------------------------------<"<<endl;
			continue;
		} // end if

		if(strcmp(line, "rows") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%d", &rows);
			cout<<"rows = "<<rows<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(strcmp(line, "columns") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%d", &columns);
			cout<<"columns = "<<columns<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(strcmp(line, "roomsize_x") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%f", &roomsize_x);
			cout<<"roomsize_x = "<<roomsize_x<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(strcmp(line, "roomsize_y") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%f", &roomsize_y);
			cout<<"roomsize_y = "<<roomsize_y<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(strcmp(line, "roomsize_z") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%f", &roomsize_z);
			cout<<"roomsize_z = "<<roomsize_z<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(strcmp(line, "thread_x") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%d", &thread_x);
			cout<<"thread_x = "<<thread_x<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(strcmp(line, "thread_y") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%d", &thread_y);
			cout<<"thread_y = "<<thread_y<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}	

		if(strcmp(line, "thread_z") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%d", &thread_z);
			cout<<"thread_z = "<<thread_z<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(strcmp(line, "block_x") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%d", &block_x);
			cout<<"block_x = "<<block_x<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}

		if(strcmp(line, "block_y") == 0){
			infile.getline(line, 100, '\n');
			sscanf(line, "%d", &block_y);
			cout<<"block_y = "<<block_y<<endl;
			cout<<">---------------------------------------<"<<endl;
			continue;
		}
	} /// end while

	///
	ratio_x = roomsize_x/( thread_x*block_x );
	ratio_y = roomsize_y/( thread_y*block_y );
	ratio_z = roomsize_z/( thread_z );

	cout<<"ratio_x = "<<ratio_x<<endl;
	cout<<">---------------------------------------<"<<endl;
	cout<<"ratio_y = "<<ratio_y<<endl;
	cout<<">---------------------------------------<"<<endl;
	cout<<"ratio_z = "<<ratio_z<<endl;
	cout<<">---------------------------------------<"<<endl;
} // end parse