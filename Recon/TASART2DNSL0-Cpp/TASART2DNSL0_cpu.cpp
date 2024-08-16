// CUDA Runtime
#include <cuda_runtime.h>

// CUDA utilities and CUDA includes
#include <helper_cuda.h>

// Helper functions
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_timer.h> 
#include <time.h>

#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

#include "TASART2DNSL0.h"
#include <stdio.h>
#include <windows.h>

#include <math.h>
std::string  gpuname;
int  *pArgc;
char ** pArgv;

std::string CUDA_SET_BEST_DEVICE()
{
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	std::string deviceName;
	int max_multiprocessors = 0, max_device = 0;
	for (int device = 0; device<num_devices; device++)
	{
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);
		if (max_multiprocessors < properties.multiProcessorCount)
		{
			max_multiprocessors = properties.multiProcessorCount;
			max_device = device;
			deviceName = std::string(properties.name);
		}
	}
	cudaSetDevice(max_device);
	return deviceName;
}
///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int  main(int argc, char** argv)
{
	pArgc = &argc;
	pArgv = argv;

	// define input parameters list
	
	Parameters  params =
	{
		59.5f,
		108.56f,
		512,
		512,
		42.0f / 512.0f,
		42.0f / 512.0f,
		.0f,
		.0f,
		912,
		0.0010125f,
		-3.75f,
		0.0f,
		2000,
		1501,
		181,
		(42.0f / 512.0f)*sqrt(2)*0.5f / 1500.0f,
		45.0f / 180.0f
	};

	//// copy raw data to host memory
	float *h_proj = new float[params.nr * params.na];
	memset(h_proj, 0, sizeof(float)* params.nr * params.na);

	FILE *fp1 = NULL;
	//fp1 = fopen("F://Ming Li/2D phantom/MF_water_proj257.txt", "rb");
	fp1 = fopen("E:/Liaofeiyang/DDPM/nngen-main/models/diffusion_models/AAAAAAA0.txt", "rb");
	fread(h_proj, sizeof(float), params.nr * params.na, fp1);
	fclose(fp1);

	//copy fbp data to host memory
	float *fbp_volume = new float[params.nx * params.ny];
	memset(fbp_volume, 0, sizeof(float)* params.nx * params.ny);

	//FILE *fp2 = NULL;
	//fp2 = fopen("F://Ming Li/2D phantom/fbp_volume.txt", "rb");
	//fread(fbp_volume, sizeof(float), params.nx * params.ny, fp2);
	//fclose(fp2);

	// copy table data to host memory
	float *lut_area = new float[params.TaDimx * params.TaDimy];
	memset(lut_area, 0, sizeof(float)* params.TaDimx * params.TaDimy);

	FILE *fp3 = NULL;
	fp3 = fopen("C:/Users/Study/Desktop/Simens_alut.txt", "rb");
	//fp3 = fopen("F://Ming Li/2D phantom/MF_Head_alut.txt", "rb");
	fread(lut_area, sizeof(float), params.TaDimx * params.TaDimy, fp3);
	fclose(fp3);

	//Copy betas to host memory
	float *betas = new float[params.na];
	memset(betas, 0, sizeof(float)* params.na);

	FILE *fp4 = NULL;
	//fp4 = fopen("F://Ming Li/2D phantom/MF_water_betas2.txt", "rb");
	fp4 = fopen("C:/Users/Study/Desktop/Simens_theta.txt", "rb");
	fread(betas, sizeof(float), params.na, fp4);
	fclose(fp4);

	// perform TASART2DNSL0 reconstruction  
	gpuname = CUDA_SET_BEST_DEVICE();

	clock_t clock_Start;
	clock_t clock_End;

	TASART2DNSL0 recons;
	recons.SetGeometry(params);
	// allocate TASART2DNSL0  reconstruction 
	float *h_volume = new float[params.nx * params.ny];
	memset(h_volume, 0, sizeof(float)*params.nx * params.ny);

	clock_Start = clock();
	recons.DoReconstruction(h_proj, h_volume, fbp_volume, lut_area, betas,10,1);
	clock_End = clock();

	// write reconstruction results to assigned files
	FILE *fp5 = NULL;
	fp5 = fopen("C:/Users/Study/Desktop/TASART2DNSL0_volume.txt", "wb");
	fwrite(h_volume, sizeof(float), params.nx * params.ny, fp5);
	fclose(fp5);
	printf("CUDA TASART2DNSL0 Time is %d ms\n", clock_End - clock_Start);
	cudaDeviceReset();

	// free memory space
	delete[] h_volume;
	delete[]fbp_volume;
	delete[] lut_area;
	delete[] h_proj;
	delete[] betas;

	getchar();
	return 0;

}
