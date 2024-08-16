
#include "TASART2DNSL0.h"
//#include "TAFPRAY2D.h"
#include <stdio.h>
#include <torch/extension.h>
#include <pybind11/numpy.h>

namespace py=pybind11;

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
	(42.0f / 512.0f) * sqrt(2.0f) * 0.5f / 1500.0f,
	45.0f / 180.0f
};


extern "C" {
	torch::Tensor Reconstruction_torch(torch::Tensor h_proj, py::array_t<float>lut_area, py::array_t<float>betas, uint nstart, uint ntv, uint sample_rate=1, bool permute=true) {
		//// copy raw data to host memory
		//float* h_proj = new float[params.nr * params.na];
		//memset(h_proj, 0, sizeof(float) * params.nr * params.na);
		params.na = int(2000 / sample_rate);

		//copy fbp data to host memory
		float* fbp_volume = new float[params.nx * params.ny];
		memset(fbp_volume, 0, sizeof(float) * params.nx * params.ny);

		TASART2DNSL0 recons;
		recons.SetGeometry(params);
		// allocate TASART2DNSL0  reconstruction 
		auto BS = h_proj.size(0);
		torch::Tensor h_volume = torch::zeros({ BS, 512,512 }, at::kFloat);
		for (int i = 0; i < BS; i++) {
			recons.DoReconstruction(h_proj.select(0,i).data_ptr<float>(), h_volume.select(0, i).data_ptr<float>(), fbp_volume, (float*)lut_area.request().ptr, (float*)betas.request().ptr, nstart, ntv);
		}
		delete[]fbp_volume;
		if (permute) {
			h_volume = h_volume.permute({ 0,2,1 });
		}
		
		return h_volume;
	}
};



extern "C" {
	torch::Tensor Projection_torch(torch::Tensor h_volume, py::array_t<float>lut_area, py::array_t<float>betas) {
		//FILE* fp1 = NULL;
		//fp1 = fopen(input_img_path.c_str(), "rb");
		//fread(h_volume, sizeof(float), params.nx * params.ny, fp1);
		//fclose(fp1);

		// Initialize projection array

		auto BS = h_volume.size(0);
		torch::Tensor h_proj = torch::zeros({ BS, 2000, 912 }, at::kFloat);

		TASART2DNSL0 RayProj;
		RayProj.SetGeometry(params);
		for (int i = 0; i < BS; i++) {
			RayProj.DoProjection(h_volume.select(0, i).data_ptr<float>(), h_proj.select(0, i).data_ptr<float>(), (float*)lut_area.request().ptr, (float*)betas.request().ptr);
		}
		return h_proj;
	}
};


PYBIND11_MODULE(TASART2DNSL0, m) {
	m.doc() = "do reconstruction/projection on gpu"; // optional module docstring
	m.def("recons_torch", &Reconstruction_torch, "do reconstruction on gpu and the input is Tensor",
		py::arg("h_proj"),py::arg("lut_area"), py::arg("betas"), py::arg("nstart"), py::arg("ntv"), py::arg("sample_rate"), py::arg("permute"));
	m.def("proj_torch", &Projection_torch, "do projection on gpu and the input is Tensor",
		py::arg("h_volume"), py::arg("lut_area"), py::arg("betas"));
};