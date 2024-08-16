#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "TASART2DNSL0.h"
// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> 
#include <helper_math.h>

#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h> 
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "cuda_texture_types.h"
// Volume size setting
int2 volumeSize;
cudaArray* d_volumeArray = 0;
float* cudavolume;
float* cur_volume;
float* norm_volume;

// projection data setting
int2 ProjSize;
cudaArray* d_ProjArray = 0;
float* cudaProj;

// correction data setting
int2 ProjBatchsize;
cudaArray* d_ProjBatchArray = 0;
float* cur_proj;
float* norm_proj;

//Table size setting
int2 tableSize;
cudaArray* d_tableArray = 0;
float* cudaTable;

//GeoDiv size setting
int2 geodivSize;
cudaArray* d_geodivArray = 0;

//Line size setting
float4* cudaLines;

// 2D texture for reconstruction
texture<float, cudaTextureType2D, cudaReadModeElementType> texVolume;
texture<float, cudaTextureType2D, cudaReadModeElementType> PrjtexRef;
texture<float, cudaTextureType2D, cudaReadModeElementType> CortexRef;
texture<float, cudaTextureType2D, cudaReadModeElementType> areaTex;
texture<float, cudaTextureType2D, cudaReadModeElementType> geodivTex;
texture<float4, cudaTextureType1D, cudaReadModeElementType> lineTex;

//CUDA  constant parameters definition
__constant__ float  c_lamda;
__constant__  int   c_nx;
__constant__  int   c_ny;
__constant__  float c_dx;
__constant__  float c_dy;
__constant__  float c_offset_x;
__constant__  float c_offset_y;
__constant__  float c_dso;
__constant__  float c_dsd;
__constant__  int   c_nr;
__constant__  float c_dr;
__constant__  float c_offset_r;
__constant__  float c_angle_start;
__constant__  int   c_angle_num;
__constant__  int   c_TaDimx;
__constant__  int   c_TaDimy;
__constant__  float c_TaDeltax;
__constant__  float c_TaDeltay;
__constant__  float c_rr;
__constant__  float c_xx;
__constant__  float c_yy;
__constant__  int c_nfoot;
__constant__ float2 c_alutOffset;
__constant__ float  c_voxBase;
__constant__ int    c_nlines;
__constant__ float2 c_src;
__constant__ float2 c_uv_s;
__constant__ float2 c_uv_t;
__constant__ float  c_viewang;


typedef unsigned int  uint;

////////// help functions
// compute y <- y + Ax
struct saxpy_functor
{
	const float a;

	saxpy_functor(float _a) : a(_a) {}

	__host__ __device__
		float operator()(const float& x, const float& y) const
	{
		return a * x + y;
	}
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{

	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}


// compute vector norm
// define transformation f(x)<- x^2
struct square
{
	__host__ __device__
		float operator()(float x)
	{
		return x * x;
	}
};

// fusion with transform_iterator
float vec_norm2_fast(thrust::device_vector<float>& x)
{
	return sqrt(thrust::transform_reduce(x.begin(), x.end(), square(), 0.0f, thrust::plus<float>()));
}


int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//rotate around Z 
float2 rotateCCW_z(float2 v, float cs, float sn)
{
	return make_float2(v.x * cs - v.y * sn, v.x * sn + v.y * cs);
}

struct Ray
{
	float2  o;  // ray origin site
	float2  d;  // ray  direction
};

extern "C"
void InitReconstrcution()
{
	//create 2D array
	cudaChannelFormatDesc chaDesc1 = cudaCreateChannelDesc<float>();

	//allocate global memoery to store correction data and backprojection data
	cudaMalloc((void**)&(cur_proj), ProjBatchsize.x * ProjBatchsize.y * sizeof(float));
	cudaMalloc((void**)&(norm_proj), ProjBatchsize.x * ProjBatchsize.y * sizeof(float));
	cudaMalloc((void**)&(cur_volume), volumeSize.x * volumeSize.y * sizeof(float));
	cudaMalloc((void**)&(norm_volume), volumeSize.x * volumeSize.y * sizeof(float));

	// allocate 2D array to bind 2D texture in cuda
	cudaMallocArray(&d_volumeArray, &chaDesc1, volumeSize.x, volumeSize.y);
	cudaMallocArray(&d_ProjBatchArray, &chaDesc1, ProjBatchsize.x, ProjBatchsize.y);

}

extern "C"
void MemsetCorrection()
{
	//cuda global memoery memset
	cudaMemset(cur_proj, 0, ProjBatchsize.x * ProjBatchsize.y * sizeof(float));
	cudaMemset(norm_proj, 0, ProjBatchsize.x * ProjBatchsize.y * sizeof(float));

}

extern "C"
void MemsetSumvolume()
{
	//cuda global memoery memset
	cudaMemset(cur_volume, 0, volumeSize.x * volumeSize.y * sizeof(float));
	cudaMemset(norm_volume, 0, volumeSize.x * volumeSize.y * sizeof(float));

}


extern "C"
void BindCorToTexRef()
{
	cudaChannelFormatDesc channelDescCor = cudaCreateChannelDesc<float>();

	CortexRef.addressMode[0] = cudaAddressModeClamp;
	CortexRef.addressMode[1] = cudaAddressModeClamp;
	CortexRef.filterMode = cudaFilterModePoint;
	CortexRef.normalized = false;

	// bind texture array to 2D volume
	cudaBindTextureToArray(CortexRef, d_ProjBatchArray, channelDescCor);
}

extern "C"
void BindImgToTex2D()
{

	cudaChannelFormatDesc channelDescVolume = cudaCreateChannelDesc<float>();

	// set texture parameters
	texVolume.normalized = false;
	texVolume.filterMode = cudaFilterModePoint;
	texVolume.addressMode[0] = cudaAddressModeClamp;
	texVolume.addressMode[1] = cudaAddressModeClamp;

	// bind texture array to 2D volume
	cudaBindTextureToArray(texVolume, d_volumeArray, channelDescVolume);
}

extern "C"
void InitGeodiv()
{
	cudaChannelFormatDesc chaDesc1 = cudaCreateChannelDesc<float>();

	// allocate 2D array to bind 2D texture in cuda
	cudaMallocArray(&d_geodivArray, &chaDesc1, geodivSize.x, geodivSize.y);
}

extern "C"
void BindGeodiv(float* Geodiv)
{
	cudaChannelFormatDesc channelDescGeo = cudaCreateChannelDesc<float>();

	// copy data to 2D array
	cudaMemcpyToArray(d_geodivArray, 0, 0, Geodiv, geodivSize.x * geodivSize.y * sizeof(float), cudaMemcpyHostToDevice);

	geodivTex.addressMode[0] = cudaAddressModeClamp;
	geodivTex.addressMode[1] = cudaAddressModeClamp;
	geodivTex.filterMode = cudaFilterModePoint;
	geodivTex.normalized = false;

	// bind texture array to 2D Geodiv
	cudaBindTextureToArray(geodivTex, d_geodivArray, channelDescGeo);

}

/////////////
////////////CUDA kernel functions
__device__  static float fetchAreaLut(int sidx, float2	xy)
{
	sidx = clamp(sidx, 0, c_nlines - 1);
	float4 L = tex1Dfetch(lineTex, sidx);
	float  ang, pos;
	ang = L.x,
		pos = L.y * xy.x + L.z * xy.y + L.w;

	float value;
	value = tex2D(areaTex,
		fabsf(pos) * c_alutOffset.x + 0.5f,
		ang * c_alutOffset.y + 0.5f);

	return pos < 0.0f ? c_voxBase - value : value;

}

__global__ static void update_lines_kernel(float4* lines, float beta)
{
	int is = blockIdx.x * blockDim.x + threadIdx.x;
	if (is >= c_nlines)
		return;

	float s0 = -c_rr + c_offset_r * c_dr;
	float gamma = s0 + is * c_dr;

	float2 P1 = c_src + make_float2(c_dsd * sin(beta + gamma), -c_dsd * cos(beta + gamma));
	float2 rayvec = P1 - c_src;

	float ang = atan2f(rayvec.y, rayvec.x) * (360.0f / (2.0f * (float)M_PI));
	if (ang < 0.0f)
		ang += 360.0f;

	float A = P1.y - c_src.y;
	float B = c_src.x - P1.x;
	float C = P1.x * c_src.y - c_src.x * P1.y;
	float Z = sqrtf(A * A + B * B);

	if (ang <= 45.0f) { ; }
	else if (ang <= 90.0f) { ang = 90.0f - ang; }
	else if (ang <= 135.0f) { ang = ang - 90.0f; }
	else if (ang <= 180.0f) { ang = 180.0f - ang; }
	else if (ang <= 225.0f) { ang = ang - 180.0f; }
	else if (ang <= 270.0f) { ang = 270.0f - ang; }
	else if (ang <= 315.0f) { ang = ang - 270.0f; }
	else { ang = 360.0f - ang; }

	lines[is] = make_float4(ang, A / Z, B / Z, C / Z);

}

__global__ static void  lut_init_foot_kernel(float* footinfo)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= c_nx || iy >= c_ny)
		return;

	int stride = c_nx * c_ny;
	footinfo += (iy * c_nx + ix);

	int nfoot2 = c_nfoot / 2;
	float2 xy = make_float2(((ix + 0.5) * c_dx - c_xx * 1.0f + c_offset_x),
		((iy + 0.5) * c_dy - c_yy * 1.0f + c_offset_y));

	float div = sqrt(SQR(xy.x - c_src.x) + SQR(xy.y - c_src.y));
	*footinfo = div; footinfo += stride;

	float mag = c_dsd / sqrt(SQR(dot(c_uv_s, xy)) + SQR((dot(c_uv_t, xy) + c_dso)));
	*footinfo = mag;
	footinfo += stride;

	float gamma = atan(dot(c_uv_s, xy) / (dot(c_uv_t, xy) + c_dso));
	int   s_bin = (int)floorf(gamma / c_dr + 0.5f * (c_nr - 1) - c_offset_r) - nfoot2;

	*footinfo = float(s_bin);
	footinfo += stride;

	int is = s_bin;
	float area0 = fetchAreaLut(is, xy); ++is;
	for (int ifoot = 0; ifoot < c_nfoot; ++ifoot, ++is, footinfo += stride)
	{
		float area1 = fetchAreaLut(is, xy);
		*footinfo = fabsf(area0 - area1);
		area0 = area1;
	}

}

__global__ static void lut_fp_kernel(float* d_proj, float* footinfo, const float val)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= c_nx || iy >= c_ny)
		return;

	int stride = c_nx * c_ny;
	footinfo += (iy * c_nx + ix);

	float shdata[8];
	float* p_div = shdata + 0;
	float* p_sbin = shdata + 2;
	float* p_foot = shdata + 3;

	for (int idx = 0; idx < 3 + c_nfoot; ++idx)
	{
		shdata[idx] = footinfo[idx * stride];

	}
	__syncthreads();


	float att = (val > 0.0f) ? val : tex2D(texVolume, ix + 0.5f, iy + 0.5f);

	if (att == 0.0f)
		return;

	float div = att / (*p_div);
	int is = int(*p_sbin);
	for (int ifoot = 0; ifoot < c_nfoot; ++is, ++ifoot)
	{
		float area = p_foot[ifoot];

		if (is < 0 || is >= c_nr || area <= 0.0f || div == 0.0f)
			continue;

		atomicAdd(d_proj + is, div * area);
	}
}

__global__ static void apply_geodiv_kernel(float* proj)
{
	int is = blockIdx.x * blockDim.x + threadIdx.x;

	if (is >= c_nr)
		return;

	proj[is] *= tex2D(geodivTex, is + 0.5f, 0.5f);
}



__global__ static void lut_bp_kernel(float* image, float* footinfo, const float val)
{

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= c_nx || iy >= c_ny)
		return;

	int stride = c_nx * c_ny;
	footinfo += (iy * c_nx + ix);

	float shdata[8];
	float* p_div = shdata + 0;
	float* p_sbin = shdata + 2;
	float* p_foot = shdata + 3;

	for (int idx = 0; idx < 3 + c_nfoot; ++idx)
	{
		shdata[idx] = footinfo[idx * stride];

	}
	__syncthreads();

	float div = 1.0f / (*p_div);
	float sum = 0.0f;
	int is = int(*p_sbin);

	for (int ifoot = 0; ifoot < c_nfoot; ++is, ++ifoot)
	{
		float area = p_foot[ifoot];
		if (val <= 0.0f)
		{
			float density = tex2D(CortexRef, is + 0.5f, 0.5f);
			sum += density * div * area;
		}
		else
		{
			float density = tex2D(geodivTex, is + 0.5f, 0.5f);
			sum += density * div * area;
		}
	}
	atomicAdd(image + iy * c_nx + ix, sum);

}

__global__ static void correction_kernel(float* d_proj, float* d_norm, int angle_index)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= c_nr)
		return;

	float p = d_proj[index];
	float n = d_norm[index];
	float m = tex2D(PrjtexRef, index, angle_index);

	float sc;
	sc = tex2D(geodivTex, index + 0.5f, 0.5f);

	float out = (n > 0.0f) ? sc * ((m - p) / n) : 0.0f;

	d_proj[index] = out;
}

__global__ static void update_kernel(float* cudavolume, float* d_back, float* d_norm)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= c_nx || iy >= c_ny)
		return;

	int index = iy * c_nx + ix;

	float reval = cudavolume[index];
	float bpval = d_back[index];
	float nval = d_norm[index];

	float out = (nval > 0.0f) ? c_lamda * (bpval / nval) : 0.0f;
	cudavolume[index] = fmaxf(reval + out, 0.0f);

}


// NSL0TV Kernel used to compute image gradient in descent step
__global__ void Grad_NSL0TV(float* tvgrad, float sigma)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= c_nx || iy >= c_ny)
		return;
	int index = iy * c_nx + ix;

	float x = ix * 1.0f + 0.5f;
	float y = iy * 1.0f + 0.5f;

	float x_plus = (ix + 1) * 1.0f + 0.5f;
	float y_plus = (iy + 1) * 1.0f + 0.5f;

	float x_minus = (ix - 1) * 1.0f + 0.5f;
	float y_minus = (iy - 1) * 1.0f + 0.5f;

	float mins = 0.0001f;
	float temp = 0;

	// Weight factor
	float D_xy = sqrt(mins * mins
		+ (tex2D(texVolume, x, y) - tex2D(texVolume, x_plus, y)) * (tex2D(texVolume, x, y) - tex2D(texVolume, x_plus, y))
		+ (tex2D(texVolume, x, y) - tex2D(texVolume, x, y_plus)) * (tex2D(texVolume, x, y) - tex2D(texVolume, x, y_plus)));

	float Dx_minus = sqrt(mins * mins
		+ (tex2D(texVolume, x_minus, y) - tex2D(texVolume, x, y)) * (tex2D(texVolume, x_minus, y) - tex2D(texVolume, x, y))
		+ (tex2D(texVolume, x_minus, y) - tex2D(texVolume, x_minus, y_plus)) * (tex2D(texVolume, x_minus, y) - tex2D(texVolume, x_minus, y_plus)));

	float Dy_minus = sqrt(mins * mins
		+ (tex2D(texVolume, x, y_minus) - tex2D(texVolume, x, y)) * (tex2D(texVolume, x, y_minus) - tex2D(texVolume, x, y))
		+ (tex2D(texVolume, x, y_minus) - tex2D(texVolume, x_plus, y_minus)) * (tex2D(texVolume, x, y_minus) - tex2D(texVolume, x_plus, y_minus)));

	float W_xy = (2 / sigma) / ((exp(D_xy / (2 * sigma)) + exp(-D_xy / (2 * sigma))) * (exp(D_xy / (2 * sigma)) + exp(-D_xy / (2 * sigma))));

	float Wx_minus = (2 / sigma) / ((exp(Dx_minus / (2 * sigma)) + exp(-Dx_minus / (2 * sigma))) * (exp(Dx_minus / (2 * sigma)) + exp(-Dx_minus / (2 * sigma))));

	float Wy_minus = (2 / sigma) / ((exp(Dy_minus / (2 * sigma)) + exp(-Dy_minus / (2 * sigma))) * (exp(Dy_minus / (2 * sigma)) + exp(-Dy_minus / (2 * sigma))));

	// NSL0TV derivative
	temp += W_xy * (tex2D(texVolume, x, y) - tex2D(texVolume, x_plus, y)
		+ tex2D(texVolume, x, y) - tex2D(texVolume, x, y_plus))
		/ (D_xy);

	temp -= Wx_minus * (tex2D(texVolume, x_minus, y) - tex2D(texVolume, x, y))
		/ (Dx_minus);

	temp -= Wy_minus * (tex2D(texVolume, x, y_minus) - tex2D(texVolume, x, y))
		/ (Dy_minus);

	if (temp < mins * mins)
		temp = 0;

	tvgrad[index] = temp;

}


// pixel and gradient value nonnegative Kernel
__global__ void nonnegative(float* x, float* g)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= c_nx || iy >= c_ny)
		return;

	int index = iy * c_nx + ix;

	if (x[index] < 0)
		x[index] = 0;

	if (x[index] < 0 & g[index] > 0)
		g[index] = 0.00000001f;
}


////////////////////
///////// class member functions

void TASART2DNSL0::InitArealut()
{
	// setting Lines 
	int m_nlines = m_nr + 1;
	cudaMalloc((void**)&cudaLines, m_nlines * sizeof(float4));
	cudaMemset(cudaLines, 0, m_nlines * sizeof(float4));

	//create 2D array
	cudaChannelFormatDesc chaDesc1 = cudaCreateChannelDesc<float>();

	// allocate 2D array to bind 2D texture in cuda
	cudaMallocArray(&d_tableArray, &chaDesc1, tableSize.x, tableSize.y);
	cudaMemcpyToSymbol(c_nlines, &m_nlines, sizeof(int));
}


void TASART2DNSL0::BindAreaLut()
{
	cudaChannelFormatDesc channelDescLut = cudaCreateChannelDesc<float>();

	// copy data to 2D array
	cudaMemcpyToArray(d_tableArray, 0, 0, cudaTable, tableSize.x * tableSize.y * sizeof(float), cudaMemcpyDeviceToDevice);
	areaTex.normalized = false;
	areaTex.filterMode = cudaFilterModeLinear;
	areaTex.addressMode[0] = cudaAddressModeClamp;
	areaTex.addressMode[1] = cudaAddressModeClamp;
	// bind texture array to 2D volume
	cudaBindTextureToArray(areaTex, d_tableArray, channelDescLut);
	cudaBindTexture(0, lineTex, cudaLines, lineTex.channelDesc);
	float2 alutOffset = make_float2(1.0f / m_TaDeltax, 1.0f / m_TaDeltay);
	float  voxBase = fabsf(m_dx * m_dy);
	cudaMemcpyToSymbol(c_alutOffset, &alutOffset, sizeof(float2));
	cudaMemcpyToSymbol(c_voxBase, &voxBase, sizeof(float));

}

//Computing ray spreading term
void TASART2DNSL0::_cmpGeoDiv()
{
	std::vector<float>   geodiv(m_nr);
	for (int i = 0; i < m_nr; ++i)
	{
		geodiv[i] = (1.0f / m_dr);

	}
	InitGeodiv();
	BindGeodiv(&geodiv[0]);
}
//Computing Lines parameters
void TASART2DNSL0::_updateConst(float2 uv_s, float2 uv_t, float2 src, float ang)
{
	cudaMemcpyToSymbol(c_src, &src, sizeof(float2));
	cudaMemcpyToSymbol(c_uv_s, &uv_s, sizeof(float2));
	cudaMemcpyToSymbol(c_uv_t, &uv_t, sizeof(float2));

	ang = fmodf(ang, 360.0f);
	if (ang < 0.0f)
		ang += 360.0f;

	cudaMemcpyToSymbol(c_viewang, &ang, sizeof(float));

}

void TASART2DNSL0::_updateLines(float4* cudaLines, float beta)
{
	dim3 blk(256, 1, 1);
	dim3 grd(iDivUp(m_nr + 1, blk.x), 1, 1);
	update_lines_kernel << <grd, blk >> > (cudaLines, beta);
}

// Computing pixel foot information
void TASART2DNSL0::_init_footprint(float* footinfo)
{
	dim3 blk(16, 16, 1);
	dim3 grd(iDivUp(m_nx, blk.x), iDivUp(m_ny, blk.y));
	lut_init_foot_kernel << <grd, blk >> > (footinfo);
}

//Forward projection Ax
void TASART2DNSL0::_Fp_Ax(float* d_proj, float* footinfo, const float val)
{
	// do forward projection
	_do_projection(d_proj, footinfo, val);
	// normalization by scaling the projection with geometric spreading term
	_do_proj_scaling(d_proj);
}

void TASART2DNSL0::_do_projection(float* d_proj, float* footinfo, const float val)
{
	dim3 blk(16, 16, 1);
	dim3 grd(iDivUp(m_nx, blk.x), iDivUp(m_ny, blk.y));
	lut_fp_kernel << <grd, blk >> > (d_proj, footinfo, val);
}

void TASART2DNSL0::_do_proj_scaling(float* d_proj)
{
	dim3 blk(256, 1, 1);
	dim3 grd(iDivUp(m_nr, blk.x), 1, 1);
	apply_geodiv_kernel << <grd, blk >> > (d_proj);
}

void TASART2DNSL0::sart2d_correct(float* d_proj, float* d_norm, int angle_index)
{
	dim3 blk(256, 1, 1);
	dim3 grd(iDivUp(m_nr, blk.x), 1, 1);
	correction_kernel << <grd, blk >> > (d_proj, d_norm, angle_index);
}

void TASART2DNSL0::_Bp_Ab(float* image, float* footinfo, const float val)
{
	dim3 blk(16, 16, 1);
	dim3 grd(iDivUp(m_nx, blk.x), iDivUp(m_ny, blk.y));
	lut_bp_kernel << <grd, blk >> > (image, footinfo, val);

}

void TASART2DNSL0::sart2d_update(float* cudavolume, float* d_back, float* d_norm)
{
	dim3 blk(16, 16, 1);
	dim3 grd(iDivUp(m_nx, blk.x), iDivUp(m_ny, blk.y));
	update_kernel << <grd, blk >> > (cudavolume, d_back, d_norm);

}


TASART2DNSL0::TASART2DNSL0(void)
{

}

TASART2DNSL0::~TASART2DNSL0(void)
{

}

void TASART2DNSL0::SetGeometry(Parameters params)
{
	m_dso = params.dso;
	m_dsd = params.dsd;
	m_nx = params.nx;
	m_ny = params.ny;
	m_dx = params.dx;
	m_dy = params.dy;
	m_offset_x = params.offset_x;
	m_offset_y = params.offset_y;
	m_nr = params.nr;
	m_dr = params.dr;
	m_offset_r = params.offset_r;
	m_angle_start = params.angle_start;
	m_na = params.na;
	m_TaDimx = params.TaDimx;
	m_TaDimy = params.TaDimy;
	m_TaDeltax = params.TaDeltax;
	m_TaDeltay = params.TaDeltay;

}

int TASART2DNSL0::DoReconstruction(float* h_proj, float* h_volume, float* fbp_volume, float* lut_area, float* betas, uint nsart, uint ntv)
{

	// compute semi-length of image and detector
	float m_xx = m_nx * m_dx * 0.5f;
	float m_yy = m_ny * m_dy * 0.5f;

	float m_rr = m_nr * m_dr * 0.5f;

	float lamda = 0.24f;
	float alpha = 0.1f;
	const int  nfoot = 5;

	// copy const variable to device
	cudaMemcpyToSymbol(c_lamda, &lamda, sizeof(float));
	cudaMemcpyToSymbol(c_dso, &m_dso, sizeof(float));
	cudaMemcpyToSymbol(c_dsd, &m_dsd, sizeof(float));
	cudaMemcpyToSymbol(c_nx, &m_nx, sizeof(int));
	cudaMemcpyToSymbol(c_ny, &m_ny, sizeof(int));
	cudaMemcpyToSymbol(c_dx, &m_dx, sizeof(float));
	cudaMemcpyToSymbol(c_dy, &m_dy, sizeof(float));
	cudaMemcpyToSymbol(c_offset_x, &m_offset_x, sizeof(float));
	cudaMemcpyToSymbol(c_offset_y, &m_offset_y, sizeof(float));
	cudaMemcpyToSymbol(c_nr, &m_nr, sizeof(int));
	cudaMemcpyToSymbol(c_dr, &m_dr, sizeof(float));
	cudaMemcpyToSymbol(c_offset_r, &m_offset_r, sizeof(float));
	cudaMemcpyToSymbol(c_angle_start, &m_angle_start, sizeof(float));
	cudaMemcpyToSymbol(c_angle_num, &m_na, sizeof(int));
	cudaMemcpyToSymbol(c_TaDimx, &m_TaDimx, sizeof(int));
	cudaMemcpyToSymbol(c_TaDimy, &m_TaDimy, sizeof(int));
	cudaMemcpyToSymbol(c_TaDeltax, &m_TaDeltax, sizeof(float));
	cudaMemcpyToSymbol(c_TaDeltay, &m_TaDeltay, sizeof(float));

	cudaMemcpyToSymbol(c_xx, &m_xx, sizeof(float));
	cudaMemcpyToSymbol(c_yy, &m_yy, sizeof(float));
	cudaMemcpyToSymbol(c_rr, &m_rr, sizeof(float));
	cudaMemcpyToSymbol(c_nfoot, &nfoot, sizeof(int));

	volumeSize = make_int2(m_nx, m_ny);
	ProjBatchsize = make_int2(m_nr, 1);
	tableSize = make_int2(m_TaDimx, m_TaDimy);
	geodivSize = make_int2(m_nr, 1);

	// setting blockSize and GridSize for NSL0TV descent
	dim3 TVblockSize(16, 16, 1);
	dim3 TVgridSize(m_nx / TVblockSize.x, m_ny / TVblockSize.y, 1);

	//Computing ray spreading term and bind texture
	_cmpGeoDiv();

	//Set footprint size and set table lut
	float* footinfo;
	cudaMalloc((void**)&footinfo, m_nx * m_ny * (nfoot + 3) * sizeof(float));
	cudaMemset(footinfo, 0, m_nx * m_ny * (nfoot + 3) * sizeof(float));

	InitArealut();
	thrust::device_vector<float> lut_for(lut_area, lut_area + m_TaDimx * m_TaDimy);
	cudaTable = thrust::raw_pointer_cast(&lut_for[0]);
	BindAreaLut();

	//Initialization
	InitReconstrcution();
	BindCorToTexRef();

	thrust::device_vector<float> x_for(fbp_volume, fbp_volume + m_nx * m_ny);
	thrust::device_vector<float> x_back(fbp_volume, fbp_volume + m_nx * m_ny);

	BindImgToTex2D();

	// transfer device_vector to raw pointer
	float* pxfor = thrust::raw_pointer_cast(&x_for[0]);
	float* pxback = thrust::raw_pointer_cast(&x_back[0]);

	// Copy projection data to cuda and bind to texture array
	ProjSize = make_int2(m_nr, m_na);
	thrust::device_vector<float> d_proj(h_proj, h_proj + m_nr * m_na);
	cudaProj = thrust::raw_pointer_cast(&d_proj[0]);

	//create 2D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	// allocate 2D array to bind 2D texture in cuda
	cudaMallocArray(&d_ProjArray, &channelDesc, ProjSize.x, ProjSize.y);

	cudaChannelFormatDesc channelDescProj = cudaCreateChannelDesc<float>();

	// copy data to 2D array
	cudaMemcpyToArray(d_ProjArray, 0, 0, cudaProj, ProjSize.x * ProjSize.y * sizeof(float), cudaMemcpyDeviceToDevice);

	PrjtexRef.addressMode[0] = cudaAddressModeClamp;
	PrjtexRef.addressMode[1] = cudaAddressModeClamp;
	PrjtexRef.filterMode = cudaFilterModePoint;
	PrjtexRef.normalized = false;

	// bind texture array to 2D volume
	cudaBindTextureToArray(PrjtexRef, d_ProjArray, channelDescProj);

	// define device vector to save gradient data
	thrust::device_vector<float> grad_new(m_nx * m_ny);
	float* pgrad_new = thrust::raw_pointer_cast(&grad_new[0]);
	thrust::fill(grad_new.begin(), grad_new.end(), 0);

	// save sart result as output
	thrust::device_vector<float> x_res(m_nx * m_ny);
	float* px_new = thrust::raw_pointer_cast(&x_res[0]);
	thrust::fill(x_res.begin(), x_res.end(), 0);

	// iterative reconstruction procedure

	float sigma = 0.8;

	for (uint iteration = 0; iteration < nsart; ++iteration)
	{
		// extract raw pointer from device vector
		cudavolume = thrust::raw_pointer_cast(&x_for[0]);
		// save the results of TV iteration or initial image
		thrust::copy(x_for.begin(), x_for.end(), x_back.begin());
		// SART algorithm
		for (int ia = 0; ia < m_na; ia++)
		{
			float beta = (betas[ia] - m_angle_start) * float(DEG2RAD);
			float   cs = cosf(beta);
			float   sn = sinf(beta);
			float2 uv_t = rotateCCW_z(make_float2(0, -1), cs, sn);
			float2 uv_s = rotateCCW_z(make_float2(1, 0), cs, sn);
			float2 src = rotateCCW_z(make_float2(0, m_dso), cs, sn);

			_updateConst(uv_s, uv_t, src, beta * float(RAD2DEG));

			//Update Lines 
			_updateLines(cudaLines, beta);

			//Update pixel foot information
			_init_footprint(footinfo);

			// copy last reconstruction data to 2D array
			cudaMemcpyToArray(d_volumeArray, 0, 0, cudavolume, volumeSize.x * volumeSize.y * sizeof(float), cudaMemcpyDeviceToDevice);

			//memset cur_proj and norm_proj
			MemsetCorrection();

			//Projection
			_Fp_Ax(cur_proj, footinfo, -1.0f);
			_Fp_Ax(norm_proj, footinfo, 1.0f);

			// compute correction
			sart2d_correct(cur_proj, norm_proj, ia);

			// copy current correction data to 2D array
			cudaMemcpyToArray(d_ProjBatchArray, 0, 0, cur_proj, ProjBatchsize.x * ProjBatchsize.y * sizeof(float), cudaMemcpyDeviceToDevice);

			//memset cur_volume  and norm_volume
			MemsetSumvolume();

			//Back-projection
			_Bp_Ab(cur_volume, footinfo, -1.0f);
			_Bp_Ab(norm_volume, footinfo, 1.0f);

			//Update cudavolume
			sart2d_update(cudavolume, cur_volume, norm_volume);

		}

		// compute residual between x_for and x_back
		saxpy_fast(-1.0f, x_for, x_back);
		float dp = vec_norm2_fast(x_back);

		// save the results of sart iteration
		thrust::copy(x_for.begin(), x_for.end(), x_back.begin());
		thrust::copy(x_for.begin(), x_for.end(), x_res.begin());

		sigma = sigma * 0.90f;
		sigma = (sigma > 0.1f) ? sigma : 0.1f;

		float dtvg = alpha * dp;

		for (uint itv = 0; itv < ntv; ++itv)
		{
			// extract raw pointer from device vector
			cudavolume = thrust::raw_pointer_cast(&x_for[0]);

			// copy last reconstruction data to 2D array
			cudaMemcpyToArray(d_volumeArray, 0, 0, cudavolume, volumeSize.x * volumeSize.y * sizeof(float), cudaMemcpyDeviceToDevice);

			// NSL0TV norm derivative
			Grad_NSL0TV << < TVgridSize, TVblockSize >> > (pgrad_new, sigma);

			// nonnegative constraint
			nonnegative << < TVgridSize, TVblockSize >> > (cudavolume, pgrad_new);

			// Normalization
			float normg = vec_norm2_fast(grad_new);
			saxpy_fast(-1.0f * dtvg / normg, grad_new, x_for);

		}

		//compute residual between x_for and x_back
		saxpy_fast(-1.0f, x_for, x_back);
		float dg = vec_norm2_fast(x_back);

		// decreasing relaxation factor
		if (dg > (0.995 * dp))
			alpha = alpha * 0.96;
		lamda = lamda * 0.95;
		cudaMemcpyToSymbol(c_lamda, &lamda, sizeof(float));

	}

	// copy reconstruction data to host h_volume
	cudaMemcpy(h_volume, px_new, volumeSize.x * volumeSize.y * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_volume, pxfor, volumeSize.x * volumeSize.y * sizeof(float), cudaMemcpyDeviceToHost);

	// cuda unbind
	cudaUnbindTexture(texVolume);
	cudaUnbindTexture(PrjtexRef);
	cudaUnbindTexture(CortexRef);
	cudaUnbindTexture(geodivTex);
	cudaUnbindTexture(areaTex);
	cudaUnbindTexture(lineTex);

	//Cuda free
	cudaFree(footinfo);
	cudaFree(cudaLines);
	cudaFree(cur_proj);
	cudaFree(norm_proj);
	cudaFree(cur_volume);
	cudaFree(norm_volume);

	cudaFreeArray(d_volumeArray);
	cudaFreeArray(d_ProjArray);
	cudaFreeArray(d_ProjBatchArray);
	cudaFreeArray(d_tableArray);
	cudaFreeArray(d_geodivArray);

	return 0;
}


//typedef unsigned int  uint;

// Volume size setting
//int2 volumeSize;
//cudaArray *d_volumeArray = 0;
//float *cudavolume;

//Table size setting
//int2 tableSize;
//cudaArray *d_tableArray = 0;
//float *cudaTable;

//GeoDiv size setting
//int2 geodivSize;
//cudaArray *d_geodivArray = 0;

//Line size setting
//float4 *cudaLines;

// 2D texture for projection
//texture<float, cudaTextureType2D, cudaReadModeElementType> texVolume;
//texture<float, cudaTextureType2D, cudaReadModeElementType> areaTex;
//texture<float, cudaTextureType2D, cudaReadModeElementType> geodivTex;
//texture<float4, cudaTextureType1D, cudaReadModeElementType> lineTex;

//CUDA  constant parameters definition
//__constant__  int   c_nx;
//__constant__  int   c_ny;
//__constant__  float c_dx;
//__constant__  float c_dy;
//__constant__  float c_offset_x;
//__constant__  float c_offset_y;
//__constant__  float c_dso;
//__constant__  float c_dsd;
//__constant__  int   c_nr;
//__constant__  float c_dr;
//__constant__  float c_offset_r;
//__constant__  float c_angle_start;
//__constant__  int   c_angle_num;
//__constant__  int   c_TaDimx;
//__constant__  int   c_TaDimy;
//__constant__  float c_TaDeltax;
//__constant__  float c_TaDeltay;
//__constant__  float c_rr;
//__constant__  float c_xx;
//__constant__  float c_yy;
//__constant__  int c_nfoot;
//__constant__ float2 c_alutOffset;
//__constant__ float  c_voxBase;
//__constant__ int    c_nlines;
//__constant__ float2 c_src;
//__constant__ float2 c_uv_s;
//__constant__ float2 c_uv_t;
//__constant__ float  c_viewang;

//struct Ray
//{
//	float2  o;  // ray origin site
//	float2  d;  // ray  direction
//};
// help functions
//int iDivUp(int a, int b) {
//	return (a % b != 0) ? (a / b + 1) : (a / b);
//}

//rotate around Z 
//float2 rotateCCW_z(float2 v, float cs, float sn)
//{
//	return make_float2(v.x * cs - v.y * sn, v.x * sn + v.y * cs);
//}

extern "C"
void InitProjection()
{
	//create 2D array
	cudaChannelFormatDesc chaDesc1 = cudaCreateChannelDesc<float>();

	// allocate 2D array to bind 2D texture in cuda
	cudaMallocArray(&d_volumeArray, &chaDesc1, volumeSize.x, volumeSize.y);

}

extern "C"
void Proj_BindImgToTex2D()
{

	cudaChannelFormatDesc channelDescVolume = cudaCreateChannelDesc<float>();

	// copy data to 2D array
	cudaMemcpyToArray(d_volumeArray, 0, 0, cudavolume, volumeSize.x * volumeSize.y * sizeof(float), cudaMemcpyDeviceToDevice);

	// set texture parameters
	texVolume.normalized = false;
	texVolume.filterMode = cudaFilterModePoint;
	texVolume.addressMode[0] = cudaAddressModeClamp;
	texVolume.addressMode[1] = cudaAddressModeClamp;

	// bind texture array to 2D volume
	cudaBindTextureToArray(texVolume, d_volumeArray, channelDescVolume);
}

//extern "C"
//void InitGeodiv()
//{
//	cudaChannelFormatDesc chaDesc1 = cudaCreateChannelDesc<float>();
//
//	// allocate 2D array to bind 2D texture in cuda
//	cudaMallocArray(&d_geodivArray, &chaDesc1, geodivSize.x, geodivSize.y);
//}

//extern "C"
//void BindGeodiv(float* Geodiv)
//{
//	cudaChannelFormatDesc channelDescGeo = cudaCreateChannelDesc<float>();
//
//	// copy data to 2D array
//	cudaMemcpyToArray(d_geodivArray, 0, 0, Geodiv, geodivSize.x * geodivSize.y * sizeof(float), cudaMemcpyHostToDevice);
//
//	geodivTex.addressMode[0] = cudaAddressModeClamp;
//	geodivTex.addressMode[1] = cudaAddressModeClamp;
//	geodivTex.filterMode = cudaFilterModePoint;
//	geodivTex.normalized = false;
//
//	// bind texture array to 2D Geodiv
//	cudaBindTextureToArray(geodivTex, d_geodivArray, channelDescGeo);
//
//}

///////////
//////////CUDA kernel functions
//__device__  static float fetchAreaLut(int sidx, float2	xy)
//{
//	sidx = clamp(sidx, 0, c_nlines - 1);
//	float4 L = tex1Dfetch(lineTex, sidx);
//	float  ang, pos;
//	ang = L.x,
//		pos = L.y * xy.x + L.z * xy.y + L.w;
//
//	float value;
//	value = tex2D(areaTex,
//		fabsf(pos) * c_alutOffset.x + 0.5f,
//		ang * c_alutOffset.y + 0.5f);
//
//	return pos < 0.0f ? c_voxBase - value : value;
//
//}

//__global__ static void update_lines_kernel(float4* lines, float beta)
//{
//	int is = blockIdx.x * blockDim.x + threadIdx.x;
//	if (is >= c_nlines)
//		return;
//
//	float s0 = -c_rr + c_offset_r * c_dr;
//	float gamma = s0 + is * c_dr;
//
//	float2 P1 = c_src + make_float2(c_dsd * sin(beta + gamma), -c_dsd * cos(beta + gamma));
//	float2 rayvec = P1 - c_src;
//
//	float ang = atan2f(rayvec.y, rayvec.x) * (360.0f / (2.0f * (float)M_PI));
//	if (ang < 0.0f)
//		ang += 360.0f;
//
//	float A = P1.y - c_src.y;
//	float B = c_src.x - P1.x;
//	float C = P1.x * c_src.y - c_src.x * P1.y;
//	float Z = sqrtf(A * A + B * B);
//
//	if (ang <= 45.0f) { ; }
//	else if (ang <= 90.0f) { ang = 90.0f - ang; }
//	else if (ang <= 135.0f) { ang = ang - 90.0f; }
//	else if (ang <= 180.0f) { ang = 180.0f - ang; }
//	else if (ang <= 225.0f) { ang = ang - 180.0f; }
//	else if (ang <= 270.0f) { ang = 270.0f - ang; }
//	else if (ang <= 315.0f) { ang = ang - 270.0f; }
//	else { ang = 360.0f - ang; }
//
//	lines[is] = make_float4(ang, A / Z, B / Z, C / Z);
//
//}

//__global__ static void  lut_init_foot_kernel(float* footinfo)
//{
//	int ix = blockIdx.x * blockDim.x + threadIdx.x;
//	int iy = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (ix >= c_nx || iy >= c_ny)
//		return;
//
//	int stride = c_nx * c_ny;
//	footinfo += (iy * c_nx + ix);
//
//	int nfoot2 = c_nfoot / 2;
//	float2 xy = make_float2(((ix + 0.5) * c_dx - c_xx * 1.0f + c_offset_x),
//		((iy + 0.5) * c_dy - c_yy * 1.0f + c_offset_y));
//
//	float div = sqrt(SQR(xy.x - c_src.x) + SQR(xy.y - c_src.y));
//	*footinfo = div; footinfo += stride;
//
//	float mag = c_dsd / sqrt(SQR(dot(c_uv_s, xy)) + SQR((dot(c_uv_t, xy) + c_dso)));
//	*footinfo = mag;
//	footinfo += stride;
//
//	float gamma = atan(dot(c_uv_s, xy) / (dot(c_uv_t, xy) + c_dso));
//	int   s_bin = (int)floorf(gamma / c_dr + 0.5f * (c_nr - 1) - c_offset_r) - nfoot2;
//
//	*footinfo = float(s_bin);
//	footinfo += stride;
//
//	int is = s_bin;
//	float area0 = fetchAreaLut(is, xy); ++is;
//	for (int ifoot = 0; ifoot < c_nfoot; ++ifoot, ++is, footinfo += stride)
//	{
//		float area1 = fetchAreaLut(is, xy);
//		*footinfo = fabsf(area0 - area1);
//		area0 = area1;
//	}
//
//}

//__global__ static void lut_fp_kernel(float* d_proj, float* footinfo, const float val)
//{
//	int ix = blockIdx.x * blockDim.x + threadIdx.x;
//	int iy = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (ix >= c_nx || iy >= c_ny)
//		return;
//
//	int stride = c_nx * c_ny;
//	footinfo += (iy * c_nx + ix);
//
//	float shdata[8];
//	float* p_div = shdata + 0;
//	float* p_sbin = shdata + 2;
//	float* p_foot = shdata + 3;
//
//	for (int idx = 0; idx < 3 + c_nfoot; ++idx)
//	{
//		shdata[idx] = footinfo[idx * stride];
//
//	}
//	__syncthreads();
//
//
//	float att = (val > 0.0f) ? val : tex2D(texVolume, ix + 0.5f, iy + 0.5f);
//
//	if (att == 0.0f)
//		return;
//
//	float div = att / (*p_div);
//	int is = int(*p_sbin);
//	for (int ifoot = 0; ifoot < c_nfoot; ++is, ++ifoot)
//	{
//		float area = p_foot[ifoot];
//
//		if (is < 0 || is >= c_nr || area <= 0.0f || div == 0.0f)
//			continue;
//
//		atomicAdd(d_proj + is, div * area);
//	}
//}

//__global__ static void apply_geodiv_kernel(float* proj)
//{
//	int is = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (is >= c_nr)
//		return;
//
//	proj[is] *= tex2D(geodivTex, is + 0.5f, 0.5f);
//}

////////////////
///// class member functions

//void TASART2DNSL0::InitArealut()
//{
//	// setting Lines 
//	int m_nlines = m_nr + 1;
//	cudaMalloc((void**)&cudaLines, m_nlines * sizeof(float4));
//	cudaMemset(cudaLines, 0, m_nlines * sizeof(float4));
//
//	//create 2D array
//	cudaChannelFormatDesc chaDesc1 = cudaCreateChannelDesc<float>();
//
//	// allocate 2D array to bind 2D texture in cuda
//	cudaMallocArray(&d_tableArray, &chaDesc1, tableSize.x, tableSize.y);
//	cudaMemcpyToSymbol(c_nlines, &m_nlines, sizeof(int));
//}


//void TASART2DNSL0::BindAreaLut()
//{
//	cudaChannelFormatDesc channelDescLut = cudaCreateChannelDesc<float>();
//
//	// copy data to 2D array
//	cudaMemcpyToArray(d_tableArray, 0, 0, cudaTable, tableSize.x * tableSize.y * sizeof(float), cudaMemcpyDeviceToDevice);
//	areaTex.normalized = false;
//	areaTex.filterMode = cudaFilterModeLinear;
//	areaTex.addressMode[0] = cudaAddressModeClamp;
//	areaTex.addressMode[1] = cudaAddressModeClamp;
//	// bind texture array to 2D volume
//	cudaBindTextureToArray(areaTex, d_tableArray, channelDescLut);
//	cudaBindTexture(0, lineTex, cudaLines, lineTex.channelDesc);
//	float2 alutOffset = make_float2(1.0f / m_TaDeltax, 1.0f / m_TaDeltay);
//	float  voxBase = fabsf(m_dx * m_dy);
//	cudaMemcpyToSymbol(c_alutOffset, &alutOffset, sizeof(float2));
//	cudaMemcpyToSymbol(c_voxBase, &voxBase, sizeof(float));
//
//}

//Computing ray spreading term
//void TASART2DNSL0::_cmpGeoDiv()
//{
//	std::vector<float>   geodiv(m_nr);
//	for (int i = 0; i < m_nr; ++i)
//	{
//		geodiv[i] = (1.0f / m_dr);
//
//	}
//	InitGeodiv();
//	BindGeodiv(&geodiv[0]);
//}
//Computing Lines parameters
//void TASART2DNSL0::_updateConst(float2 uv_s, float2 uv_t, float2 src, float ang)
//{
//	cudaMemcpyToSymbol(c_src, &src, sizeof(float2));
//	cudaMemcpyToSymbol(c_uv_s, &uv_s, sizeof(float2));
//	cudaMemcpyToSymbol(c_uv_t, &uv_t, sizeof(float2));
//
//	ang = fmodf(ang, 360.0f);
//	if (ang < 0.0f)
//		ang += 360.0f;
//
//	cudaMemcpyToSymbol(c_viewang, &ang, sizeof(float));
//
//}

//void TASART2DNSL0::_updateLines(float4* cudaLines, float beta)
//{
//	dim3 blk(256, 1, 1);
//	dim3 grd(iDivUp(m_nr + 1, blk.x), 1, 1);
//	update_lines_kernel << <grd, blk >> > (cudaLines, beta);
//}
// Computing pixel foot information
//void TASART2DNSL0::_init_footprint(float* footinfo)
//{
//	dim3 blk(16, 16, 1);
//	dim3 grd(iDivUp(m_nx, blk.x), iDivUp(m_ny, blk.y));
//	lut_init_foot_kernel << <grd, blk >> > (footinfo);
//}
//Forward projection Ax
//void TASART2DNSL0::_Fp_Ax(float* d_proj, float* footinfo, const float val)
//{
//	// do forward projection
//	_do_projection(d_proj, footinfo, val);
//	// normalization by scaling the projection with geometric spreading term
//	_do_proj_scaling(d_proj);
//}

//void TASART2DNSL0::_do_projection(float* d_proj, float* footinfo, const float val)
//{
//	dim3 blk(16, 16, 1);
//	dim3 grd(iDivUp(m_nx, blk.x), iDivUp(m_ny, blk.y));
//	lut_fp_kernel << <grd, blk >> > (d_proj, footinfo, val);
//}

//void TASART2DNSL0::_do_proj_scaling(float* d_proj)
//{
//	dim3 blk(256, 1, 1);
//	dim3 grd(iDivUp(m_nr, blk.x), 1, 1);
//	apply_geodiv_kernel << <grd, blk >> > (d_proj);
//}



int TASART2DNSL0::DoProjection(float* h_volume, float* h_proj, float* lut_area, float* betas)
{
	// compute semi-length of image and detector
	float m_xx = m_nx * m_dx * 0.5f;
	float m_yy = m_ny * m_dy * 0.5f;

	float m_rr = m_nr * m_dr * 0.5f;

	const int  nfoot = 5;

	// copy const variable to device
	cudaMemcpyToSymbol(c_dso, &m_dso, sizeof(float));
	cudaMemcpyToSymbol(c_dsd, &m_dsd, sizeof(float));
	cudaMemcpyToSymbol(c_nx, &m_nx, sizeof(int));
	cudaMemcpyToSymbol(c_ny, &m_ny, sizeof(int));
	cudaMemcpyToSymbol(c_dx, &m_dx, sizeof(float));
	cudaMemcpyToSymbol(c_dy, &m_dy, sizeof(float));
	cudaMemcpyToSymbol(c_offset_x, &m_offset_x, sizeof(float));
	cudaMemcpyToSymbol(c_offset_y, &m_offset_y, sizeof(float));
	cudaMemcpyToSymbol(c_nr, &m_nr, sizeof(int));
	cudaMemcpyToSymbol(c_dr, &m_dr, sizeof(float));
	cudaMemcpyToSymbol(c_offset_r, &m_offset_r, sizeof(float));
	cudaMemcpyToSymbol(c_angle_start, &m_angle_start, sizeof(float));
	cudaMemcpyToSymbol(c_angle_num, &m_na, sizeof(int));
	cudaMemcpyToSymbol(c_TaDimx, &m_TaDimx, sizeof(int));
	cudaMemcpyToSymbol(c_TaDimy, &m_TaDimy, sizeof(int));
	cudaMemcpyToSymbol(c_TaDeltax, &m_TaDeltax, sizeof(float));
	cudaMemcpyToSymbol(c_TaDeltay, &m_TaDeltay, sizeof(float));

	cudaMemcpyToSymbol(c_xx, &m_xx, sizeof(float));
	cudaMemcpyToSymbol(c_yy, &m_yy, sizeof(float));
	cudaMemcpyToSymbol(c_rr, &m_rr, sizeof(float));
	cudaMemcpyToSymbol(c_nfoot, &nfoot, sizeof(int));

	volumeSize = make_int2(m_nx, m_ny);
	tableSize = make_int2(m_TaDimx, m_TaDimy);
	geodivSize = make_int2(m_nr, 1);

	//Computing ray spreading term and bind texture
	_cmpGeoDiv();

	//Set footprint size and set table lut
	float* footinfo;
	cudaMalloc((void**)&footinfo, m_nx * m_ny * (nfoot + 3) * sizeof(float));
	cudaMemset(footinfo, 0, m_nx * m_ny * (nfoot + 3) * sizeof(float));

	InitArealut();
	thrust::device_vector<float> lut_for(lut_area, lut_area + m_TaDimx * m_TaDimy);
	cudaTable = thrust::raw_pointer_cast(&lut_for[0]);
	BindAreaLut();

	//Initialization d_proj and cudavolume
	float* d_proj;
	uint size = m_nr * m_na * sizeof(float);
	cudaMalloc((void**)&d_proj, size);
	cudaMemset(d_proj, 0, size);

	InitProjection();
	thrust::device_vector<float> x_for(h_volume, h_volume + m_nx * m_ny);
	cudavolume = thrust::raw_pointer_cast(&x_for[0]);
	Proj_BindImgToTex2D();

	//Forward projection

	for (int ia = 0; ia < m_na; ++ia)
	{
		float beta = (betas[ia] - m_angle_start) * float(DEG2RAD);
		float   cs = cosf(beta);
		float   sn = sinf(beta);
		float2 uv_t = rotateCCW_z(make_float2(0, -1), cs, sn);
		float2 uv_s = rotateCCW_z(make_float2(1, 0), cs, sn);
		float2 src = rotateCCW_z(make_float2(0, m_dso), cs, sn);

		_updateConst(uv_s, uv_t, src, beta * float(RAD2DEG));

		//Update Lines 
		_updateLines(cudaLines, beta);

		//Update pixel foot information
		_init_footprint(footinfo);

		//Projection
		_Fp_Ax(d_proj + ia * m_nr, footinfo, -1.0f);

	}
	// Copy data to host
	cudaMemcpy(h_proj, d_proj, size, cudaMemcpyDeviceToHost);

	// cuda unbind
	cudaUnbindTexture(texVolume);
	cudaUnbindTexture(geodivTex);
	cudaUnbindTexture(areaTex);
	cudaUnbindTexture(lineTex);

	//Cuda free
	cudaFree(footinfo);
	cudaFree(d_proj);
	cudaFree(cudaLines);
	cudaFreeArray(d_volumeArray);
	cudaFreeArray(d_tableArray);
	cudaFreeArray(d_geodivArray);

	return 0;
}
