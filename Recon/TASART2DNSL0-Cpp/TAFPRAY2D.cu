#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "TAFPRAY2D.h"
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

typedef unsigned int  uint;

// Volume size setting
int2 volumeSize;
cudaArray *d_volumeArray = 0;
float *cudavolume;

//Table size setting
int2 tableSize;
cudaArray *d_tableArray = 0;
float *cudaTable;

//GeoDiv size setting
int2 geodivSize;
cudaArray *d_geodivArray = 0;

//Line size setting
float4 *cudaLines;

// 2D texture for projection
texture<float, cudaTextureType2D, cudaReadModeElementType> texVolume;
texture<float, cudaTextureType2D, cudaReadModeElementType> areaTex;
texture<float, cudaTextureType2D, cudaReadModeElementType> geodivTex;
texture<float4, cudaTextureType1D, cudaReadModeElementType> lineTex;

//CUDA  constant parameters definition
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

struct Ray
{
	float2  o;  // ray origin site
	float2  d;  // ray  direction
};
// help functions
int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//rotate around Z 
float2 rotateCCW_z(float2 v, float cs, float sn)
{
	return make_float2(v.x*cs - v.y*sn, v.x*sn + v.y*cs);
}

extern "C"
void InitProjection()
{
	//create 2D array
	cudaChannelFormatDesc chaDesc1 = cudaCreateChannelDesc<float>();

	// allocate 2D array to bind 2D texture in cuda
	cudaMallocArray(&d_volumeArray, &chaDesc1, volumeSize.x, volumeSize.y);

}

extern "C"
void BindImgToTex2D()
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
		pos = L.y*xy.x + L.z*xy.y + L.w;

	float value;
	value = tex2D(areaTex,
		fabsf(pos)*c_alutOffset.x + 0.5f,
		ang*c_alutOffset.y + 0.5f);

	return pos < 0.0f ? c_voxBase - value : value;

}

__global__ static void update_lines_kernel(float4 *lines, float beta)
{
	int is = blockIdx.x*blockDim.x + threadIdx.x;
	if (is >= c_nlines)
		return;

	float s0 = -c_rr + c_offset_r*c_dr;
	float gamma = s0 + is*c_dr;

	float2 P1 = c_src + make_float2(c_dsd*sin(beta + gamma), -c_dsd*cos(beta + gamma));
	float2 rayvec = P1 - c_src;

	float ang = atan2f(rayvec.y, rayvec.x) * (360.0f / (2.0f*(float)M_PI));
	if (ang < 0.0f)
		ang += 360.0f;

	float A = P1.y - c_src.y;
	float B = c_src.x - P1.x;
	float C = P1.x*c_src.y - c_src.x*P1.y;
	float Z = sqrtf(A*A + B*B);

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
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= c_nx || iy >= c_ny)
		return;

	int stride = c_nx*c_ny;
	footinfo += (iy*c_nx + ix);

	int nfoot2 = c_nfoot / 2;
	float2 xy = make_float2(((ix + 0.5) * c_dx - c_xx*1.0f + c_offset_x),
		((iy + 0.5) * c_dy - c_yy*1.0f + c_offset_y));

	float div = sqrt(SQR(xy.x - c_src.x) + SQR(xy.y - c_src.y));
	*footinfo = div; footinfo += stride;

	float mag = c_dsd / sqrt(SQR(dot(c_uv_s, xy)) + SQR((dot(c_uv_t, xy) + c_dso)));
	*footinfo = mag;
	footinfo += stride;

	float gamma = atan(dot(c_uv_s, xy) / (dot(c_uv_t, xy) + c_dso));
	int   s_bin = (int)floorf(gamma / c_dr + 0.5f*(c_nr - 1) - c_offset_r) - nfoot2;

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

__global__ static void lut_fp_kernel(float *d_proj, float *footinfo, const float val)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= c_nx || iy >= c_ny)
		return;

	int stride = c_nx*c_ny;
	footinfo += (iy*c_nx + ix);

	float shdata[8];
	float* p_div = shdata + 0;
	float* p_sbin = shdata + 2;
	float* p_foot = shdata + 3;

	for (int idx = 0; idx < 3 + c_nfoot; ++idx)
	{
		shdata[idx] = footinfo[idx * stride];

	}
	__syncthreads();


	float att = (val>0.0f) ? val : tex2D(texVolume, ix + 0.5f, iy + 0.5f);

	if (att == 0.0f)
		return;

	float div = att / (*p_div);
	int is = int(*p_sbin);
	for (int ifoot = 0; ifoot<c_nfoot; ++is, ++ifoot)
	{
		float area = p_foot[ifoot];

		if (is < 0 || is >= c_nr || area <= 0.0f || div == 0.0f)
			continue;

		atomicAdd(d_proj + is, div*area);
	}
}

__global__ static void apply_geodiv_kernel(float* proj)
{
	int is = blockIdx.x*blockDim.x + threadIdx.x;

	if (is >= c_nr)
		return;

	proj[is] *= tex2D(geodivTex, is + 0.5f, 0.5f);
}

//////////////////
/////// class member functions

void TAFPRAY2D::InitArealut()
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


void TAFPRAY2D::BindAreaLut()
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
void TAFPRAY2D::_cmpGeoDiv()
{
	std::vector<float>   geodiv(m_nr);
	for (int i = 0; i<m_nr; ++i)
	{
		geodiv[i] = (1.0f / m_dr);

	}
	InitGeodiv();
	BindGeodiv(&geodiv[0]);
}
//Computing Lines parameters
void TAFPRAY2D::_updateConst(float2 uv_s, float2 uv_t, float2 src, float ang)
{
	cudaMemcpyToSymbol(c_src, &src, sizeof(float2));
	cudaMemcpyToSymbol(c_uv_s, &uv_s, sizeof(float2));
	cudaMemcpyToSymbol(c_uv_t, &uv_t, sizeof(float2));

	ang = fmodf(ang, 360.0f);
	if (ang < 0.0f)
		ang += 360.0f;

	cudaMemcpyToSymbol(c_viewang, &ang, sizeof(float));

}

void TAFPRAY2D::_updateLines(float4 *cudaLines, float beta)
{
	dim3 blk(256, 1, 1);
	dim3 grd(iDivUp(m_nr + 1, blk.x), 1, 1);
	update_lines_kernel << <grd, blk >> >(cudaLines, beta);
}
// Computing pixel foot information
void TAFPRAY2D::_init_footprint(float *footinfo)
{
	dim3 blk(16, 16, 1);
	dim3 grd(iDivUp(m_nx, blk.x), iDivUp(m_ny, blk.y));
	lut_init_foot_kernel << <grd, blk >> >(footinfo);
}
//Forward projection Ax
void TAFPRAY2D::_Fp_Ax(float *d_proj, float *footinfo, const float val)
{
	// do forward projection
	_do_projection(d_proj, footinfo, val);
	// normalization by scaling the projection with geometric spreading term
	_do_proj_scaling(d_proj);
}

void TAFPRAY2D::_do_projection(float *d_proj, float *footinfo, const float val)
{
	dim3 blk(16, 16, 1);
	dim3 grd(iDivUp(m_nx, blk.x), iDivUp(m_ny, blk.y));
	lut_fp_kernel << <grd, blk >> >(d_proj, footinfo, val);
}

void TAFPRAY2D::_do_proj_scaling(float *d_proj)
{
	dim3 blk(256, 1, 1);
	dim3 grd(iDivUp(m_nr, blk.x), 1, 1);
	apply_geodiv_kernel << <grd, blk >> >(d_proj);
}




TAFPRAY2D::TAFPRAY2D(void)
{

}

TAFPRAY2D::~TAFPRAY2D(void)
{

}

void TAFPRAY2D::SetGeometry(Parameters params)
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

int TAFPRAY2D::DoProjection(float* h_volume, float* h_proj, float* lut_area, float* betas)
{
	// compute semi-length of image and detector
	float m_xx = m_nx * m_dx *0.5f;
	float m_yy = m_ny * m_dy *0.5f;

	float m_rr = m_nr * m_dr *0.5f;

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
	float *footinfo;
	cudaMalloc((void**)&footinfo, m_nx * m_ny * (nfoot + 3) * sizeof(float));
	cudaMemset(footinfo, 0, m_nx * m_ny * (nfoot + 3) * sizeof(float));

	InitArealut();
	thrust::device_vector<float> lut_for(lut_area, lut_area + m_TaDimx * m_TaDimy);
	cudaTable = thrust::raw_pointer_cast(&lut_for[0]);
	BindAreaLut();

	//Initialization d_proj and cudavolume
	float *d_proj;
	uint size = m_nr * m_na * sizeof(float);
	cudaMalloc((void **)&d_proj, size);
	cudaMemset(d_proj, 0, size);

	InitProjection();
	thrust::device_vector<float> x_for(h_volume, h_volume + m_nx * m_ny);
	cudavolume = thrust::raw_pointer_cast(&x_for[0]);
	BindImgToTex2D();

	//Forward projection

	for (int ia = 0; ia < m_na; ++ia)
	{
		float beta = (betas[ia] - m_angle_start)*float(DEG2RAD);
		float   cs = cosf(beta);
		float   sn = sinf(beta);
		float2 uv_t = rotateCCW_z(make_float2(0, -1), cs, sn);
		float2 uv_s = rotateCCW_z(make_float2(1, 0), cs, sn);
		float2 src = rotateCCW_z(make_float2(0, m_dso), cs, sn);

		_updateConst(uv_s, uv_t, src, beta*float(RAD2DEG));

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
