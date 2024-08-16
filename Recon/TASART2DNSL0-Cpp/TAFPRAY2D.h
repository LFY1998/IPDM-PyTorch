#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <string>
#ifndef SQR
#define SQR(x) (x)*(x)
#endif

#ifndef RAD2DEG 
#define RAD2DEG 180.0f/(float)M_PI
#endif

#ifndef DEG2RAD
#define DEG2RAD (float)M_PI/180.0f
#endif

typedef struct __parameters
{
	float dso;
	float dsd;
	int nx;
	int ny;
	float dx;
	float dy;
	float offset_x;
	float offset_y;
	int nr;
	float dr;
	float offset_r;
	float angle_start;
	int na;
	int TaDimx;
	int TaDimy;
	float TaDeltax;
	float TaDeltay;
}Parameters;


class  TAFPRAY2D
{
public:
	TAFPRAY2D(void);
	~TAFPRAY2D(void);
public:
	void SetGeometry(Parameters params);
	int  DoProjection(float* h_volume, float* h_proj, float* lut_area, float* betas);

private:
	void InitArealut();
	void BindAreaLut();
	void _cmpGeoDiv();
	void _updateConst(float2 uv_s, float2 uv_t, float2 src, float ang);
	void _updateLines(float4 *cudaLines, float beta);
	void _init_footprint(float *footinfo);
	void _Fp_Ax(float *d_proj, float *footinfo, const float val);
	void _do_projection(float *d_proj, float *footinfo, const float val);
	void _do_proj_scaling(float *d_proj);

private:
	float m_dso, m_dsd, m_dx, m_dy, m_dr, m_TaDeltax, m_TaDeltay;
	float m_offset_r, m_offset_x, m_offset_y, m_angle_start;
	int   m_nx, m_ny, m_nr, m_na, m_TaDimx, m_TaDimy;

};