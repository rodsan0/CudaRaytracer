#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"
#include "ray.h"
#include "hitable.h"

#define DLLEXPORT __declspec(dllexport)

DLLEXPORT cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

DLLEXPORT __device__ vec3 color(const ray& r, hitable** world);

DLLEXPORT __global__ void render(vec3* fb, int max_x, int max_y,
    vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
    hitable** world);