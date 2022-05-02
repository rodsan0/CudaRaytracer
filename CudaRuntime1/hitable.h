#pragma once 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include "ray.h"
class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

