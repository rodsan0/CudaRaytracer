#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

class vec3;
class hitable;
#define DLLEXPORT __declspec(dllexport)

class DLLEXPORT Renderer {
public:
    int nx = 1200;
    int ny = 800;
    int tx;
    int ty;
    int ns = 10;
    float i = 0;
    hitable** d_list;
    hitable** d_world;
    camera** d_camera;

    struct Keys {
        bool left;
        bool right;
        bool up;
        bool down;

        bool w;
        bool s;
        bool a;
        bool d;

        bool space;
        bool shift;
    } keys;

    vec3* fb = nullptr;
    size_t fb_size;

    curandState* d_rand_state;
    curandState* d_rand_state2;

    Renderer() = default;
    void Render_Init();
    ~Renderer();
    void render();
};