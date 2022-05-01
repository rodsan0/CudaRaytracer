#pragma once

#define DLLEXPORT __declspec(dllexport)

class vec3;
class hitable;
class camera;
class curandState;


class Renderer {
public:
    int nx = 1200;
    int ny = 800;
    int tx;
    int ty;
    int ns = 1;
    int i = 0;
    hitable** d_list;
    hitable** d_world;
    camera** d_camera;

    vec3* fb = nullptr;
    size_t fb_size;

    curandState* d_rand_state;
    curandState* d_rand_state2;

    Renderer();
    void Render_Init();
    void Renderer_End();
    ~Renderer();
    void render();
};