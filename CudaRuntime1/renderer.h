#pragma once

#define DLLEXPORT __declspec(dllexport)

class vec3;
class hitable;

class DLLEXPORT Renderer {
public:
    int nx;
    int ny;
    int tx;
    int ty;

    hitable** d_list;
    hitable** d_world;

    vec3* fb;
    size_t fb_size;

    Renderer();
    ~Renderer();
    void render();
};