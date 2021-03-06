#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <float.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "renderer.h"

#define RND (curand_uniform(&local_rand_state))
#define DLLEXPORT __declspec(dllexport)

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

DLLEXPORT void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        cudaDeviceReset();
        exit(99);
    }
}

DLLEXPORT __device__ vec3 color(const ray& r, hitable **world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = normalize(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    // hit nothing
    return vec3(0.0, 0.0, 0.0);
}

DLLEXPORT __global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

DLLEXPORT __global__ void render_hidden(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray_random(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

DLLEXPORT __global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

DLLEXPORT __global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1001.0, -1), 1000, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        d_list[1] = new sphere(vec3(3, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        d_list[2] = new sphere(vec3(-3, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[3] = new sphere(vec3(3, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        d_list[4] = new sphere(vec3(-3, 1, 0), 1.0, new dielectric(1.5));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 5);

        vec3 lookfrom(13, 2, 0);
        vec3 lookat(0, 0, 0);

        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.05;
        *d_camera = new camera(
            lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus
        );
    }
}

DLLEXPORT __global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < 3; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

DLLEXPORT __global__ void update_world(float time, hitable** d_list) {
    ((sphere*)d_list[1])->center = vec3(3*cos(time), .5*sin(5*time)+1, 3 * sin(time));
    ((sphere*)d_list[2])->center = vec3(-3 * cos(time), .5*cos(5*time)+1, -3 * sin(time));
    ((sphere*)d_list[3])->center = vec3(3 * sin(-1*time), .5 * sin(5 * time) + 1, 3 * cos(time));
    ((sphere*)d_list[4])->center = vec3(-3 * sin(-1*time), .5 * cos(5 * time) + 1, -3 * cos(time));
}

DLLEXPORT __global__ void update_camera(const float time, camera** camera, const double aspect_ratio, const Renderer::Keys keys) {
    // make copies for ease of use
    vec3 origin = camera[0]->origin;
    vec3 lookat = camera[0]->lookat;
    vec3 up = camera[0]->up;

    // determine viewing direction
    vec3 dir = normalize(lookat - origin);
    // determine z axis direction
    vec3 z = normalize(cross(dir, up));

    // number of steps in a rotation
    const size_t rot_steps = 100;
    const double angle = 2 * M_PI / rot_steps;

    // handle translations
    if (keys.up) {
        origin += dir;
        lookat += dir;
    }
    if (keys.down) {
        origin -= dir;
        lookat -= dir;
    }
    if (keys.left) {
        origin -= z;
        lookat -= z;
    }
    if (keys.right) {
        origin += z;
        lookat += z;
    }
    if (keys.space) {
        origin += up;
        lookat += up;
    }
    if (keys.shift) {
        origin -= up;
        lookat -= up;
    }
   
    // handle rotations
    if (keys.w) {
        lookat = vec3(
            origin.x() + (lookat.x() - origin.x()) * cos(-angle) - (lookat.y() - origin.y()) * sin(-angle),
            origin.y() + (lookat.x() - origin.x()) * sin(-angle) + (lookat.y() - origin.y()) * cos(-angle),
            lookat.z()
        );
    }
    if (keys.a) {
        lookat = vec3(
            origin.x() + (lookat.x() - origin.x()) * cos(angle) + (lookat.z() - origin.z()) * sin(angle),
            lookat.y(),
            origin.z() - (lookat.x() - origin.x()) * sin(angle) + (lookat.z() - origin.z()) * cos(angle)
        );
    }
    if (keys.s) {
        lookat = vec3(
            origin.x() + (lookat.x() - origin.x()) * cos(angle) - (lookat.y() - origin.y()) * sin(angle),
            origin.y() + (lookat.x() - origin.x()) * sin(angle) + (lookat.y() - origin.y()) * cos(angle),
            lookat.z()
        );
    }
    if (keys.d) {
        lookat = vec3(
            origin.x() + (lookat.x() - origin.x()) * cos(-angle) + (lookat.z() - origin.z()) * sin(-angle),
            lookat.y(),
            origin.z() - (lookat.x() - origin.x()) * sin(-angle) + (lookat.z() - origin.z()) * cos(-angle)
        );
    }

    // write copies back
    camera[0]->lookat = lookat;
    camera[0]->origin = origin;

    // determine focus distance and aperture
    const float focus_distance = (origin - lookat).length();
    const float aperture = 0.05;

    camera[0]->update_camera(
        origin,
        lookat,
        camera[0]->up,
        30.0,
        aspect_ratio,
        aperture,
        focus_distance
    );
}

void Renderer::Render_Init() {
    tx = 4;
    ty = 128;

    int num_pixels = nx * ny;
    fb_size = num_pixels * 3 * sizeof(float);

    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    int num_hitables = 5;
    checkCudaErrors(cudaMallocManaged((void**)&d_list, num_hitables * sizeof(hitable*)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());

}

Renderer::~Renderer() {
    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}

void Renderer::render() {
    // render buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render_hidden<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    update_world<<<1, 1>>>(i, d_list);
    checkCudaErrors(cudaDeviceSynchronize());
    update_camera<<<1, 1>>>(i, d_camera, float(nx) / float(ny), keys);
    checkCudaErrors(cudaDeviceSynchronize());
    i += 1.f/15;
}
