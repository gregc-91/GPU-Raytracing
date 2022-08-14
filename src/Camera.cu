#define _USE_MATH_DEFINES

#include <math.h>

#include "Common.cuh"
#include "Input.cuh"

void UpdateCamera(Camera& camera)
{
    if (camera.pitch > M_PI / 2) {
        camera.pitch = float(M_PI / 2 - 0.0001f);
    } else if (camera.pitch < -M_PI / 2) {
        camera.pitch = float(-M_PI / 2 + 0.0001f);
    }

    float pitch = camera.pitch;
    float yaw = camera.yaw;

    camera.w.x = -sin(yaw) * cos(pitch);
    camera.w.y = -sin(pitch);
    camera.w.z = cos(yaw) * cos(pitch);
    camera.w = normalize(camera.w);

    camera.u = cross(camera.w, make_float3(0, 1, 0));
    camera.u = normalize(camera.u);

    camera.v = cross(camera.w, camera.u);
    camera.v = normalize(camera.v);
}

void UpdateCameraPosition(Camera& camera, InputState input)
{
    if (input.key_pressed_w)
        camera.position = camera.position + camera.w * camera.scale * 0.25;
    if (input.key_pressed_s)
        camera.position = camera.position - camera.w * camera.scale * 0.25;
    if (input.key_pressed_a)
        camera.position = camera.position - camera.u * camera.scale * 0.25;
    if (input.key_pressed_d)
        camera.position = camera.position + camera.u * camera.scale * 0.25;
    if (input.key_pressed_q)
        camera.position = camera.position - camera.v * camera.scale * 0.25;
    if (input.key_pressed_e)
        camera.position = camera.position + camera.v * camera.scale * 0.25;
}

void UpdateCameraLookDelta(Camera& camera, float dx, float dy)
{
    camera.yaw += dx * 0.01f;
    camera.pitch += dy * 0.01f;
}

void UpdateCameraZoom(Camera& camera, int dir)
{
    if (dir > 0) {
        camera.position = camera.position + camera.w * camera.scale;
    } else {
        camera.position = camera.position - camera.w * camera.scale;
    }
}

void InitialiseCamera(Camera& camera, AABB scene_aabb)
{
    camera.position = {0, 0, 0};
    camera.pitch = 0;
    camera.w = {0, 0, 1};
    camera.yaw = 0;
    camera.u = {-1, 0, 0};
    camera.scale = 1.0f;
    camera.v = {0, -1, 0};
    camera.max_depth = 1;

    float3 centre = ((scene_aabb.max + scene_aabb.min) * 0.5f);
    float3 length = scene_aabb.max - scene_aabb.min;

    camera.scale = length.z / 10.0f;
    camera.position =
        make_float3(centre.x, centre.y, scene_aabb.min.z - length.z);
    camera.max_depth = max(max(length.x, length.y), length.z) * 1.5f;

	camera.position = scene_aabb.max * 1.2f;

	//camera.w = (scene_aabb.max + scene_aabb.min)*0.5f - scene_aabb.max;
	//camera.w = normalize(camera.w);
	//camera.pitch = asin(-camera.w.y);
	//camera.yaw = acos(camera.w.z / cos(camera.pitch));

	camera.position = centre;
	camera.yaw = M_PI / 2;

	UpdateCamera(camera);
}