#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "Common.cuh"
#include "Input.cuh"

void UpdateCamera(Camera& camera);

void UpdateCameraPosition(Camera& camera, InputState input);

void UpdateCameraLookDelta(Camera& camera, float dx, float dy);

void UpdateCameraZoom(Camera& camera, int dir);

void InitialiseCamera(Camera& camera, AABB scene_aabb);

#endif