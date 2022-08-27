
#define _USE_MATH_DEFINES

#include <GL/glew.h>
#include <GL/glext.h>
#include <GL/wglew.h>
#include <GL/wglext.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>

#include <chrono>

#include "Arguments.h"
#include "BottomUpBuilder.cuh"
#include "BuildWrapper.cuh"
#include "Camera.cuh"
#include "Common.cuh"
#include "FileIO.h"
#include "Input.cuh"
#include "MemoryBuffer.h"
#include "Multiblock.cuh"
#include "PerInstanceBuilder.cuh"
#include "RadixSort.cuh"
#include "SharedTaskBuilder.cuh"
#include "Tracer.cuh"
#include "Utilities.h"
#include "device_launch_parameters.h"
#include "gl.cuh"
#include "helper_math.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define REFRESH_DELAY 10  // ms

int fps_count = 0;
int fps_limit = 1;
unsigned int frame_count = 0;
float avg_fps = 0.0f;
auto prev_time = std::chrono::high_resolution_clock::now();

const unsigned int window_width = 1024;
const unsigned int window_height = 768;

GLuint viewGLTexture;
cudaGraphicsResource* viewCudaResource;

Scene scene;
Arguments args;
InputState input;
BuildInput buildInput;

MemoryBuffer<Camera>* cu_camera;
MemoryBuffer<float>* cu_depth;
MemoryBuffer<uint32_t>* cu_num_tests;

void Display();
void Keyboard(unsigned char key, int x, int y);
void KeyboardUp(unsigned char key, int x, int y);
void Mouse(int button, int state, int x, int y);
void MouseWheel(int, int, int, int);
void Motion(int x, int y);
void TimerEvent(int value);

void CreateTexture()
{
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &viewGLTexture);
    glBindTexture(GL_TEXTURE_2D, viewGLTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, window_width, window_height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register the texture with cuda
    cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D,
                                cudaGraphicsRegisterFlagsNone);
    check(cudaPeekAtLastError());
}

bool InitGL(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("GPU Acceleration Structure");
    glutDisplayFunc(Display);
    glutKeyboardFunc(Keyboard);
    glutKeyboardUpFunc(KeyboardUp);
    glutMotionFunc(Motion);
    glutMouseFunc(Mouse);
    glutMouseWheelFunc(MouseWheel);
    glutTimerFunc(REFRESH_DELAY, TimerEvent, 0);

    // initialize necessary OpenGL extensions
    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr,
                "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    wglSwapIntervalEXT(0);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, window_width, 0, window_height);

    CreateTexture();

    return true;
}

void Trace(TrianglePair* cu_triangles, Node* cu_nodes,
           MemoryBuffer<float>& cu_depth, int2 dims,
           MemoryBuffer<Camera>& camera, unsigned root, unsigned count)
{
    dim3 block_size = {32, 32, 1};
    dim3 grid_size = {dims.x / block_size.x, dims.y / block_size.y, 1};
    bool should_print = frame_count == 0;

    // map OpenGL buffer object for writing from CUDA
    cudaGraphicsMapResources(1, &viewCudaResource);
    check(cudaPeekAtLastError());

    cudaArray_t viewCudaArray;
    cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0,
                                          0);
    check(cudaPeekAtLastError());

    cudaResourceDesc viewCudaArrayResourceDesc;
    memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = viewCudaArray;

    cudaSurfaceObject_t viewCudaSurfaceObject;
    cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);
    check(cudaPeekAtLastError());

    camera.toDevice();

    run("TraceRays",
        (TraceRays<<<grid_size, block_size>>>(
            cu_triangles, cu_nodes, scene.gpu_attributes,
            scene.library.gpu_materials, scene.library.gpu_textures, camera,
            cu_num_tests->gpu(), args.render_type, viewCudaSurfaceObject, root,
            count, scene.light)));
    check(cudaPeekAtLastError());

    // unmap buffer object
    cudaGraphicsUnmapResources(1, &viewCudaResource, 0);

    check(cudaPeekAtLastError());
    check(cudaDeviceSynchronize());

    if (frame_count == 0) {
        cu_num_tests->toHost();
        printf("TraceRays number of tests %d\n", (*cu_num_tests)[0]);
    }

    // if (frameCount == 0) {
    //	printf("Writing output...");
    //	stbi_write_png("depth.png", dims.x, dims.y, 1, bytes.data(), dims.x);
    //	printf("Done\n");
    // }

    // getchar();
}

void ComputeFPS()
{
    frame_count++;
    fps_count++;

    if (fps_count == fps_limit) {
        auto curTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = curTime - prev_time;

        avg_fps = float(fps_limit / diff.count());
        fps_count = 0;
        fps_limit = (int)fmax(avg_fps, 1.f);

        prev_time = curTime;

        char fps[256];
        sprintf(fps, "CUDA AS Builder: %3.1f fps", avg_fps);
        glutSetWindowTitle(fps);
    }
}

void Display()
{
    glClear(GL_COLOR_BUFFER_BIT);

    UpdateCameraPosition(*cu_camera->data(), input);

    unsigned num_triangles = scene.triangles.size();
    unsigned rootIndex = args.build_type == kHybrid ? num_triangles * 2 + 1 : 0;
    unsigned rootCount = args.build_type == kSAH ? 1 : 2;

    if (frame_count == 0) {
        size_t bytes_required = (args.build_type == kSAH)
                                    ? SahMemoryRequirements(num_triangles)
                                    : BuMemoryRequirements(num_triangles);

        buildInput.num_triangles = num_triangles;
        cudaMalloc(&buildInput.triangles_in, sizeof(Triangle) * num_triangles);
        cudaMalloc(&buildInput.triangles_out,
                   sizeof(TrianglePair) * num_triangles * 2);
        cudaMalloc(&buildInput.scratch, bytes_required);
        cudaMalloc(
            &buildInput.nodes_out,
            sizeof(Node) * (num_triangles + max(512, NUM_BLOCKS)) * 2 * 2);
        cudaMemcpy(buildInput.triangles_in, scene.triangles.data(),
                   scene.triangles.size() * sizeof(Triangle),
                   cudaMemcpyHostToDevice);

        if (args.build_type == kSAH) {
            RunSahBuild(buildInput, args);
        } else {
            RunBottomUpBuild(buildInput, args, args.build_type == kHybrid);
        }

        Node* nodes = (Node*)malloc(sizeof(Node) * num_triangles * 2 * 2);
        cudaMemcpy(nodes, buildInput.nodes_out,
                   sizeof(Node) * num_triangles * 2 * 2,
                   cudaMemcpyDeviceToHost);

        HierarchyStats hierarchyStats = CountNodes(nodes, rootIndex, rootCount);
        printf("Hierarchy Stats:\n");
        printf("  num nodes: %d\n", hierarchyStats.numNodes);
        printf("  num tree nodes: %d\n", hierarchyStats.numTreeNodes);
        printf("  num leaf nodes: %d\n", hierarchyStats.numLeafNodes);

        VerifyHierarchy(nodes, rootIndex, rootCount);
    }

    TrianglePair* tris = buildInput.triangles_out;
    Trace(tris, buildInput.nodes_out, *cu_depth,
          make_int2(window_width, window_height), *cu_camera, rootIndex,
          rootCount);

    // Draw
    glBindTexture(GL_TEXTURE_2D, viewGLTexture);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(window_width, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(window_width, window_height);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, window_height);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();
    // glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // glDrawPixels(window_width, window_height, GL_LUMINANCE, GL_UNSIGNED_BYTE,
    // fb.data());

    glutSwapBuffers();

    ComputeFPS();

    glutPostRedisplay();
}

void MouseWheel(int button, int dir, int x, int y)
{
    Camera& camera = *cu_camera->data();

    UpdateCameraZoom(camera, dir);

    return;
}

void Keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    Camera& camera = *cu_camera->data();

    switch (key) {
        case 'w':
            input.key_pressed_w = true;
            break;
        case 's':
            input.key_pressed_s = true;
            break;
        case 'a':
            input.key_pressed_a = true;
            break;
        case 'd':
            input.key_pressed_d = true;
            break;
        case 'q':
            input.key_pressed_q = true;
            break;
        case 'e':
            input.key_pressed_e = true;
            break;
        case ' ':
            input.key_pressed_space = true;
            break;
        case 'm':
            args.render_type =
                RenderType((args.render_type + 1) % RenderType::kCount);
            break;
        case (27):
            glutDestroyWindow(glutGetWindow());
            return;
    }
}

void KeyboardUp(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) {
        case 'w':
            input.key_pressed_w = false;
            break;
        case 's':
            input.key_pressed_s = false;
            break;
        case 'a':
            input.key_pressed_a = false;
            break;
        case 'd':
            input.key_pressed_d = false;
            break;
        case 'q':
            input.key_pressed_q = false;
            break;
        case 'e':
            input.key_pressed_e = false;
            break;
        case ' ':
            input.key_pressed_space = false;
            break;
    }
}

void Mouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            input.mouse_down = true;
            input.prev_x = x;
            input.prev_y = y;
        }
        if (state == GLUT_UP) {
            input.mouse_down = false;
        }
    }
}

void Motion(int x, int y)
{
    Camera& camera = *cu_camera->data();

    if (input.mouse_down) {
        int dx = x - input.prev_x;
        int dy = y - input.prev_y;

        UpdateCameraLookDelta(camera, dx, dy);

        input.prev_x = x;
        input.prev_y = y;

        UpdateCamera(*cu_camera->data());
    }
}

void TimerEvent(int value) {}

unsigned CountNodes(Node* nodes, Triangle* triangles, unsigned parent,
                    unsigned index, unsigned depth)
{
    unsigned result = 2;

    // if (index != 0) assert(nodes[index].parent == parent && nodes[index +
    // 1].parent == parent);

    if (nodes[index].type == ChildType_Box) {
        assert(nodes[index].count == 2);
        result +=
            CountNodes(nodes, triangles, index, nodes[index].child, depth + 1);
    }
    if (nodes[index + 1].type == ChildType_Box) {
        assert(nodes[index + 1].count == 2);
        result += CountNodes(nodes, triangles, index + 1,
                             nodes[index + 1].child, depth + 1);
    }

    return result;
}

void Scene::CopyToDevice()
{
    check(cudaMalloc(&gpu_attributes, attributes.size() * sizeof(Attributes)));
    check(cudaMemcpy(gpu_attributes, attributes.data(),
                     attributes.size() * sizeof(Attributes),
                     cudaMemcpyHostToDevice));
    library.CopyToDevice();
}

void Library::CopyToDevice()
{
    // Allocate textures and materials arrays on the device
    check(cudaMalloc(&gpu_textures,
                     scene.library.textures.size() * sizeof(Texture)));
    check(cudaMalloc(&gpu_materials,
                     scene.library.materials.size() * sizeof(Material)));

    // For each texture object allocate and copy the texture data to the device
    for (unsigned i = 0; i < textures.size(); i++) {
        Texture& texture = scene.library.GetTexture(i);
        for (unsigned j = 0; j <= texture.max_lod; j++) {
            size_t size = 4 * texture.sizes[j].x * texture.sizes[j].y;
            check(cudaMalloc(&texture.gpu_mips[j], size));
            check(cudaMemcpy(texture.gpu_mips[j], texture.mips[j], size,
                             cudaMemcpyHostToDevice));
        }
    }

    // Copy the textre and material arrays to the device
    check(cudaMemcpy(gpu_materials, materials.data(),
                     materials.size() * sizeof(Material),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(gpu_textures, textures.data(),
                     textures.size() * sizeof(Texture),
                     cudaMemcpyHostToDevice));
}

int main(int argc, char** argv)
{
    InitGL(&argc, argv);

    {
        // Parse options from commandline
        args = ParseCmd(argc, argv);

        // Load the triangles from file
        scene = LoadOBJFromFile(g_filename);

        cu_camera = new MemoryBuffer<Camera>(1);
        cu_depth = new MemoryBuffer<float>(window_width * window_height);
        cu_num_tests = new MemoryBuffer<uint32_t>(1);

        cu_num_tests->toDevice();

        InitialiseCamera(*cu_camera->data(), scene.aabb);

        cu_camera->toDevice();

        scene.CopyToDevice();

        glutMainLoop();
    }
    // Exit scope to force mem buffer destructors before device reset
    cudaDeviceReset();
}
