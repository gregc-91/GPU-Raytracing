
#include "FileIO.h"

#include <assert.h>
#include <stdlib.h>

#include <fstream>
#include <sstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define MAX_LENGTH 256
#define WHITESPACE " \t\n"

// Check if a file exists
bool FileExists(std::string filename)
{
    FILE* fp = fopen(filename.c_str(), "r");
    if (fp) {
        fclose(fp);
        return true;
    } else {
        return false;
    }
}

std::string BaseDirectory(std::string filename)
{
    size_t found = filename.find_last_of("\\/");
    if (found == std::string::npos) return std::string("");
    return filename.substr(0, found);
}

static std::string __w;
static std::vector<std::string> __r;
std::vector<std::string> Split(std::string s, char delim = ' ')
{
    std::istringstream ss(s);
    __r.clear();

    while (std::getline(ss, __w, delim)) {
        __r.push_back(__w);
    }
    return __r;
}

std::string ltrim(const std::string& s)
{
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start == std::string::npos) ? "" : s.substr(start);
}

// Faster than builtin strtok but maybe less robust
char strtok_buf[256];
char* gstrtok(char** s, const char* delims)
{
    char* begin = *s;
    for (; **s != '\0'; (*s)++) {
        for (const char* d = delims; *d != '\0'; ++d) {
            if (**s == *d) {
                memcpy(strtok_buf, begin, *s - begin);
                strtok_buf[*s - begin] = '\0';
                (*s)++;
                return strtok_buf;
            }
        }
    }

    return NULL;
}

float3 SetupLight(std::string obj_name, AABB& aabb)
{
    float3 result = aabb.Centre();
    std::string lights_name = BaseDirectory(obj_name) + "/light.txt";
    if (FileExists(lights_name)) {
        FILE* lights_fp = fopen(lights_name.c_str(), "r");
        assert(lights_fp);
        int res =
            fscanf(lights_fp, "%f %f %f", &result.x, &result.y, &result.z);
        fclose(lights_fp);
    }
    return result;
}

float3 GenerateNormal(const Triangle& tri)
{
    float3 e1 = tri.v1 - tri.v0;
    float3 e2 = tri.v2 - tri.v1;
    return normalize(cross(e1, e2));
}

uchar4 Material::ReadTexel(Texture* textures, int2 coord, int lod)
{
    Texture& tex = textures[texture];
    return tex.ReadTexel(coord, lod);
}

void Material::WriteTexel(Texture* textures, int2 coord, int lod, uchar4 val)
{
    Texture& tex = textures[texture];
    tex.WriteTexel(coord, lod, val);
}

uchar4 Texture::ReadTexel(int2 coord, int lod)
{
    int2 size = sizes[lod];
    int2 clamped = clamp(coord, make_int2(0), size - 1);
    return mips[lod][clamped.y * size.x + clamped.x];
}

void Texture::WriteTexel(int2 coord, int lod, uchar4 val)
{
    int2 size = sizes[lod];
    int2 clamped = clamp(coord, make_int2(0), size - 1);
    mips[lod][clamped.y * size.x + clamped.x] = val;
}

void Texture::GenerateLODs()
{
    uint32_t lod = 0;
    while (sizes[lod].x > 1 || sizes[lod].y > 1) {
        sizes[lod + 1] =
            make_int2((sizes[lod].x + 1) / 2, (sizes[lod].y + 1) / 2);
        mips[lod + 1] =
            (uchar4*)malloc(sizes[lod + 1].x * sizes[lod + 1].y * 4);

        for (unsigned j = 0; j < sizes[lod + 1].y; j++)
            for (unsigned i = 0; i < sizes[lod + 1].x; i++) {
                // Sample the 4 pixels, the ReadTexel function handles clamping
                uchar4 a = ReadTexel(make_int2(i * 2 + 0, j * 2 + 0), lod);
                uchar4 b = ReadTexel(make_int2(i * 2 + 1, j * 2 + 0), lod);
                uchar4 c = ReadTexel(make_int2(i * 2 + 0, j * 2 + 1), lod);
                uchar4 d = ReadTexel(make_int2(i * 2 + 1, j * 2 + 1), lod);

                uchar4 avg = make_uchar4((make_float4(a) + make_float4(b) +
                                          make_float4(c) + make_float4(d)) *
                                         0.25f);

                WriteTexel(make_int2(i, j), lod + 1, avg);
            }
        lod++;
    }
    max_lod = lod;
    for (unsigned i = max_lod + 1; i < NUM_LODS; i++) {
        mips[i] = NULL;
    }
}

bool Material::HasTexture() { return texture != -1; }

Material::~Material()
{
    // Todo: This causes vector resizes to double-free
    // Can we use unique pointers or something to handle owndership?
    // if (texture) stbi_image_free(texture);
}

void Library::AddMaterial(const std::string name)
{
    name_to_mat[name] = materials.size();
    materials.push_back(Material(name));
}

int32_t Library::AddTexture(const std::string name)
{
    int2 dims = {0, 0};
    int channels = 0;

    // Check if the texture is already loaded
    if (GetTextureId(name) == -1) {
        printf("Loading %s\n", name.c_str());
        uchar4* mip0 =
            (uchar4*)stbi_load(name.c_str(), &dims.x, &dims.y, &channels, 4);
        name_to_tex[name] = textures.size();
        textures.push_back(Texture(name, mip0, dims));
        textures.back().GenerateLODs();
        return textures.size() - 1;
    } else {
        return GetTextureId(name);
    }
}

int32_t Library::GetMaterialId(const std::string name)
{
    auto it = name_to_mat.find(name);
    if (it != name_to_mat.end())
        return it->second;
    else
        return -1;
}

Material& Library::GetMaterial(std::string name)
{
    int32_t id = GetMaterialId(name);
    assert(id != -1);
    return materials[id];
}

Material& Library::GetMaterial(uint32_t i) { return materials[i]; }

int32_t Library::GetTextureId(std::string name)
{
    auto it = name_to_tex.find(name);
    if (it != name_to_tex.end())
        return it->second;
    else
        return -1;
}

Texture& Library::GetTexture(uint32_t i) { return textures[i]; }

Texture& Library::GetTextureFromMat(uint32_t i)
{
    Material& mat = materials[i];
    assert(mat.texture != -1);
    return textures[mat.texture];
}

Library LoadMTLFromFile(const std::string filename)
{
    Library library;

    printf("Loading MTL file: %s\n", filename.c_str());

    std::ifstream fs(filename);

    std::string line;
    while (std::getline(fs, line)) {
        line = ltrim(line);

        if (!line.length()) continue;
        auto tokens = Split(line);
        if (!tokens.size()) continue;

        tokens[0] = ltrim(tokens[0]);

        if (tokens[0] == "newmtl") {
            assert(tokens.size() > 1);
            library.AddMaterial(tokens[1]);
        } else if (tokens[0] == "Ka" || tokens[0] == "Kd" ||
                   tokens[0] == "Ks") {
            assert(library.materials.size());
            assert(tokens.size() > 1);

            float3 vals;
            if (tokens.size() >= 4) {
                vals = make_float3(std::stof(tokens[1]), std::stof(tokens[2]),
                                   std::stof(tokens[3]));
            } else
                vals = make_float3(std::stof(tokens[1]));

            if (tokens[0] == "Ka") library.materials.back().ambient = vals;
            if (tokens[0] == "Kd") library.materials.back().diffuse = vals;
            if (tokens[0] == "Ks") library.materials.back().specular = vals;
        } else if (tokens[0] == "map_Kd") {
            assert(tokens.size() > 1);

            // Load the diffuse texture
            std::string texture_name(BaseDirectory(filename) + "/" + tokens[1]);
            int32_t texture_index = library.AddTexture(texture_name);
            library.materials.back().texture = texture_index;

        } else if (tokens[0] == "bump") {
            assert(tokens.size() > 1);

            std::string texture_name(BaseDirectory(filename) + "/" + tokens[1]);
            int32_t texture_index = library.AddTexture(texture_name);
            library.materials.back().bump = texture_index;

        } else if (tokens[0] == "map_Disp") {
            assert(tokens.size() > 1);

            std::string texture_name(BaseDirectory(filename) + "/" + tokens[1]);
            int32_t texture_index = library.AddTexture(texture_name);
            library.materials.back().disp = texture_index;

        } else if (tokens[0] == "Ns") {
            assert(tokens.size() > 1);
            library.materials.back().specular_exp = std::stof(tokens[1]);
        }
    }

    return library;
}

struct Indices {
    int v;
    int t;
    int n;
};

std::vector<Indices> _indices;

std::vector<Indices> GetIndices(std::string s, uint32_t num_verts,
                                uint32_t num_uvs, uint32_t num_normals)
{
    _indices.clear();
    auto triplets = Split(s);
    for (auto trip : triplets) {
        if (trip == "\n") continue;

        auto i = Split(trip, '/');
        Indices idx = {-1, -1, -1};

        if (i.size() > 0) {
            int f = atoi(i[0].c_str());
            idx.v = f < 0 ? f + num_verts : f - 1;
        }
        if (i.size() > 1 && i[1].length()) {
            int f = atoi(i[1].c_str());
            idx.t = f < 0 ? f + num_uvs : f - 1;
        }
        if (i.size() > 2 && i[2].length()) {
            int f = atoi(i[2].c_str());
            idx.n = f < 0 ? f + num_normals : f - 1;
        }

        _indices.push_back(idx);
    }

    return _indices;
}

Scene LoadOBJFromFile(const std::string filename)
{
    FILE* fp;
    char* line = (char*)malloc(MAX_LENGTH);
    char* token = NULL;
    char* a[4];
    unsigned idx[4];
    uint32_t vertex_count = 0;
    uint32_t normal_count = 0;
    uint32_t current_cmp_count = 0;
    int32_t current_material_id = -1;

    Scene scene;
    scene.triangles.reserve(100000);
    scene.attributes.reserve(100000);

    std::vector<float3> vertex_buffer;
    std::vector<Indices> index_buffer;
    std::vector<float3> normals_buffer;
    std::vector<float2> uv_buffer;

    std::vector<Indices> idxs;

    index_buffer.reserve(100000);
    vertex_buffer.reserve(100000);
    uv_buffer.reserve(100000);

    fopen_s(&fp, filename.c_str(), "r");
    if (fp == NULL) {
        fprintf(stderr, "Can't open OBJ file %s!\n", filename.c_str());
        exit(1);
    }

    while (fgets(line, MAX_LENGTH, fp)) {
        char* line_copy = line;
        token = gstrtok(&line_copy, " \t\n\r");

        if (token == NULL || strcmp(token, "#") == 0)
            continue;

        else if (strcmp(token, "mtllib") == 0) {
            std::string mtl_filename(gstrtok(&line_copy, WHITESPACE));

            // The filename may be relative to the obj, so create the full path
            if (!FileExists(mtl_filename)) {
                mtl_filename = BaseDirectory(filename) + "/" + mtl_filename;
            }

            scene.library = LoadMTLFromFile(mtl_filename);
        } else if (strcmp(token, "usemtl") == 0) {
            token = gstrtok(&line_copy, WHITESPACE);
            current_material_id =
                scene.library.GetMaterialId(std::string(token));
        } else if (strcmp(token, "v") == 0) {
            token = gstrtok(&line_copy, WHITESPACE);
            if (token[0] == '\0') {
                token = gstrtok(&line_copy, WHITESPACE);
            }

            vertex_buffer.push_back(
                {(float)atof(token),
                 (float)atof(gstrtok(&line_copy, WHITESPACE)),
                 (float)atof(gstrtok(&line_copy, WHITESPACE))});
        } else if (strcmp(token, "vt") == 0) {
            float2 uv = {float(atof(gstrtok(&line_copy, WHITESPACE))),
                         float(atof(gstrtok(&line_copy, WHITESPACE)))};

            uv_buffer.push_back(uv);
        } else if (strcmp(token, "vn") == 0) {
            float3 normal = {float(atof(gstrtok(&line_copy, WHITESPACE))),
                             float(atof(gstrtok(&line_copy, WHITESPACE))),
                             float(atof(gstrtok(&line_copy, WHITESPACE)))};

            normals_buffer.push_back(normal);
        } else if (strcmp(token, "f") == 0) {
            idxs = GetIndices(line_copy, vertex_buffer.size(), uv_buffer.size(),
                              normals_buffer.size());

            for (unsigned i = 2; i < idxs.size(); i++) {
                index_buffer.push_back(idxs[0]);
                index_buffer.push_back(idxs[i - 1]);
                index_buffer.push_back(idxs[i]);

                scene.triangles.push_back(Triangle(vertex_buffer[idxs[0].v],
                                                   vertex_buffer[idxs[i - 1].v],
                                                   vertex_buffer[idxs[i].v]));

                Attributes a;
                a.material_id = current_material_id;
                a.uv[0] = idxs[0 + 0].t >= 0 ? uv_buffer[idxs[0 + 0].t]
                                             : make_float2(0.0f);
                a.uv[1] = idxs[i - 1].t >= 0 ? uv_buffer[idxs[i - 1].t]
                                             : make_float2(0.0f);
                a.uv[2] = idxs[i + 0].t >= 0 ? uv_buffer[idxs[i + 0].t]
                                             : make_float2(0.0f);
                a.normal[0] = idxs[0 + 0].n >= 0
                                  ? normals_buffer[idxs[0 + 0].n]
                                  : GenerateNormal(scene.triangles.back());
                a.normal[1] = idxs[i - 1].n >= 0
                                  ? normals_buffer[idxs[i - 1].n]
                                  : GenerateNormal(scene.triangles.back());
                a.normal[2] = idxs[i + 0].n >= 0
                                  ? normals_buffer[idxs[i + 0].n]
                                  : GenerateNormal(scene.triangles.back());
                scene.attributes.push_back(a);
            }
        }
    }
    printf("Geometry\n");
    printf("  faces:        %d\n", (unsigned)scene.triangles.size());
    printf("  verts:        %d\n", (unsigned)vertex_buffer.size());
    fclose(fp);

    // Compute the AABB of the scene
    scene.aabb = {make_float3(FLT_MAX, FLT_MAX, FLT_MAX),
                  make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX)};
    for (auto t : scene.triangles) {
        scene.aabb = Combine(scene.aabb, t.v0);
        scene.aabb = Combine(scene.aabb, t.v1);
        scene.aabb = Combine(scene.aabb, t.v2);
    }

    printf("  aabb: (%f %f %f %f %f %f)\n", scene.aabb.min.x, scene.aabb.min.y,
           scene.aabb.min.z, scene.aabb.max.x, scene.aabb.max.y,
           scene.aabb.max.z);

    scene.light = SetupLight(filename, scene.aabb);

    printf("  light: %f %f %f\n", scene.light.x, scene.light.y, scene.light.z);

    return scene;
}