
#include "FileIO.h"

#include <vector>

#include "Common.cuh"

#define MAX_LENGTH 256
#define WHITESPACE " \t\n"

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

std::vector<Triangle> LoadOBJFromFile(std::string filename)
{
    FILE* fp;
    char* line = (char*)malloc(MAX_LENGTH);
    char* token = NULL;
    char* a[4];
    unsigned idx[4];
    uint32_t vertex_count = 0;
    uint32_t normal_count = 0;
    uint32_t current_cmp_count = 0;

    std::vector<float> vertex_buffer;
    std::vector<unsigned> index_buffer;

    index_buffer.reserve(200000);
    vertex_buffer.reserve(200000);

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

        else if (strcmp(token, "v") == 0) {
            token = gstrtok(&line_copy, WHITESPACE);
            if (token[0] == '\0') {
                token = gstrtok(&line_copy, WHITESPACE);
            }

            vertex_buffer.push_back((float)atof(token));
            vertex_buffer.push_back(
                (float)atof(gstrtok(&line_copy, WHITESPACE)));
            vertex_buffer.push_back(
                (float)atof(gstrtok(&line_copy, WHITESPACE)));
        } else if (strcmp(token, "f") == 0) {
            for (int i = 0; i < 3; i++) {
                a[i] = gstrtok(&line_copy, WHITESPACE);
                int f = atoi(a[i]);
                idx[i] = f < 0 ? f + (unsigned)vertex_buffer.size() / 3 : f - 1;
            }
            index_buffer.push_back(idx[0]);
            index_buffer.push_back(idx[1]);
            index_buffer.push_back(idx[2]);

            if ((a[3] = gstrtok(&line_copy, WHITESPACE)) != NULL &&
                a[3][0] != '\0') {
                idx[1] = idx[2];
                idx[2] = atoi(a[3]);
                idx[2] += idx[2] < 0 ? (unsigned)vertex_buffer.size() / 3 : -1;

                index_buffer.push_back(idx[0]);
                index_buffer.push_back(idx[1]);
                index_buffer.push_back(idx[2]);
            }
        }
    }
    printf("Geometry\n");
    printf("  faces:        %d\n", (unsigned)index_buffer.size() / 3);
    printf("  verts:        %d\n", (unsigned)vertex_buffer.size() / 3);
    fclose(fp);

    std::vector<Triangle> triangles;
    triangles.reserve(index_buffer.size() / 3);
    for (unsigned i = 0; i < index_buffer.size(); i += 3) {
        triangles.push_back(
            Triangle(vertex_buffer[index_buffer[i + 0] * 3 + 0],
                     vertex_buffer[index_buffer[i + 0] * 3 + 1],
                     vertex_buffer[index_buffer[i + 0] * 3 + 2],
                     vertex_buffer[index_buffer[i + 1] * 3 + 0],
                     vertex_buffer[index_buffer[i + 1] * 3 + 1],
                     vertex_buffer[index_buffer[i + 1] * 3 + 2],
                     vertex_buffer[index_buffer[i + 2] * 3 + 0],
                     vertex_buffer[index_buffer[i + 2] * 3 + 1],
                     vertex_buffer[index_buffer[i + 2] * 3 + 2]));
    }
    return triangles;
}