#pragma once

#include <string>
#include <vector>

struct Triangle;

std::vector<Triangle> LoadOBJFromFile(std::string filename);
