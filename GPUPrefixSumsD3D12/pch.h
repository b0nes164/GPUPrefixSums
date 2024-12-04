/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once

#define WIN32_LEAN_AND_MEAN
#define STRICT
#define NOMINMAX

#include <Windows.h>
#include <d3d12.h>
#include <d3dx12/d3dx12.h>
#include <dxcapi.h>
#include <dxgi1_6.h>
#include <wil/resource.h>
#include <winrt/base.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")