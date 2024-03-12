/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/6/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once

struct DeviceInfo
{
    std::wstring Description;
    std::wstring SupportedShaderModel;
    uint32_t SIMDWidth;
    uint32_t SIMDLaneCount;
    uint32_t SIMDMaxWidth;
    bool SupportsWaveIntrinsics;
    bool SupportsReduceThenScan;
    bool SupportsChainedScanDecoupledLookback;
};