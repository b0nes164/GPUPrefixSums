/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "pch.h"
#include "GPUPrefixSums.h"
#include "GPUPrefixSumBase.h"
#include "CSDLKernels.h"

class ChainedScanDecoupledLookback : public GPUPrefixSumBase
{
    winrt::com_ptr<ID3D12Resource> m_indexBuffer;

    CSDLKernels::InitCSDL* m_initCSDL;
    CSDLKernels::CSDLInclusive* m_csdlInclusive;
    CSDLKernels::CSDLExclusive* m_csdlExclusive;

public:
    ChainedScanDecoupledLookback(
        winrt::com_ptr<ID3D12Device> _device,
        GPUPrefixSums::DeviceInfo _deviceInfo);

    ~ChainedScanDecoupledLookback();

protected:
    void InitComputeShaders() override;

    void DisposeBuffers() override;

    void InitStaticBuffers() override;

    void PrepareScanCmdListInclusive() override;

    void PrepareScanCmdListExclusive() override;
};