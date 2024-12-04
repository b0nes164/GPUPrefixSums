/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "pch.h"
#include "GPUPrefixSums.h"
#include "GPUPrefixSumBase.h"
#include "CSDLDFKernels.h"

class ChainedScanDecoupledLookbackDecoupledFallback : public GPUPrefixSumBase {
    winrt::com_ptr<ID3D12Resource> m_scanBumpBuffer;

    CSDLDFKernels::InitCSDLDF* m_initCSDLDF;
    CSDLDFKernels::CSDLDFInclusive* m_csdldfInclusive;
    CSDLDFKernels::CSDLDFExclusive* m_csdldfExclusive;

   public:
    ChainedScanDecoupledLookbackDecoupledFallback(winrt::com_ptr<ID3D12Device> _device,
                                                  GPUPrefixSums::DeviceInfo _deviceInfo);

    ~ChainedScanDecoupledLookbackDecoupledFallback();

   protected:
    void InitComputeShaders() override;

    void DisposeBuffers() override;

    void InitStaticBuffers() override;

    void PrepareScanCmdListInclusive() override;

    void PrepareScanCmdListExclusive() override;
};