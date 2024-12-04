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
#include "RTSKernels.h"

class ReduceThenScan : public GPUPrefixSumBase {
    RTSKernels::Reduce* m_rtsReduce;
    RTSKernels::Scan* m_rtsScan;
    RTSKernels::PropagateInclusive* m_rtsPropagateInclusive;
    RTSKernels::PropagateExclusive* m_rtsPropagateExclusive;

   public:
    ReduceThenScan(winrt::com_ptr<ID3D12Device> _device, GPUPrefixSums::DeviceInfo _deviceInfo);

    ~ReduceThenScan();

   protected:
    void InitComputeShaders() override;

    void DisposeBuffers() override;

    void InitStaticBuffers() override;

    void PrepareScanCmdListInclusive() override;

    void PrepareScanCmdListExclusive() override;
};