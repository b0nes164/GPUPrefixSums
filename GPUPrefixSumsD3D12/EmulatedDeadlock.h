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
#include "EmulatedDeadlockKernels.h"

class EmulatedDeadlock : public GPUPrefixSumBase {
    winrt::com_ptr<ID3D12Resource> m_scanBumpBuffer;

    EmulatedDeadlockKernels::InitEmulatedDeadlock* m_initEmulatedDeadlock;
    EmulatedDeadlockKernels::ClearBump* m_clearBump;
    EmulatedDeadlockKernels::EmulatedDeadlockFirstPass* m_emulateDeadlockFirstPass;
    EmulatedDeadlockKernels::EmulatedDeadlockSecondPass* m_emulateDeadlockSecPass;
    EmulatedDeadlockKernels::Thrasher* m_thrasher;

   public:
    EmulatedDeadlock(winrt::com_ptr<ID3D12Device> _device, GPUPrefixSums::DeviceInfo _deviceInfo);

    ~EmulatedDeadlock();

    void TestExclusiveScanInitOne(uint32_t testSize, bool shouldReadBack,
                                  bool shouldValidate) override;

    void TestExclusiveScanInitRandom(uint32_t testSize, bool shouldReadBack,
                                     bool shouldValidate) override;

    void TestAll() override;

   protected:
    void InitComputeShaders() override;

    void DisposeBuffers() override;

    void InitStaticBuffers() override;

    void PrepareScanCmdListInclusive() override;

    void PrepareScanCmdListExclusive() override;
};