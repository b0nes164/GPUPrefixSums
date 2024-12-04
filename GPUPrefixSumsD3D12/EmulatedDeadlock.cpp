/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#include "pch.h"
#include "EmulatedDeadlock.h"

EmulatedDeadlock::EmulatedDeadlock(winrt::com_ptr<ID3D12Device> _device,
                                   GPUPrefixSums::DeviceInfo _deviceInfo)
    : GPUPrefixSumBase("EmulatedDeadlock ", 3072, 1 << 13) {
    m_device.copy_from(_device.get());
    m_devInfo = _deviceInfo;

    Initialize();
}

EmulatedDeadlock::~EmulatedDeadlock() {}

void EmulatedDeadlock::TestExclusiveScanInitOne(uint32_t testSize, bool shouldReadBack,
                                                bool shouldValidate) {
    printf("\nEmulated deadlock is inclusive only.\n");
}

void EmulatedDeadlock::TestExclusiveScanInitRandom(uint32_t testSize, bool shouldReadBack,
                                                   bool shouldValidate) {
    printf("\nEmulated deadlock is inclusive only.\n");
}

void EmulatedDeadlock::TestAll() {
    printf("\nBeginning ");
    printf(k_scanName);
    printf("test all.\n");

    uint32_t testsPassed = 0;
    for (uint32_t i = k_partitionSize; i < k_partitionSize * 2; ++i) {
        testsPassed += ValidateScan(i, i, ValidationType::RAND_INCLUSIVE);

        if ((i & 127) == 0)
            printf(". ");
    }
    printf("\n");

    for (uint32_t i = 21; i <= 28; ++i) {
        if (ValidateScan(1 << i, i, ValidationType::ONE_INCLUSIVE))
            testsPassed++;
        else
            printf("err at size %u \n", 1 << i);
    }

    const uint32_t testsExpected = k_partitionSize + 8;
    printf(k_scanName);
    if (testsPassed == testsExpected)
        printf(" %u/%u ALL TESTS PASSED\n", testsPassed, testsExpected);
    else
        printf(" %u/%u TEST FAILED\n", testsPassed, testsExpected);
}

void EmulatedDeadlock::InitComputeShaders() {
    const std::filesystem::path path = "Shaders/EmulatedDeadlock.hlsl";

    m_initEmulatedDeadlock = new EmulatedDeadlockKernels::InitEmulatedDeadlock(
        m_device, m_devInfo, m_compileArguments, path);
    m_clearBump =
        new EmulatedDeadlockKernels::ClearBump(m_device, m_devInfo, m_compileArguments, path);
    m_emulateDeadlockFirstPass = new EmulatedDeadlockKernels::EmulatedDeadlockFirstPass(
        m_device, m_devInfo, m_compileArguments, path);
    m_emulateDeadlockSecPass = new EmulatedDeadlockKernels::EmulatedDeadlockSecondPass(
        m_device, m_devInfo, m_compileArguments, path);
    m_thrasher =
        new EmulatedDeadlockKernels::Thrasher(m_device, m_devInfo, m_compileArguments, path);
}

void EmulatedDeadlock::DisposeBuffers() {
    m_scanInBuffer = nullptr;
    m_threadBlockReductionBuffer = nullptr;
    m_scanValidationBuffer = nullptr;
}

void EmulatedDeadlock::InitStaticBuffers() {
    m_scanBumpBuffer =
        CreateBuffer(m_device, sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT,
                     D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_errorCountBuffer =
        CreateBuffer(m_device, sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT,
                     D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_readBackBuffer =
        CreateBuffer(m_device, k_maxReadBack * sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK,
                     D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
}

void EmulatedDeadlock::PrepareScanCmdListInclusive() {
    m_initEmulatedDeadlock->Dispatch(m_cmdList, m_scanBumpBuffer->GetGPUVirtualAddress(),
                                     m_threadBlockReductionBuffer->GetGPUVirtualAddress(),
                                     m_partitions);
    UAVBarrierSingle(m_cmdList, m_scanBumpBuffer);
    UAVBarrierSingle(m_cmdList, m_threadBlockReductionBuffer);

    m_emulateDeadlockFirstPass->Dispatch(
        m_cmdList, m_scanInBuffer->GetGPUVirtualAddress(), m_scanOutBuffer->GetGPUVirtualAddress(),
        m_scanBumpBuffer->GetGPUVirtualAddress(), m_threadBlockReductionBuffer, m_vectorizedSize,
        m_partitions);
    UAVBarrierSingle(m_cmdList, m_scanBumpBuffer);
    UAVBarrierSingle(m_cmdList, m_threadBlockReductionBuffer);

    m_clearBump->Dispatch(m_cmdList, m_scanBumpBuffer->GetGPUVirtualAddress());
    UAVBarrierSingle(m_cmdList, m_scanBumpBuffer);

    m_emulateDeadlockSecPass->Dispatch(
        m_cmdList, m_scanInBuffer->GetGPUVirtualAddress(), m_scanOutBuffer->GetGPUVirtualAddress(),
        m_scanBumpBuffer->GetGPUVirtualAddress(), m_threadBlockReductionBuffer, m_vectorizedSize,
        m_partitions);
}

void EmulatedDeadlock::PrepareScanCmdListExclusive() {}
