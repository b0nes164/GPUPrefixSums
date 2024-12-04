/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#include "pch.h"
#include "ChainedScanDecoupledLookbackDecoupledFallback.h"

ChainedScanDecoupledLookbackDecoupledFallback::ChainedScanDecoupledLookbackDecoupledFallback(
    winrt::com_ptr<ID3D12Device> _device, GPUPrefixSums::DeviceInfo _deviceInfo)
    : GPUPrefixSumBase("ChainedScanDecoupledLookbackDecoupledFallback ", 3072, 1 << 13) {
    m_device.copy_from(_device.get());
    m_devInfo = _deviceInfo;

    Initialize();
}

ChainedScanDecoupledLookbackDecoupledFallback::~ChainedScanDecoupledLookbackDecoupledFallback() {}

void ChainedScanDecoupledLookbackDecoupledFallback::InitComputeShaders() {
    const std::filesystem::path path = "Shaders/ChainedScanDecoupledLookbackDecoupledFallback.hlsl";
    m_initCSDLDF = new CSDLDFKernels::InitCSDLDF(m_device, m_devInfo, m_compileArguments, path);
    m_csdldfInclusive =
        new CSDLDFKernels::CSDLDFInclusive(m_device, m_devInfo, m_compileArguments, path);
    m_csdldfExclusive =
        new CSDLDFKernels::CSDLDFExclusive(m_device, m_devInfo, m_compileArguments, path);
}

void ChainedScanDecoupledLookbackDecoupledFallback::DisposeBuffers() {
    m_scanInBuffer = nullptr;
    m_scanOutBuffer = nullptr;
    m_threadBlockReductionBuffer = nullptr;
    m_scanValidationBuffer = nullptr;
}

void ChainedScanDecoupledLookbackDecoupledFallback::InitStaticBuffers() {
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

void ChainedScanDecoupledLookbackDecoupledFallback::PrepareScanCmdListInclusive() {
    m_initCSDLDF->Dispatch(m_cmdList, m_scanBumpBuffer->GetGPUVirtualAddress(),
                           m_threadBlockReductionBuffer->GetGPUVirtualAddress(), m_partitions);
    UAVBarrierSingle(m_cmdList, m_scanBumpBuffer);
    UAVBarrierSingle(m_cmdList, m_threadBlockReductionBuffer);

    m_csdldfInclusive->Dispatch(m_cmdList, m_scanInBuffer->GetGPUVirtualAddress(),
                                m_scanOutBuffer->GetGPUVirtualAddress(),
                                m_scanBumpBuffer->GetGPUVirtualAddress(),
                                m_threadBlockReductionBuffer, m_vectorizedSize, m_partitions);
}

void ChainedScanDecoupledLookbackDecoupledFallback::PrepareScanCmdListExclusive() {
    m_initCSDLDF->Dispatch(m_cmdList, m_scanBumpBuffer->GetGPUVirtualAddress(),
                           m_threadBlockReductionBuffer->GetGPUVirtualAddress(), m_partitions);
    UAVBarrierSingle(m_cmdList, m_scanBumpBuffer);
    UAVBarrierSingle(m_cmdList, m_threadBlockReductionBuffer);

    m_csdldfExclusive->Dispatch(m_cmdList, m_scanInBuffer->GetGPUVirtualAddress(),
                                m_scanOutBuffer->GetGPUVirtualAddress(),
                                m_scanBumpBuffer->GetGPUVirtualAddress(),
                                m_threadBlockReductionBuffer, m_vectorizedSize, m_partitions);
}