/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#include "pch.h"
#include "ReduceThenScan.h"

ReduceThenScan::ReduceThenScan(
	winrt::com_ptr<ID3D12Device> _device,
	GPUPrefixSums::DeviceInfo _deviceInfo) :
	GPUPrefixSumBase("ReduceThenScan ", 3072, 1 << 13)
{
	m_device.copy_from(_device.get());
	m_devInfo = _deviceInfo;

	Initialize();
}

ReduceThenScan::~ReduceThenScan()
{
}

void ReduceThenScan::InitComputeShaders()
{
	const std::filesystem::path path = "Shaders/ReduceThenScan.hlsl";
	m_rtsReduce = new RTSKernels::Reduce(m_device, m_devInfo, m_compileArguments, path);
	m_rtsScan = new RTSKernels::Scan(m_device, m_devInfo, m_compileArguments, path);
	m_rtsDownSweepInclusive = new RTSKernels::DownSweepInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_rtsDownSweepExclusive = new RTSKernels::DownSweepExclusive(m_device, m_devInfo, m_compileArguments, path);
}

void ReduceThenScan::DisposeBuffers()
{
	m_scanBuffer = nullptr;
	m_threadBlockReductionBuffer = nullptr;
	m_scanValidationBuffer = nullptr;
}

void ReduceThenScan::InitStaticBuffers()
{
	m_errorCountBuffer = CreateBuffer(
		m_device,
		sizeof(uint32_t),
		D3D12_HEAP_TYPE_DEFAULT,
		D3D12_RESOURCE_STATE_COMMON,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	m_readBackBuffer = CreateBuffer(
		m_device,
		k_maxReadBack * sizeof(uint32_t),
		D3D12_HEAP_TYPE_READBACK,
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_RESOURCE_FLAG_NONE);
}

void ReduceThenScan::PrepareScanCmdListInclusive()
{
	m_rtsReduce->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_threadBlockReductionBuffer->GetGPUVirtualAddress(),
		m_vectorizedSize,
		m_partitions);
	UAVBarrierSingle(m_cmdList, m_threadBlockReductionBuffer);

	m_rtsScan->Dispatch(
		m_cmdList,
		m_threadBlockReductionBuffer->GetGPUVirtualAddress(),
		m_partitions);
	UAVBarrierSingle(m_cmdList, m_scanBuffer);
	UAVBarrierSingle(m_cmdList, m_threadBlockReductionBuffer);

	m_rtsDownSweepInclusive->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_threadBlockReductionBuffer->GetGPUVirtualAddress(),
		m_vectorizedSize,
		m_partitions);
}

void ReduceThenScan::PrepareScanCmdListExclusive()
{
	m_rtsReduce->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_threadBlockReductionBuffer->GetGPUVirtualAddress(),
		m_vectorizedSize,
		m_partitions);
	UAVBarrierSingle(m_cmdList, m_threadBlockReductionBuffer);

	m_rtsScan->Dispatch(
		m_cmdList,
		m_threadBlockReductionBuffer->GetGPUVirtualAddress(),
		m_partitions);
	UAVBarrierSingle(m_cmdList, m_scanBuffer);
	UAVBarrierSingle(m_cmdList, m_threadBlockReductionBuffer);

	m_rtsDownSweepExclusive->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_threadBlockReductionBuffer->GetGPUVirtualAddress(),
		m_vectorizedSize,
		m_partitions);
}