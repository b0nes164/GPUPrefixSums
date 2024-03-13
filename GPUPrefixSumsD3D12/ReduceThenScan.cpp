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
	DeviceInfo _deviceInfo) :
	GPUPrefixSummer("ReduceThenScan ", 3072, 1 << 13)
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
	m_rtsReduce = new RTSKernels::Reduce(m_device, m_devInfo, m_compileArguments);
	m_rtsScan = new RTSKernels::Scan(m_device, m_devInfo, m_compileArguments);
	m_rtsDownSweepInclusive = new RTSKernels::DownSweepInclusive(m_device, m_devInfo, m_compileArguments);
	m_rtsDownSweepExclusive = new RTSKernels::DownSweepExclusive(m_device, m_devInfo, m_compileArguments);

	m_initOne = new UtilityKernels::InitOne(m_device, m_devInfo, m_compileArguments);
	m_initRandom = new UtilityKernels::InitRandom(m_device, m_devInfo, m_compileArguments);
	m_clearErrorCount = new UtilityKernels::ClearErrorCount(m_device, m_devInfo, m_compileArguments);
	m_validateOneExclusive = new UtilityKernels::ValidateOneExclusive(m_device, m_devInfo, m_compileArguments);
	m_validateOneInclusive = new UtilityKernels::ValidateOneInclusive(m_device, m_devInfo, m_compileArguments);
	m_validateRandomExclusive =
		new UtilityKernels::ValidateRandomExclusive(m_device, m_devInfo, m_compileArguments);
	m_validateRandomInclusive =
		new UtilityKernels::ValidateRandomInclusive(m_device, m_devInfo, m_compileArguments);
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