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
#include "GPUPrefixSummer.h"
#include "RTSKernels.h"

class ReduceThenScan : public GPUPrefixSummer
{
	RTSKernels::Reduce* m_rtsReduce;
	RTSKernels::Scan* m_rtsScan;
	RTSKernels::DownSweepInclusive* m_rtsDownSweepInclusive;
	RTSKernels::DownSweepExclusive* m_rtsDownSweepExclusive;

public:
	ReduceThenScan(
		winrt::com_ptr<ID3D12Device> _device,
		DeviceInfo _deviceInfo);

	~ReduceThenScan();

protected:
	void InitComputeShaders() override;

	void DisposeBuffers() override;

	void InitStaticBuffers() override;

	void PrepareScanCmdListInclusive() override;

	void PrepareScanCmdListExclusive() override;
};