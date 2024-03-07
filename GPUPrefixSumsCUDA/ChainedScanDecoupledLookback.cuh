/******************************************************************************
 * GPUPrefixSums
 * Chained Scan with Decoupled Lookback Implementation
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 * Based off of Research by:
 *          Duane Merrill, Nvidia Corporation
 *          Michael Garland, Nvidia Corporation
 *          https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
 *
 ******************************************************************************/
#pragma once
#include "Utils.cuh"
#include "LocalScan.cuh"

namespace ChainedScanDecoupledLookback
{
	__global__ void CSDLExclusive(
		uint32_t* scan,
		volatile uint32_t* threadBlockReduction,
		volatile uint32_t* index,
		uint32_t alignedSize);

	__global__ void CSDLInclusive(
		uint32_t* scan,
		volatile uint32_t* threadBlockReduction,
		volatile uint32_t* index,
		uint32_t alignedSize);
}