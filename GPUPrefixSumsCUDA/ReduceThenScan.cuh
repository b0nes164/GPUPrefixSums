/******************************************************************************
 * GPUPrefixSums
 * Reduce then Scan 
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "Utils.cuh"
#include "ScanCommon.cuh"

namespace ReduceThenScan
{
	__global__ void Reduce(
		uint32_t* scan,
		uint32_t* threadBlockReductions,
		uint32_t vectorizedSize);

	__global__ void Scan(
		uint32_t* threadBlockReductions,
		uint32_t threadBlocks);

	__global__ void DownSweepExclusive(
		uint32_t* scan,
		uint32_t* threadBlockReductions,
		uint32_t vectorizedSize);

	__global__ void DownSweepInclusive(
		uint32_t* scan,
		uint32_t* threadBlockReductions,
		uint32_t vectorizedSize);
}