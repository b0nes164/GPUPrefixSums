/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#include "ChainedScanDecoupledLookbackDispatcher.cuh"
#include "CubDispatcher.cuh"

int main()
{
	ChainedScanDecoupledLookbackDispatcher* csdl =
		new ChainedScanDecoupledLookbackDispatcher(1 << 28);
	csdl->TestAllExclusive();
	csdl->TestAllInclusive();
	csdl->BatchTimingInclusive(1 << 28, 100);
	csdl->~ChainedScanDecoupledLookbackDispatcher();

	CubDispatcher* cub = new CubDispatcher(1 << 28);
	cub->BatchTimingCubChainedScan(1 << 28, 100);
	cub->~CubDispatcher();

	return 0;
}