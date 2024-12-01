/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 11/30/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#include "ChainedScanDecoupledLookbackDispatcher.cuh"
#include "CubDispatcher.cuh"
#include "EmulatedDeadlockingDispatcher.cuh"
#include "ReduceThenScanDispatcher.cuh"

int main() {
    ChainedScanDecoupledLookbackDispatcher* csdl =
        new ChainedScanDecoupledLookbackDispatcher(1 << 28);
    csdl->TestAllExclusive();
    csdl->TestAllInclusive();
    csdl->BatchTimingInclusive(1 << 28, 100);
    csdl->~ChainedScanDecoupledLookbackDispatcher();

    ReduceThenScanDispatcher* rts = new ReduceThenScanDispatcher(1 << 28);
    rts->TestAllExclusive();
    rts->TestAllInclusive();
    rts->BatchTimingInclusive(1 << 28, 100);
    rts->~ReduceThenScanDispatcher();

    EmulatedDeadlockingDispatcher* ed = new EmulatedDeadlockingDispatcher(1 << 28);
    ed->TestAllInclusive();
    ed->BatchTimingInclusive(1 << 28, 100);
    ed->~EmulatedDeadlockingDispatcher();

    CubDispatcher* cub = new CubDispatcher(1 << 28);
    cub->BatchTimingCubChainedScan(1 << 28, 100);
    cub->~CubDispatcher();

    return 0;
}