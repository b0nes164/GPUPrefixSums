/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "Utils.cuh"

__device__ __forceinline__ void ScanExclusiveFull(
    uint32_t* _scan,
    uint32_t* _reduction,
    uint4* _shared,
    const uint32_t& _partStart,
    const uint32_t& _warpParts,
    const uint32_t& _warpPartStart)
{
    uint32_t warpReduction = 0;

    #pragma unroll
    for (uint32_t i = getLaneId() + _warpPartStart, k = 0;
        k < _warpParts;
        i += LANE_COUNT, ++k)
    {
        uint4 t = reinterpret_cast<uint4*>(_scan)[i + _partStart];

        uint32_t t2 = t.x;
        t.x += t.y;
        t.y = t2;

        t2 = t.x;
        t.x += t.z;
        t.z = t2;

        t2 = t.x;
        t.x += t.w;
        t.w = t2;

        t2 = InclusiveWarpScanCircularShift(t.x);
        _shared[i] = SetXAddYZW((getLaneId() ? t2 : 0) + (k ? warpReduction : 0), t);
        warpReduction += __shfl_sync(0xffffffff, t2, 0);
    }

    if (getLaneId() == 0)
        _reduction[WARP_INDEX] = warpReduction;
}

__device__ __forceinline__ void ScanExclusivePartial(
    uint32_t* _scan,
    uint32_t* _reduction,
    uint4* _shared,
    const uint32_t& _partStart,
    const uint32_t& _warpParts,
    const uint32_t& _warpPartStart,
    const uint32_t& _vectorizedSize)
{
    uint32_t warpReduction = 0;
    const uint32_t finalPartSize = _vectorizedSize - _partStart;
    #pragma unroll
    for (uint32_t i = getLaneId() + _warpPartStart, k = 0;
        k < _warpParts;
        i += LANE_COUNT, ++k)
    {
        uint4 t = i < finalPartSize ? reinterpret_cast<uint4*>(_scan)[i + _partStart] :
            make_uint4(0, 0, 0, 0);

        uint32_t t2 = t.x;
        t.x += t.y;
        t.y = t2;

        t2 = t.x;
        t.x += t.z;
        t.z = t2;

        t2 = t.x;
        t.x += t.w;
        t.w = t2;

        t2 = InclusiveWarpScanCircularShift(t.x);
        _shared[i] = SetXAddYZW((getLaneId() ? t2 : 0) + (k ? warpReduction : 0), t);
        warpReduction += __shfl_sync(0xffffffff, t2, 0);
    }

    if (getLaneId() == 0)
        _reduction[WARP_INDEX] = warpReduction;
}

__device__ __forceinline__ void ScanInclusiveFull(
    uint32_t* _scan,
    uint32_t* _reduction,
    uint4* _shared,
    const uint32_t& _partStart,
    const uint32_t& _warpParts,
    const uint32_t& _warpPartStart)
{
    uint32_t warpReduction = 0;

    #pragma unroll
    for (uint32_t i = getLaneId() + _warpPartStart, k = 0;
        k < _warpParts;
        i += LANE_COUNT, ++k)
    {
        uint4 t = reinterpret_cast<uint4*>(_scan)[i + _partStart];
        t.y += t.x;
        t.z += t.y;
        t.w += t.z;

        const uint32_t t2 = InclusiveWarpScanCircularShift(t.w);
        _shared[i] = AddUintToUint4((getLaneId() ? t2 : 0) + (k ? warpReduction : 0), t);
        warpReduction += __shfl_sync(0xffffffff, t2, 0);
    }

    if (getLaneId() == 0)
        _reduction[WARP_INDEX] = warpReduction;
}

__device__ __forceinline__ void ScanInclusivePartial(
    uint32_t* _scan,
    uint32_t* _reduction,
    uint4* _shared,
    const uint32_t& _partStart,
    const uint32_t& _warpParts,
    const uint32_t& _warpPartStart,
    const uint32_t& _vectorizedSize)
{
    uint32_t warpReduction = 0;
    const uint32_t finalPartSize = _vectorizedSize - _partStart;
    #pragma unroll
    for (uint32_t i = getLaneId() + _warpPartStart, k = 0;
        k < _warpParts;
        i += LANE_COUNT, ++k)
    {
        uint4 t = i < finalPartSize ? reinterpret_cast<uint4*>(_scan)[i + _partStart] :
            make_uint4(0, 0, 0, 0);
        t.y += t.x;
        t.z += t.y;
        t.w += t.z;

        const uint32_t t2 = InclusiveWarpScanCircularShift(t.w);
        _shared[i] = AddUintToUint4((getLaneId() ? t2 : 0) + (k ? warpReduction : 0), t);
        warpReduction += __shfl_sync(0xffffffff, t2, 0);
    }

    if (getLaneId() == 0)
        _reduction[WARP_INDEX] = warpReduction;
}

__device__ __forceinline__ void DownSweepFull(
    uint32_t* _scan,
    uint4* _csdl,
    const uint32_t& _prevReduction,
    const uint32_t& _partStart,
    const uint32_t& _warpParts,
    const uint32_t& _warpPartStart)
{
    #pragma unroll
    for (uint32_t i = getLaneId() + _warpPartStart, k = 0;
        k < _warpParts;
        i += LANE_COUNT, ++k)
    {
        reinterpret_cast<uint4*>(_scan)[i + _partStart] =
            AddUintToUint4(_prevReduction, _csdl[i]);
    }
}

__device__ __forceinline__ void DownSweepPartial(
    uint32_t* _scan,
    uint4* _csdl,
    const uint32_t& _prevReduction,
    const uint32_t& _partStart,
    const uint32_t& _warpParts,
    const uint32_t& _warpPartStart,
    const uint32_t& _vectorizedSize)
{
    const uint32_t finalPartSize = _vectorizedSize - _partStart;
    for (uint32_t i = getLaneId() + _warpPartStart, k = 0;
        k < _warpParts && i < finalPartSize;
        i += LANE_COUNT, ++k)
    {
        reinterpret_cast<uint4*>(_scan)[i + _partStart] =
            AddUintToUint4(_prevReduction, _csdl[i]);
    }
}