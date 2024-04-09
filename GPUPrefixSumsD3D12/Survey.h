/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/4/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "pch.h"
#include "Utils.h"
#include "UtilityKernels.h"
#include "GPUPrefixSums.h"
#include "SurveyKernels.h"

//Enough differences that instead of inheriting from base class,
//a seperate class is created. There is probably a more elegant way to do this
class Survey
{
    const uint32_t k_maxReadBack = 1 << 16;
    const uint32_t k_maxWaveSize = 128;
    const uint32_t k_groupSize = 256;

    winrt::com_ptr<ID3D12Device> m_device;
    GPUPrefixSums::DeviceInfo m_devInfo{};
    std::vector<std::wstring> m_compileArguments;
    uint32_t m_size;

    winrt::com_ptr<ID3D12GraphicsCommandList> m_cmdList;
    winrt::com_ptr<ID3D12CommandQueue> m_cmdQueue;
    winrt::com_ptr<ID3D12CommandAllocator> m_cmdAllocator;

    winrt::com_ptr<ID3D12QueryHeap> m_queryHeap;
    winrt::com_ptr<ID3D12Fence> m_fence;
    wil::unique_event_nothrow m_fenceEvent;
    uint64_t m_nextFenceValue;

    winrt::com_ptr<ID3D12Resource> m_scanBuffer;
    winrt::com_ptr<ID3D12Resource> m_validationInfoBuffer;
    winrt::com_ptr<ID3D12Resource> m_errorCountBuffer;
    winrt::com_ptr<ID3D12Resource> m_readBackBuffer;

    //Utility
    UtilityKernels::InitOne* m_initOne;
    UtilityKernels::ClearErrorCount* m_clearErrorCount;
    UtilityKernels::ValidateOneInclusive* m_validateInclusive;
    UtilityKernels::ValidateOneExclusive* m_validateExclusive;

    //Serial
    SurveyKernels::SerialInclusive* m_serialInclusive;
    SurveyKernels::SerialExclusive* m_serialExclusive;

    //Wave Level
    SurveyKernels::WaveKoggeStoneInclusive* m_waveKoggeStoneInclusive;
    SurveyKernels::WaveKoggeStoneExclusive* m_waveKoggeStoneExclusive;
    SurveyKernels::WaveKoggeStoneShuffleInclusive* m_waveKoggeStoneShuffleInclusive;
    SurveyKernels::WaveKoggeStoneShuffleExclusive* m_waveKoggeStoneShuffleExclusive;
    SurveyKernels::WaveKoggeStoneIntrinsicInclusive* m_waveKoggeStoneIntrinsicInclusive;
    SurveyKernels::WaveKoggeStoneIntrinsicExclusive* m_waveKoggeStoneIntrinsicExclusive;
    SurveyKernels::WaveRakingReduceInclusive* m_waveRakingReduceInclusive;
    SurveyKernels::WaveRakingReduceExclusive* m_waveRakingReduceExclusive;

    //Block Level Without Wave Intrinsics
    SurveyKernels::BlockKoggeStoneInclusive* m_blockKoggeStoneInclusive;
    SurveyKernels::BlockKoggeStoneExclusive* m_blockKoggeStoneExclusive;

    SurveyKernels::BlockSklanskyInclusive* m_blockSklanskyInclusive;
    SurveyKernels::BlockSklanskyExclusive* m_blockSklanskyExclusive;

    SurveyKernels::BlockBrentKungBlellochInclusive* m_blockBrentKungBlellochInclusive;
    SurveyKernels::BlockBrentKungBlellochExclusive* m_blockBrentKungBlellochExclusive;

    SurveyKernels::BlockReduceScanInclusive* m_blockReduceScanInclusive;
    SurveyKernels::BlockReduceScanExclusive* m_blockReduceScanExclusive;

    //Block Level With Wave Intrinsics
    SurveyKernels::BlockBrentKungIntrinsicInclusive* m_blockBrentKungIntrinsicInclusive;
    SurveyKernels::BlockBrentKungIntrinsicExclusive* m_blockBrentKungIntrinsicExclusive;

    SurveyKernels::BlockBrentKungFusedIntrinsicInclusive* m_blockBrentKungFusedIntrinsicInclusive;

    SurveyKernels::BlockSklanskyIntrinsicInclusive* m_blockSklanskyIntrinsicInclusive;
    SurveyKernels::BlockSklanskyIntrinsicInclusiveAlt* m_blockSklanskyIntrinsicInclusiveAlt;
    SurveyKernels::BlockSklanskyIntrinsicExclusive* m_blockSklanskyIntrinsicExclusive;

public:
    Survey(
        winrt::com_ptr<ID3D12Device> _device,
        const GPUPrefixSums::DeviceInfo _deviceInfo);

    ~Survey();

    //Serial
    bool TestSerialInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestSerialExclusive(bool shouldPrint, bool shouldPrintValidation);

    //Wave
    bool TestWaveKoggeStoneInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestWaveKoggeStoneExclusive(bool shouldPrint, bool shouldPrintValidation);

    bool TestWaveKoggeStoneShuffleInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestWaveKoggeStoneShuffleExclusive(bool shouldPrint, bool shouldPrintValidation);

    bool TestWaveKoggeStoneIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestWaveKoggeStoneIntrinsicExclusive(bool shouldPrint, bool shouldPrintValidation);

    bool TestWaveRakingReduceInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestWaveRakingReduceExclusive(bool shouldPrint, bool shouldPrintValidation);

    //Block
    bool TestBlockKoggeStoneInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestBlockKoggeStoneExclusive(bool shouldPrint, bool shouldPrintValidation);

    bool TestBlockSklanskyInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestBlockSklanskyExclusive(bool shouldPrint, bool shouldPrintValidation);

    bool TestBlockBrentKungBlellochInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestBlockBrentKungBlellochExclusive(bool shouldPrint, bool shouldPrintValidation);

    bool TestBlockReduceScanInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestBlockReduceScanExclusive(bool shouldPrint, bool shouldPrintValidation);

    bool TestBlockBrentKungIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestBlockBrentKungIntrinsicExclusive(bool shouldPrint, bool shouldPrintValidation);

    bool TestBlockBrentKungFusedIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation);

    bool TestBlockSklanskyIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation);
    bool TestBlockSklanskyIntrinsicInclusiveAlt(bool shouldPrint, bool shouldPrintValidation);
    bool TestBlockSklanskyIntrinsicExclusive(bool shouldPrint, bool shouldPrintValidation);

    void TestAll();

private:
    void InitUtilityShaders();

    void InitComputeShaders();

    void UpdateSize(uint32_t size);

    void DisposeBuffers();

    void InitStaticBuffers();

    void InitBuffers(uint32_t size);

    void CreateTestInput();

    void ExecuteCommandList();

    bool ValidateInclusive(
        const char* scanName,
        uint32_t testSize,
        bool shouldPrint);

    bool ValidateExclusive(
        const char* scanName,
        uint32_t testSize,
        bool shouldPrint);

    uint32_t ReadBackValidationInfo();

    void ReadBackOutput();

    bool TestScanInclusive(
        const char* scanName,
        SurveyKernelBase* scan,
        const bool& shouldPrint,
        const bool& shouldPrintValidation);

    bool TestScanInclusive(
        const char* scanName,
        SurveyKernelBase* scan,
        const uint32_t& size,
        const bool& shouldPrint,
        const bool& shouldPrintValidation);

    bool TestScanExclusive(
        const char* scanName,
        SurveyKernelBase* scan,
        const bool& shouldPrint,
        const bool& shouldPrintValidation);

    bool TestScanExclusive(
        const char* scanName,
        SurveyKernelBase* scan,
        const uint32_t& size,
        const bool& shouldPrint,
        const bool& shouldPrintValidation);
};
