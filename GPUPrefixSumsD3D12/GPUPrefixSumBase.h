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
#include "Utils.h"
#include "UtilityKernels.h"

class GPUPrefixSumBase
{
protected:
    const char* k_scanName;
    const uint32_t k_partitionSize;
    const uint32_t k_maxReadBack;
    const uint32_t k_maxRandSize = 1 << 20;

    uint32_t m_alignedSize;
    uint32_t m_vectorizedSize;
    uint32_t m_partitions;

    winrt::com_ptr<ID3D12Device> m_device;
    GPUPrefixSums::DeviceInfo m_devInfo{};
    std::vector<std::wstring> m_compileArguments;

    winrt::com_ptr<ID3D12GraphicsCommandList> m_cmdList;
    winrt::com_ptr<ID3D12CommandQueue> m_cmdQueue;
    winrt::com_ptr<ID3D12CommandAllocator> m_cmdAllocator;

    winrt::com_ptr<ID3D12QueryHeap> m_queryHeap;
    winrt::com_ptr<ID3D12Fence> m_fence;
    wil::unique_event_nothrow m_fenceEvent;
    uint64_t m_nextFenceValue;
    uint64_t m_timestampFrequency;

    winrt::com_ptr<ID3D12Resource> m_scanBuffer;
    winrt::com_ptr<ID3D12Resource> m_threadBlockReductionBuffer;
    winrt::com_ptr<ID3D12Resource> m_scanValidationBuffer;
    winrt::com_ptr<ID3D12Resource> m_errorCountBuffer;
    winrt::com_ptr<ID3D12Resource> m_readBackBuffer;

    UtilityKernels::InitOne* m_initOne;
    UtilityKernels::InitRandom* m_initRandom;
    UtilityKernels::ClearErrorCount* m_clearErrorCount;
    UtilityKernels::ValidateOneExclusive* m_validateOneExclusive;
    UtilityKernels::ValidateOneInclusive* m_validateOneInclusive;
    UtilityKernels::ValidateRandomExclusive* m_validateRandomExclusive;
    UtilityKernels::ValidateRandomInclusive* m_validateRandomInclusive;

    enum class ValidationType
    {
        ONE_INCLUSIVE = 0,
        ONE_EXCLUSIVE = 1,
        RAND_INCLUSIVE = 2,
        RAND_EXCLUSIVE = 3
    };

    GPUPrefixSumBase(
        const char* scanName,
        uint32_t partitionSize,
        uint32_t maxReadBack) :
        k_scanName(scanName),
        k_partitionSize(partitionSize),
        k_maxReadBack(maxReadBack)
    {
    };

    ~GPUPrefixSumBase()
    {
    };

public:
    virtual void TestExclusiveScanInitOne(
        uint32_t testSize,
        bool shouldReadBack,
        bool shouldValidate)
    {
        UpdateSize(testSize, ValidationType::ONE_EXCLUSIVE);
        CreateTestInput(0, ValidationType::ONE_EXCLUSIVE);
        PrepareScanCmdListExclusive();
        ExecuteCommandList();

        if (shouldValidate)
            ValidateOutput(true, ValidationType::ONE_EXCLUSIVE);

        if (shouldReadBack)
            ReadBackOutput();
    }

    virtual void TestInclusiveScanInitOne(
        uint32_t testSize,
        bool shouldReadBack,
        bool shouldValidate)
    {
        UpdateSize(testSize, ValidationType::ONE_INCLUSIVE);
        CreateTestInput(0, ValidationType::ONE_INCLUSIVE);
        PrepareScanCmdListInclusive();
        ExecuteCommandList();

        if (shouldValidate)
            ValidateOutput(true, ValidationType::ONE_INCLUSIVE);

        if (shouldReadBack)
            ReadBackOutput();
    }

    virtual void TestExclusiveScanInitRandom(
        uint32_t testSize,
        bool shouldReadBack,
        bool shouldValidate)
    {
        if (testSize <= k_maxRandSize)
        {
            UpdateSize(testSize, ValidationType::RAND_EXCLUSIVE);
            CreateTestInput(0, ValidationType::RAND_EXCLUSIVE);
            PrepareScanCmdListExclusive();
            ExecuteCommandList();

            if (shouldValidate)
                ValidateOutput(true, ValidationType::RAND_EXCLUSIVE);

            if (shouldReadBack)
                ReadBackOutput();
        }
        else
        {
            printf("Error, test size exceeds maximum test size. \n");
            printf("Due to numeric limits, the maximum test size for");
            printf("test values initialized to random integers is %u.\n", k_maxRandSize);
        }
    }

    virtual void TestInclusiveScanInitRandom(
        uint32_t testSize,
        bool shouldReadBack,
        bool shouldValidate)
    {
        if (testSize <= k_maxRandSize)
        {
            UpdateSize(testSize, ValidationType::RAND_INCLUSIVE);
            CreateTestInput(0, ValidationType::RAND_INCLUSIVE);
            PrepareScanCmdListInclusive();
            ExecuteCommandList();

            if (shouldValidate)
                ValidateOutput(true, ValidationType::RAND_INCLUSIVE);

            if (shouldReadBack)
                ReadBackOutput();
        }
        else
        {
            printf("Error, test size exceeds maximum test size. \n");
            printf("Due to numeric limits, the maximum test size for");
            printf("test values initialized to random integers is %u.\n", k_maxRandSize);
        }
    }

    virtual void TestAll()
    {
        printf("\nBeginning ");
        printf(k_scanName);
        printf("test all.\n");

        uint32_t testsPassed = 0;
        for (uint32_t i = k_partitionSize; i < k_partitionSize * 2; ++i)
        {
            testsPassed += ValidateScan(i, i, ValidationType::RAND_INCLUSIVE);
            testsPassed += ValidateScan(i, i, ValidationType::RAND_EXCLUSIVE);

            if ((i & 127) == 0)
                printf(". ");
        }
        printf("\n");
        
        for (uint32_t i = 20; i <= 24; ++i)
        {
            testsPassed += ValidateScan(1 << i, i, ValidationType::ONE_INCLUSIVE);
            testsPassed += ValidateScan(1 << i, i, ValidationType::ONE_EXCLUSIVE);
        }

        const uint32_t testsExpected = (k_partitionSize + 5) * 2;
        printf(k_scanName);
        if (testsPassed == testsExpected)
            printf(" %u/%u ALL TESTS PASSED\n", testsPassed, testsExpected);
        else
            printf(" %u/%u TEST FAILED\n", testsPassed, testsExpected);
    }

    void BatchTimingInclusiveInitOne(uint32_t inputSize, uint32_t batchSize)
    {
        UpdateSize(inputSize, ValidationType::ONE_INCLUSIVE);
        printf("\nBeginning ");
        printf(k_scanName);
        PrintScanTestType(ValidationType::ONE_INCLUSIVE);
        printf("batch timing test at:\n");
        printf("Size: %u\n", inputSize);
        printf("Test size: %u\n", batchSize);

        double totalTime = 0.0;
        for (uint32_t i = 0; i <= batchSize; ++i)
        {
            double t = TimeScan(0, ValidationType::ONE_INCLUSIVE);
            if (i)
                totalTime += t;

            if ((i & 7) == 0)
                printf(".");
        }
        printf("\n");

        printf("Total time elapsed: %f\n", totalTime);
        printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", inputSize, inputSize / totalTime * batchSize);
    }

protected:
    void Initialize()
    {
        InitUtilityShaders();
        InitComputeShaders();

        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        winrt::check_hresult(m_device->CreateCommandQueue(&desc, IID_PPV_ARGS(m_cmdQueue.put())));
        winrt::check_hresult(m_device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(m_cmdAllocator.put())));
        winrt::check_hresult(m_device->CreateCommandList(0, desc.Type, m_cmdAllocator.get(), nullptr, IID_PPV_ARGS(m_cmdList.put())));
        winrt::check_hresult(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_fence.put())));
        m_fenceEvent.reset(CreateEvent(nullptr, FALSE, FALSE, nullptr));
        m_nextFenceValue = 1;

        D3D12_QUERY_HEAP_DESC queryHeapDesc = {};
        queryHeapDesc.Count = 2;
        queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        winrt::check_hresult(m_device->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(m_queryHeap.put())));
        winrt::check_hresult(m_cmdQueue->GetTimestampFrequency(&m_timestampFrequency));

        InitStaticBuffers();
    }
    
    virtual void InitUtilityShaders()
    {
        const std::filesystem::path path = "Shaders/Utility.hlsl";
        m_initOne = new UtilityKernels::InitOne(m_device, m_devInfo, m_compileArguments, path);
        m_initRandom = new UtilityKernels::InitRandom(m_device, m_devInfo, m_compileArguments, path);
        m_clearErrorCount = new UtilityKernels::ClearErrorCount(m_device, m_devInfo, m_compileArguments, path);
        m_validateOneExclusive = new UtilityKernels::ValidateOneExclusive(m_device, m_devInfo, m_compileArguments, path);
        m_validateOneInclusive = new UtilityKernels::ValidateOneInclusive(m_device, m_devInfo, m_compileArguments, path);
        m_validateRandomExclusive =
            new UtilityKernels::ValidateRandomExclusive(m_device, m_devInfo, m_compileArguments, path);
        m_validateRandomInclusive =
            new UtilityKernels::ValidateRandomInclusive(m_device, m_devInfo, m_compileArguments, path);
    }

    virtual void InitComputeShaders() = 0;

    void UpdateSize(uint32_t size, ValidationType valType)
    {
        //TODO : BAD 
        const uint32_t alignedSize = align16(size);  
        if (m_alignedSize != alignedSize)
        {
            m_alignedSize = alignedSize;
            m_vectorizedSize = vectorizeAlignedSize(m_alignedSize);
            m_partitions = divRoundUp(m_alignedSize, k_partitionSize);
            DisposeBuffers();
            InitBuffers(m_alignedSize, m_partitions, valType);
        }
    }

    virtual void DisposeBuffers() = 0;

    virtual void InitStaticBuffers() = 0;

    void InitBuffers(
        const uint32_t alignedSize,
        const uint32_t partitions,
        const ValidationType valType)
    {
        m_scanBuffer = CreateBuffer(
            m_device,
            alignedSize * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        m_threadBlockReductionBuffer = CreateBuffer(
            m_device,
            partitions * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        if ((uint32_t)valType > (uint32_t)ValidationType::ONE_EXCLUSIVE)
        {
            m_scanValidationBuffer = CreateBuffer(
                m_device,
                alignedSize * sizeof(uint32_t),
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        else
        {
            m_scanValidationBuffer = nullptr;
        }
    }

    void CreateTestInput(uint32_t seed, ValidationType valType)
    {
        if ((uint32_t)valType > (uint32_t)ValidationType::ONE_EXCLUSIVE)
        {
            m_initRandom->Dispatch(
                m_cmdList,
                m_scanBuffer->GetGPUVirtualAddress(),
                m_scanValidationBuffer->GetGPUVirtualAddress(),
                m_alignedSize,
                seed);
        }
        else
        {
            m_initOne->Dispatch(
                m_cmdList,
                m_scanBuffer->GetGPUVirtualAddress(),
                m_alignedSize);
        }
        UAVBarrierSingle(m_cmdList, m_scanBuffer);
        ExecuteCommandList();
    }

    virtual void PrepareScanCmdListExclusive() = 0;

    virtual void PrepareScanCmdListInclusive() = 0;

    void ExecuteCommandList()
    {
        winrt::check_hresult(m_cmdList->Close());
        ID3D12CommandList* commandLists[] = { m_cmdList.get() };
        m_cmdQueue->ExecuteCommandLists(1, commandLists);
        winrt::check_hresult(m_cmdQueue->Signal(m_fence.get(), m_nextFenceValue));
        winrt::check_hresult(m_fence->SetEventOnCompletion(m_nextFenceValue, m_fenceEvent.get()));
        ++m_nextFenceValue;
        winrt::check_hresult(m_fenceEvent.wait());
        winrt::check_hresult(m_cmdAllocator->Reset());
        winrt::check_hresult(m_cmdList->Reset(m_cmdAllocator.get(), nullptr));
    }

    bool ValidateOutput(bool shouldPrint, ValidationType valType)
    {
        m_clearErrorCount->Dispatch(
            m_cmdList,
            m_errorCountBuffer->GetGPUVirtualAddress());
        UAVBarrierSingle(m_cmdList, m_errorCountBuffer);

        switch (valType)
        {
        case ValidationType::ONE_INCLUSIVE:
            m_validateOneInclusive->Dispatch(
                m_cmdList,
                m_scanBuffer->GetGPUVirtualAddress(),
                m_errorCountBuffer->GetGPUVirtualAddress(),
                m_alignedSize);
            break;
        case ValidationType::ONE_EXCLUSIVE:
            m_validateOneExclusive->Dispatch(
                m_cmdList,
                m_scanBuffer->GetGPUVirtualAddress(),
                m_errorCountBuffer->GetGPUVirtualAddress(),
                m_alignedSize);
            break;
        case ValidationType::RAND_INCLUSIVE:
            m_validateRandomInclusive->Dispatch(
                m_cmdList,
                m_scanBuffer->GetGPUVirtualAddress(),
                m_scanValidationBuffer->GetGPUVirtualAddress(),
                m_errorCountBuffer->GetGPUVirtualAddress(),
                m_alignedSize);
            break;
        case ValidationType::RAND_EXCLUSIVE:
            m_validateRandomExclusive->Dispatch(
                m_cmdList,
                m_scanBuffer->GetGPUVirtualAddress(),
                m_scanValidationBuffer->GetGPUVirtualAddress(),
                m_errorCountBuffer->GetGPUVirtualAddress(),
                m_alignedSize);
            break;
        }

        UAVBarrierSingle(m_cmdList, m_errorCountBuffer);
        ExecuteCommandList();

        ReadbackPreBarrier(m_cmdList, m_errorCountBuffer);
        m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_errorCountBuffer.get(), 0, sizeof(uint32_t));
        ReadbackPostBarrier(m_cmdList, m_errorCountBuffer);
        ExecuteCommandList();
        std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, 1);
        uint32_t errCount = vecOut[0];

        if (shouldPrint)
        {
            printf(k_scanName);
            PrintScanTestType(valType);
            if (errCount)
                printf("failed at size %u with %u errors. \n", m_alignedSize, errCount);
            else
                printf("passed at size %u. \n", m_alignedSize);
        }

        return !errCount;
    }

    void ReadBackOutput()
    {
        uint64_t readBackSize = m_alignedSize < k_maxReadBack ? m_alignedSize : k_maxReadBack;
        ReadbackPreBarrier(m_cmdList, m_scanBuffer);
        m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_scanBuffer.get(), 0, readBackSize * sizeof(uint32_t));
        ReadbackPostBarrier(m_cmdList, m_scanBuffer);
        ExecuteCommandList();
        std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, (uint32_t)readBackSize);
        for (uint32_t i = 0; i < vecOut.size(); ++i)
            printf("%u %u \n", i, vecOut[i]);
    }

    bool ValidateScan(uint32_t size, uint32_t seed, ValidationType valType)
    {
        UpdateSize(size, valType);
        CreateTestInput(seed, valType);
        if ((uint32_t)valType & 1)
            PrepareScanCmdListExclusive();
        else
            PrepareScanCmdListInclusive();
        ExecuteCommandList();
        return ValidateOutput(false, valType);
    }

    double TimeScan(uint32_t seed, ValidationType valType)
    {
        CreateTestInput(seed, valType);
        m_cmdList->EndQuery(m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        if ((uint32_t)valType & 1)
            PrepareScanCmdListExclusive();
        else
            PrepareScanCmdListInclusive();
        m_cmdList->EndQuery(m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        ExecuteCommandList();

        m_cmdList->ResolveQueryData(m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, m_readBackBuffer.get(), 0);
        ExecuteCommandList();

        std::vector<uint64_t> vecOut = ReadBackTiming(m_readBackBuffer);
        uint64_t diff = vecOut[1] - vecOut[0];
        return diff / (double)m_timestampFrequency;
    }

    static void PrintScanTestType(ValidationType valType)
    {
        switch (valType)
        {
        case ValidationType::ONE_INCLUSIVE:
            printf("initialized to 1, inclusive, ");
            break;
        case ValidationType::ONE_EXCLUSIVE:
            printf("initialized to 1, exclusive, ");
            break;
        case ValidationType::RAND_INCLUSIVE:
            printf("initialized to random, inclusive, ");
            break;
        case ValidationType::RAND_EXCLUSIVE:
            printf("initialized to random, exclusive, ");
            break;
        }
    }

    static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
    {
        return (x + y - 1) / y;
    }

    static inline uint32_t align16(uint32_t toAlign)
    {
        return (toAlign + 3) / 4 * 4;
    }

    static inline uint32_t vectorizeAlignedSize(uint32_t alignedSize)
    {
        return alignedSize / 4;
    }
};