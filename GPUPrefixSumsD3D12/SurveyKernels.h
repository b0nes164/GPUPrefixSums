#pragma once
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
#include "SurveyKernelBase.h"

namespace SurveyKernels
{
    class SerialInclusive : public SurveyKernelBase
    {
    public:
        SerialInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"SerialInclusive")
        {
        }
    };

    class SerialExclusive : public SurveyKernelBase
    {
    public:
        SerialExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"SerialExclusive")
        {
        }
    };

    class WaveKoggeStoneInclusive : public SurveyKernelBase
    {
    public:
        WaveKoggeStoneInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"WaveKoggeStoneInclusive")
        {
        }
    };

    class WaveKoggeStoneExclusive : public SurveyKernelBase
    {
    public:
        WaveKoggeStoneExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"WaveKoggeStoneExclusive")
        {
        }
    };

    class WaveKoggeStoneShuffleInclusive : public SurveyKernelBase
    {
    public:
        WaveKoggeStoneShuffleInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"WaveKoggeStoneShuffleInclusive")
        {
        }
    };

    class WaveKoggeStoneShuffleExclusive : public SurveyKernelBase
    {
    public:
        WaveKoggeStoneShuffleExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"WaveKoggeStoneShuffleExclusive")
        {
        }
    };

    class WaveKoggeStoneIntrinsicInclusive : public SurveyKernelBase
    {
    public:
        WaveKoggeStoneIntrinsicInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"WaveKoggeStoneIntrinsicInclusive")
        {
        }
    };

    class WaveKoggeStoneIntrinsicExclusive : public SurveyKernelBase
    {
    public:
        WaveKoggeStoneIntrinsicExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"WaveKoggeStoneIntrinsicExclusive")
        {
        }
    };

    class WaveRakingReduceInclusive : public SurveyKernelBase
    {
    public:
        WaveRakingReduceInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"WaveRakingReduceInclusive")
        {
        }
    };

    class WaveRakingReduceExclusive : public SurveyKernelBase
    {
    public:
        WaveRakingReduceExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"WaveRakingReduceExclusive")
        {
        }
    };

    class BlockKoggeStoneInclusive : public SurveyKernelBase
    {
    public:
        BlockKoggeStoneInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockKoggeStoneInclusive")
        {
        }
    };

    class BlockKoggeStoneExclusive : public SurveyKernelBase
    {
    public:
        BlockKoggeStoneExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockKoggeStoneExclusive")
        {
        }
    };

    class BlockSklanskyInclusive : public SurveyKernelBase
    {
    public:
        BlockSklanskyInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockSklanskyInclusive")
        {
        }
    };

    class BlockSklanskyExclusive : public SurveyKernelBase
    {
    public:
        BlockSklanskyExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockSklanskyExclusive")
        {
        }
    };

    class BlockBrentKungBlellochInclusive : public SurveyKernelBase
    {
    public:
        BlockBrentKungBlellochInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockBrentKungBlellochInclusive")
        {
        }
    };

    class BlockBrentKungBlellochExclusive : public SurveyKernelBase
    {
    public:
        BlockBrentKungBlellochExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockBrentKungBlellochExclusive")
        {
        }
    };

    class BlockReduceScanInclusive : public SurveyKernelBase
    {
    public:
        BlockReduceScanInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockReduceScanInclusive")
        {
        }
    };

    class BlockReduceScanExclusive : public SurveyKernelBase
    {
    public:
        BlockReduceScanExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockReduceScanExclusive")
        {
        }
    };

    class BlockBrentKungIntrinsicInclusive : public SurveyKernelBase
    {
    public:
        BlockBrentKungIntrinsicInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockBrentKungIntrinsicInclusive")
        {
        }
    };

    class BlockBrentKungIntrinsicExclusive : public SurveyKernelBase
    {
    public:
        BlockBrentKungIntrinsicExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockBrentKungIntrinsicExclusive")
        {
        }
    };

    class BlockBrentKungFusedIntrinsicInclusive : public SurveyKernelBase
    {
    public:
        BlockBrentKungFusedIntrinsicInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockBrentKungFusedIntrinsicInclusive")
        {
        }
    };

    class BlockSklanskyIntrinsicInclusive : public SurveyKernelBase
    {
    public:
        BlockSklanskyIntrinsicInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockSklanskyIntrinsicInclusive")
        {
        }
    };

    class BlockSklanskyIntrinsicInclusiveAlt : public SurveyKernelBase
    {
    public:
        BlockSklanskyIntrinsicInclusiveAlt(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockSklanskyIntrinsicInclusiveAlt")
        {
        }
    };

    class BlockSklanskyIntrinsicExclusive : public SurveyKernelBase
    {
    public:
        BlockSklanskyIntrinsicExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockSklanskyIntrinsicExclusive")
        {
        }
    };

    class BlockRakingReduceIntrinsicInclusive : public SurveyKernelBase
    {
    public:
        BlockRakingReduceIntrinsicInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockRakingReduceIntrinsicInclusive")
        {
        }
    };

    class BlockRakingReduceIntrinsicExclusive : public SurveyKernelBase
    {
    public:
        BlockRakingReduceIntrinsicExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"BlockRakingReduceIntrinsicExclusive")
        {
        }
    };

    class SharedBrentKungFusedIntrinsicInclusive : public SurveyKernelBase
    {
    public:
        SharedBrentKungFusedIntrinsicInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"SharedBrentKungFusedIntrinsicInclusive")
        {
        }
    };

    class SharedBrentKungFusedIntrinsicExclusive : public SurveyKernelBase
    {
    public:
        SharedBrentKungFusedIntrinsicExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            SurveyKernelBase(
                device,
                info,
                compileArguments,
                shaderPath,
                L"SharedBrentKungFusedIntrinsicExclusive")
        {
        }
    };
}