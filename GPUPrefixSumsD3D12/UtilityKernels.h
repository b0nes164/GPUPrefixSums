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
#include "ComputeKernelBase.h"

namespace UtilityKernels
{
	enum class Reg
	{
		Scan = 0,
		ScanValidation = 1,
		ErrorCount = 2,
        ValidationInfo = 3,
	};

    class InitOne : public ComputeKernelBase
    {
    public:
        InitOne(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"InitOne",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanBuffer,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->Dispatch(256, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(2);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            return rootParams;
        }
    };

    class InitRandom : public ComputeKernelBase
    {
    public:
        InitRandom(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"InitRandom",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanValidationBuffer,
            const uint32_t& size,
            const uint32_t& seed)
        {
            std::array<uint32_t, 4> t = { size, 0, seed, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, scanValidationBuffer);
            cmdList->Dispatch(256, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ScanValidation);
            return rootParameters;
        }
    };

    class ClearErrorCount : ComputeKernelBase
    {
    public:
        ClearErrorCount(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"ClearErrorCount",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& errorCount)
        {
            SetPipelineState(cmdList);
            cmdList->SetComputeRootUnorderedAccessView(0, errorCount);
            cmdList->Dispatch(1, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(1);
            rootParameters[0].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);
            return rootParameters;
        }
    };

    class ValidateOneInclusive : public ComputeKernelBase
    {
    public:
        ValidateOneInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"ValidateOneInclusive",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& errorCount,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, errorCount);
            cmdList->Dispatch(256, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);
            return rootParameters;
        }
    };

    class ValidateOneExclusive : ComputeKernelBase
    {
    public:
        ValidateOneExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"ValidateOneExclusive",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& errorCount,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, errorCount);
            cmdList->Dispatch(256, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);
            return rootParameters;
        }
    };

    class ValidateRandomExclusive : public ComputeKernelBase
    {
    public:
        ValidateRandomExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"ValidateRandomExclusive",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanValidationBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& errorCount,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, scanValidationBuffer);
            cmdList->SetComputeRootUnorderedAccessView(3, errorCount);
            cmdList->Dispatch(1, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ScanValidation);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);
            return rootParameters;
        }
    };

    class ValidateRandomInclusive : ComputeKernelBase
    {
    public:
        ValidateRandomInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"ValidateRandomInclusive",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanValidationBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& errorCount,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, scanValidationBuffer);
            cmdList->SetComputeRootUnorderedAccessView(3, errorCount);
            cmdList->Dispatch(1, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ScanValidation);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);
            return rootParameters;
        }
    };
}