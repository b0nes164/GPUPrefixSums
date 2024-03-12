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
#include "ComputeShader.h"

namespace UtilityKernels
{
	enum class Reg
	{
		Scan = 0,
		ScanValidation = 1,
		ErrorCount = 2,
	};

    class InitOne
    {
        ComputeShader* shader;
    public:
        InitOne(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(2);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/Utility.hlsl",
                L"InitOne",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->Dispatch(256, 1, 1);
        }
    };

    class InitRandom
    {
        ComputeShader* shader;
    public:
        InitRandom(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ScanValidation);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/Utility.hlsl",
                L"InitRandom",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS scanValidationBuffer,
            const uint32_t& size,
            const uint32_t& seed)
        {
            std::array<uint32_t, 4> t = { size, 0, seed, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, scanValidationBuffer);
            cmdList->Dispatch(256, 1, 1);
        }
    };

    class ClearErrorCount
    {
        ComputeShader* shader;
    public:
        explicit ClearErrorCount(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(1);
            rootParameters[0].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/Utility.hlsl",
                L"ClearErrorCount",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS errorCount)
        {
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRootUnorderedAccessView(0, errorCount);
            cmdList->Dispatch(1, 1, 1);
        }
    };

    class ValidateOneInclusive
    {
        ComputeShader* shader;
    public:
        ValidateOneInclusive(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/Utility.hlsl",
                L"ValidateOneInclusive",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS errorCount,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, errorCount);
            cmdList->Dispatch(256, 1, 1);
        }
    };

    class ValidateOneExclusive
    {
        ComputeShader* shader;
    public:
        ValidateOneExclusive(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/Utility.hlsl",
                L"ValidateOneExclusive",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS errorCount,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, errorCount);
            cmdList->Dispatch(256, 1, 1);
        }
    };

    class ValidateRandomExclusive
    {
        ComputeShader* shader;
    public:
        ValidateRandomExclusive(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ScanValidation);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/Utility.hlsl",
                L"ValidateRandomExclusive",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS scanValidationBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS errorCount,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, scanValidationBuffer);
            cmdList->SetComputeRootUnorderedAccessView(3, errorCount);
            cmdList->Dispatch(1, 1, 1);
        }
    };

    class ValidateRandomInclusive
    {
        ComputeShader* shader;
    public:
        ValidateRandomInclusive(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ScanValidation);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/Utility.hlsl",
                L"ValidateRandomInclusive",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS scanValidationBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS errorCount,
            const uint32_t& size)
        {
            std::array<uint32_t, 4> t = { size, 0, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, scanValidationBuffer);
            cmdList->SetComputeRootUnorderedAccessView(3, errorCount);
            cmdList->Dispatch(1, 1, 1);
        }
    };
}