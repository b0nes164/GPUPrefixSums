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

namespace CSDLKernels
{
	enum class Reg
	{
		Scan = 0,
		Index = 1,
		ThreadBlockReduction = 2,
	};

    class InitCSDL
    {
        ComputeShader* shader;
    public:
        InitCSDL(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Index);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/ChainedScanDecoupledLookback.hlsl",
                L"InitChainedScan",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS indexBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS threadBlockReductionBuffer,
            const uint32_t& threadBlocks)
        {
            std::array<uint32_t, 4> t = { 0, threadBlocks, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, indexBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, threadBlockReductionBuffer);
            cmdList->Dispatch(256, 1, 1);
        }
    };

    class CSDLExclusive
    {
        ComputeShader* shader;
    public:
        CSDLExclusive(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::Index);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/ChainedScanDecoupledLookback.hlsl",
                L"ChainedScanDecoupledLookbackExclusive",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS indexBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS threadBlockReductionBuffer,
            const uint32_t& vectorizedSize,
            const uint32_t& threadBlocks)
        {
            std::array<uint32_t, 4> t = { vectorizedSize, threadBlocks, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, indexBuffer);
            cmdList->SetComputeRootUnorderedAccessView(3, threadBlockReductionBuffer);
            cmdList->Dispatch(threadBlocks, 1, 1);
        }
    };

    class CSDLInclusive
    {
        ComputeShader* shader;
    public:
        CSDLInclusive(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::Index);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/ChainedScanDecoupledLookback.hlsl",
                L"ChainedScanDecoupledLookbackInclusive",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS indexBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS threadBlockReductionBuffer,
            const uint32_t& vectorizedSize,
            const uint32_t& threadBlocks)
        {
            std::array<uint32_t, 4> t = { vectorizedSize, threadBlocks, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, indexBuffer);
            cmdList->SetComputeRootUnorderedAccessView(3, threadBlockReductionBuffer);
            cmdList->Dispatch(threadBlocks, 1, 1);
        }
    };
}