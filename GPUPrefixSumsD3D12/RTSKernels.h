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

namespace RTSKernels
{
    enum class Reg
    {
        Scan = 0,
        ThreadBlockReduction = 1,
    };

    class Reduce
    {
        ComputeShader* shader;
    public:
        Reduce(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/ReduceThenScan.hlsl",
                L"Reduce",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS threadBlockReductionBuffer,
            const uint32_t& vectorizedSize,
            const uint32_t& threadBlocks)
        {
            std::array<uint32_t, 4> t = { vectorizedSize, threadBlocks, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, threadBlockReductionBuffer);
            cmdList->Dispatch(threadBlocks, 1, 1);
        }
    };

    class Scan
    {
        ComputeShader* shader;
    public:
        Scan(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(2);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/ReduceThenScan.hlsl",
                L"Scan",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS threadBlockReductionBuffer,
            const uint32_t& threadBlocks)
        {
            std::array<uint32_t, 4> t = { 0, threadBlocks, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, threadBlockReductionBuffer);
            cmdList->Dispatch(1, 1, 1);
        }
    };

    class DownSweepInclusive
    {
        ComputeShader* shader;
    public:
        DownSweepInclusive(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/ReduceThenScan.hlsl",
                L"DownsweepInclusive",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS threadBlockReductionBuffer,
            const uint32_t& vectorizedSize,
            const uint32_t& threadBlocks)
        {
            std::array<uint32_t, 4> t = { vectorizedSize, threadBlocks, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, threadBlockReductionBuffer);
            cmdList->Dispatch(threadBlocks, 1, 1);
        }
    };

    class DownSweepExclusive
    {
        ComputeShader* shader;
    public:
        DownSweepExclusive(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/ReduceThenScan.hlsl",
                L"DownsweepExclusive",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS threadBlockReductionBuffer,
            const uint32_t& vectorizedSize,
            const uint32_t& threadBlocks)
        {
            std::array<uint32_t, 4> t = { vectorizedSize, threadBlocks, 0, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, threadBlockReductionBuffer);
            cmdList->Dispatch(threadBlocks, 1, 1);
        }
    };
}