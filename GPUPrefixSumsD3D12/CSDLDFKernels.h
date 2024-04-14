/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/13/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "pch.h"
#include "ComputeKernelBase.h"
#include "Utils.h"

namespace CSDLDFKernels
{
    enum class Reg
    {
        Scan = 0,
        Index = 1,
        ThreadBlockReduction = 2,
    };

    class InitCSDLDF : ComputeKernelBase
    {
    public:
        InitCSDLDF(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"InitCSDLDF",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& indexBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& threadBlockReductionBuffer,
            const uint32_t& threadBlocks)
        {
            std::array<uint32_t, 4> t = { 0, threadBlocks, 0, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, indexBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, threadBlockReductionBuffer);
            cmdList->Dispatch(256, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Index);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);
            return rootParameters;
        }
    };

    class CSDLDFExclusive : ComputeKernelBase
    {
    public:
        CSDLDFExclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"ChainedScanDecoupledLookbackDecoupledFallbackExclusive",
                compileArguments,
                CreateRootParameters())
        {
        }

        //Setting the bitPartition flag is unecessary for CSDL, but splitting
        //the dispatches is still necessary
        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& indexBuffer,
            winrt::com_ptr<ID3D12Resource> threadBlockReductionBuffer,
            const uint32_t& vectorizedSize,
            const uint32_t& threadBlocks)
        {
            const uint32_t fullBlocks = threadBlocks / k_maxDim;
            if (fullBlocks)
            {
                std::array<uint32_t, 4> t = {
                    vectorizedSize,
                    threadBlocks,
                    0,
                    0 };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, indexBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3,
                    threadBlockReductionBuffer->GetGPUVirtualAddress());
                cmdList->Dispatch(k_maxDim, fullBlocks, 1);

                //To stop unecessary spinning of the lookback, add a barrier here on the reductions 
                //As threadblocks in the second dispatch are dependent on the first dispatch
                UAVBarrierSingle(cmdList, threadBlockReductionBuffer);
            }

            const uint32_t partialBlocks = threadBlocks - fullBlocks * k_maxDim;
            if (partialBlocks)
            {
                std::array<uint32_t, 4> t = {
                    vectorizedSize,
                    threadBlocks,
                    0,
                    0 };
                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, indexBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3,
                    threadBlockReductionBuffer->GetGPUVirtualAddress());
                cmdList->Dispatch(partialBlocks, 1, 1);
            }
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::Index);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);
            return rootParameters;
        }
    };

    class CSDLDFInclusive : ComputeKernelBase
    {
    public:
        CSDLDFInclusive(
            winrt::com_ptr<ID3D12Device> device,
            const GPUPrefixSums::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"ChainedScanDecoupledLookbackDecoupledFallbackInclusive",
                compileArguments,
                CreateRootParameters())
        {
        }

        //Setting the bitPartition flag is unecessary for CSDL, but splitting
        //the dispatches is still necessary
        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& scanBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& indexBuffer,
            winrt::com_ptr<ID3D12Resource> threadBlockReductionBuffer,
            const uint32_t& vectorizedSize,
            const uint32_t& threadBlocks)
        {
            const uint32_t fullBlocks = threadBlocks / k_maxDim;
            if (fullBlocks)
            {
                std::array<uint32_t, 4> t = {
                    vectorizedSize,
                    threadBlocks,
                    0,
                    0 };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, indexBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3,
                    threadBlockReductionBuffer->GetGPUVirtualAddress());
                cmdList->Dispatch(k_maxDim, fullBlocks, 1);

                //To stop unecessary spinning of the lookback, add a barrier here on the reductions 
                //As threadblocks in the second dispatch are dependent on the first dispatch
                UAVBarrierSingle(cmdList, threadBlockReductionBuffer);
            }

            const uint32_t partialBlocks = threadBlocks - fullBlocks * k_maxDim;
            if (partialBlocks)
            {
                std::array<uint32_t, 4> t = {
                    vectorizedSize,
                    threadBlocks,
                    0,
                    0 };
                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, scanBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, indexBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3,
                    threadBlockReductionBuffer->GetGPUVirtualAddress());
                cmdList->Dispatch(partialBlocks, 1, 1);
            }
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Scan);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::Index);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);
            return rootParameters;
        }
    };
}