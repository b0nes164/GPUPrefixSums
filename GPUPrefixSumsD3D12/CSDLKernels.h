/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "pch.h"
#include "ComputeKernelBase.h"
#include "Utils.h"

namespace CSDLKernels {
    enum class Reg {
        ScanIn = 0,
        ScanOut = 1,
        ScanBump = 2,
        ThreadBlockReduction = 3,
    };

    class InitCSDL : ComputeKernelBase {
       public:
        InitCSDL(winrt::com_ptr<ID3D12Device> device, const GPUPrefixSums::DeviceInfo& info,
                 const std::vector<std::wstring>& compileArguments,
                 const std::filesystem::path& shaderPath)
            : ComputeKernelBase(device, info, shaderPath, L"InitChainedScan", compileArguments,
                                CreateRootParameters()) {}

        void Dispatch(winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
                      const D3D12_GPU_VIRTUAL_ADDRESS& scanBumpBuffer,
                      const D3D12_GPU_VIRTUAL_ADDRESS& threadBlockReductionBuffer,
                      const uint32_t& threadBlocks) {
            std::array<uint32_t, 4> t = {0, threadBlocks, 0, 0};
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanBumpBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, threadBlockReductionBuffer);
            cmdList->Dispatch(256, 1, 1);
        }

       protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::ScanBump);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);
            return rootParameters;
        }
    };

    class CSDLExclusive : ComputeKernelBase {
       public:
        CSDLExclusive(winrt::com_ptr<ID3D12Device> device, const GPUPrefixSums::DeviceInfo& info,
                      const std::vector<std::wstring>& compileArguments,
                      const std::filesystem::path& shaderPath)
            : ComputeKernelBase(device, info, shaderPath, L"ChainedScanDecoupledLookbackExclusive",
                                compileArguments, CreateRootParameters()) {}

        //Setting the bitPartition flag is unecessary for CSDL, but splitting
        //the dispatches is still necessary
        void Dispatch(winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
                      const D3D12_GPU_VIRTUAL_ADDRESS& scanInBuffer,
                      const D3D12_GPU_VIRTUAL_ADDRESS& scanOutBuffer,
                      const D3D12_GPU_VIRTUAL_ADDRESS& scanBumpBuffer,
                      winrt::com_ptr<ID3D12Resource> threadBlockReductionBuffer,
                      const uint32_t& vectorizedSize, const uint32_t& threadBlocks) {
            const uint32_t fullBlocks = threadBlocks / k_maxDim;
            if (fullBlocks) {
                std::array<uint32_t, 4> t = {vectorizedSize, threadBlocks, 0, 0};

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, scanInBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, scanOutBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, scanBumpBuffer);
                cmdList->SetComputeRootUnorderedAccessView(
                    4, threadBlockReductionBuffer->GetGPUVirtualAddress());
                cmdList->Dispatch(k_maxDim, fullBlocks, 1);

                //To stop unecessary spinning of the lookback, add a barrier here on the reductions
                //As threadblocks in the second dispatch are dependent on the first dispatch
                UAVBarrierSingle(cmdList, threadBlockReductionBuffer);
            }

            const uint32_t partialBlocks = threadBlocks - fullBlocks * k_maxDim;
            if (partialBlocks) {
                std::array<uint32_t, 4> t = {vectorizedSize, threadBlocks, 0, 0};
                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, scanInBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, scanOutBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, scanBumpBuffer);
                cmdList->SetComputeRootUnorderedAccessView(
                    4, threadBlockReductionBuffer->GetGPUVirtualAddress());
                cmdList->Dispatch(partialBlocks, 1, 1);
            }
        }

       protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(5);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::ScanIn);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ScanOut);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ScanBump);
            rootParameters[4].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);
            return rootParameters;
        }
    };

    class CSDLInclusive : ComputeKernelBase {
       public:
        CSDLInclusive(winrt::com_ptr<ID3D12Device> device, const GPUPrefixSums::DeviceInfo& info,
                      const std::vector<std::wstring>& compileArguments,
                      const std::filesystem::path& shaderPath)
            : ComputeKernelBase(device, info, shaderPath, L"ChainedScanDecoupledLookbackInclusive",
                                compileArguments, CreateRootParameters()) {}

        //Setting the bitPartition flag is unecessary for CSDL, but splitting
        //the dispatches is still necessary
        void Dispatch(winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
                      const D3D12_GPU_VIRTUAL_ADDRESS& scanInBuffer,
                      const D3D12_GPU_VIRTUAL_ADDRESS& scanOutBuffer,
                      const D3D12_GPU_VIRTUAL_ADDRESS& scanBumpBuffer,
                      winrt::com_ptr<ID3D12Resource> threadBlockReductionBuffer,
                      const uint32_t& vectorizedSize, const uint32_t& threadBlocks) {
            const uint32_t fullBlocks = threadBlocks / k_maxDim;
            if (fullBlocks) {
                std::array<uint32_t, 4> t = {vectorizedSize, threadBlocks, 0, 0};

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, scanInBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, scanOutBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, scanBumpBuffer);
                cmdList->SetComputeRootUnorderedAccessView(
                    4, threadBlockReductionBuffer->GetGPUVirtualAddress());
                cmdList->Dispatch(k_maxDim, fullBlocks, 1);

                //To stop unecessary spinning of the lookback, add a barrier here on the reductions
                //As threadblocks in the second dispatch are dependent on the first dispatch
                UAVBarrierSingle(cmdList, threadBlockReductionBuffer);
            }

            const uint32_t partialBlocks = threadBlocks - fullBlocks * k_maxDim;
            if (partialBlocks) {
                std::array<uint32_t, 4> t = {vectorizedSize, threadBlocks, 0, 0};
                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, scanInBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, scanOutBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, scanBumpBuffer);
                cmdList->SetComputeRootUnorderedAccessView(
                    4, threadBlockReductionBuffer->GetGPUVirtualAddress());
                cmdList->Dispatch(partialBlocks, 1, 1);
            }
        }

       protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(5);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::ScanIn);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ScanOut);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::ScanBump);
            rootParameters[4].InitAsUnorderedAccessView((UINT)Reg::ThreadBlockReduction);
            return rootParameters;
        }
    };
}  // namespace CSDLKernels