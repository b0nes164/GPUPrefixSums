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

class SurveyKernelBase : public ComputeKernelBase {
   public:
    SurveyKernelBase(winrt::com_ptr<ID3D12Device> device, const GPUPrefixSums::DeviceInfo& info,
                     const std::vector<std::wstring>& compileArguments,
                     const std::filesystem::path& shaderPath, const wchar_t* entryPoint)
        : ComputeKernelBase(device, info, shaderPath, entryPoint, compileArguments,
                            CreateRootParameters()) {}

    void Dispatch(winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
                  const D3D12_GPU_VIRTUAL_ADDRESS& prefixSumBuffer,
                  const D3D12_GPU_VIRTUAL_ADDRESS& validationInfoBuffer, const uint32_t& size) {
        std::array<uint32_t, 4> t = {size, 0, 0, 0};
        SetPipelineState(cmdList);
        cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
        cmdList->SetComputeRootUnorderedAccessView(1, prefixSumBuffer);
        cmdList->SetComputeRootUnorderedAccessView(2, validationInfoBuffer);
        cmdList->Dispatch(1, 1, 1);
    }

   protected:
    const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override {
        auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
        rootParameters[0].InitAsConstants(4, 0);
        rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::PrefixSum);
        rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::ValidationInfo);
        return rootParameters;
    }

   private:
    enum class Reg {
        PrefixSum = 0,
        ValidationInfo = 1,
    };
};