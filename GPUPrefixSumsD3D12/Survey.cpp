/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/4/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#include "pch.h"
#include "Survey.h"

Survey::Survey(
	winrt::com_ptr<ID3D12Device> _device,
	const GPUPrefixSums::DeviceInfo _deviceInfo)
{
	m_device.copy_from(_device.get());
	m_devInfo = _deviceInfo;

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

	InitStaticBuffers();
}

Survey::~Survey()
{
}

bool Survey::TestSerialInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"SerialInclusive",
		m_serialInclusive,
		32,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestSerialExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"SerialExclusive",
		m_serialExclusive,
		32,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestWaveKoggeStoneInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"WaveKoggeStoneInclusive",
		m_waveKoggeStoneInclusive,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestWaveKoggeStoneExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"WaveKoggeStoneExclusive",
		m_waveKoggeStoneExclusive,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestWaveKoggeStoneShuffleInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"WaveKoggeStoneShuffleInclusive",
		m_waveKoggeStoneShuffleInclusive,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestWaveKoggeStoneShuffleExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"WaveKoggeStoneShuffleExclusive",
		m_waveKoggeStoneShuffleExclusive,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestWaveKoggeStoneIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"WaveKoggeStoneIntrinsicInclusive",
		m_waveKoggeStoneIntrinsicInclusive,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestWaveKoggeStoneIntrinsicExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"WaveKoggeStoneIntrinsicExclusive",
		m_waveKoggeStoneIntrinsicExclusive,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestWaveRakingReduceInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"WaveRakingReduceInclusive",
		m_waveRakingReduceInclusive,
		k_maxWaveSize * 2,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestWaveRakingReduceExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"WaveRakingReduceExclusive",
		m_waveRakingReduceExclusive,
		k_maxWaveSize * 2,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockKoggeStoneInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"BlockKoggeStoneInclusive",
		m_blockKoggeStoneInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockKoggeStoneExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"BlockKoggeStoneExclusive",
		m_blockKoggeStoneExclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockSklanskyInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"BlockSklanskyInclusive",
		m_blockSklanskyInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockSklanskyExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"BlockSklanskyExclusive",
		m_blockSklanskyExclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockBrentKungBlellochInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"BlockBrentKungBlellochInclusive",
		m_blockBrentKungBlellochInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockBrentKungBlellochExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"BlockBrentKungBlellochExclusive",
		m_blockBrentKungBlellochExclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockReduceScanInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"BlockReduceScanInclusive",
		m_blockReduceScanInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockReduceScanExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"BlockReduceScanExclusive",
		m_blockReduceScanExclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockBrentKungIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"BlockBrentKungIntrinsicInclusive",
		m_blockBrentKungIntrinsicInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockBrentKungIntrinsicExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"BlockBrentKungIntrinsicExclusive",
		m_blockBrentKungIntrinsicExclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockBrentKungFusedIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"BlockBrentKungFusedIntrinsicInclusive",
		m_blockBrentKungFusedIntrinsicInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockSklanskyIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"BlockSklanskyIntrinsicInclusive",
		m_blockSklanskyIntrinsicInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockSklanskyIntrinsicInclusiveAlt(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"BlockSklanskyIntrinsicInclusiveAlt",
		m_blockSklanskyIntrinsicInclusiveAlt,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockSklanskyIntrinsicExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"BlockSklanskyIntrinsicExclusive",
		m_blockSklanskyIntrinsicExclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockRakingReduceIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"BlockRakingReduceIntrinsicInclusive",
		m_blockRakingReduceIntrinsicInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestBlockRakingReduceIntrinsicExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"BlockRakingReduceIntrinsicExclusive",
		m_blockRakingReduceIntrinsicExclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestSharedBrentKungFusedIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"SharedBrentKungFusedIntrinsicInclusive",
		m_sharedBrentKungFusedIntrinsicInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestSharedBrentKungFusedIntrinsicExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"SharedBrentKungFusedIntrinsicExclusive",
		m_sharedBrentKungFusedIntrinsicExclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestSharedRakingReduceIntrinsicInclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"SharedRakingReduceIntrinsicInclusive",
		m_sharedRakingReduceIntrinsicInclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestSharedRakingReduceIntrinsicExclusive(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"SharedRakingReduceIntrinsicExclusive",
		m_sharedRakingReduceIntrinsicExclusive,
		k_groupSize,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestTrueBlockInclusiveScan(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanInclusive(
		"TrueBlockInclusiveScan",
		m_trueBlockInclusiveScan,
		65536,
		shouldPrint,
		shouldPrintValidation);
}

bool Survey::TestTrueBlockExclusiveScan(bool shouldPrint, bool shouldPrintValidation)
{
	return TestScanExclusive(
		"TrueBlockExclusiveScan",
		m_trueBlockExclusiveScan,
		65536,
		shouldPrint,
		shouldPrintValidation);
}

void Survey::TestAll()
{
	uint32_t testsPassed = 0;
	printf("\n");

	//Serial
	testsPassed += TestSerialInclusive(false, true);
	testsPassed += TestSerialExclusive(false, true);

	//Wave
	testsPassed += TestWaveKoggeStoneInclusive(false, true);
	testsPassed += TestWaveKoggeStoneExclusive(false, true);

	testsPassed += TestWaveKoggeStoneShuffleInclusive(false, true);
	testsPassed += TestWaveKoggeStoneShuffleExclusive(false, true);

	testsPassed += TestWaveKoggeStoneIntrinsicInclusive(false, true);
	testsPassed += TestWaveKoggeStoneIntrinsicExclusive(false, true);

	testsPassed += TestWaveRakingReduceInclusive(false, true);
	testsPassed += TestWaveRakingReduceExclusive(false, true);

	//Block
	testsPassed += TestBlockKoggeStoneInclusive(false, true);
	testsPassed += TestBlockKoggeStoneExclusive(false, true);

	testsPassed += TestBlockSklanskyInclusive(false, true);
	testsPassed += TestBlockSklanskyExclusive(false, true);

	testsPassed += TestBlockBrentKungBlellochInclusive(false, true);
	testsPassed += TestBlockBrentKungBlellochExclusive(false, true);

	testsPassed += TestBlockReduceScanInclusive(false, true);
	testsPassed += TestBlockReduceScanExclusive(false, true);

	testsPassed += TestBlockBrentKungIntrinsicInclusive(false, true);
	if (m_devInfo.SIMDWidth >= 8)
		testsPassed += TestBlockBrentKungIntrinsicExclusive(false, true);
	else
		printf("\nDevice min wave width too low for BlockBrentKungIntrinsicExclusive\n\n");

	testsPassed += TestBlockBrentKungFusedIntrinsicInclusive(false, true);

	testsPassed += TestBlockSklanskyIntrinsicInclusive(false, true);
	testsPassed += TestBlockSklanskyIntrinsicInclusiveAlt(false, true);
	testsPassed += TestBlockSklanskyIntrinsicExclusive(false, true);

	if (m_devInfo.SIMDWidth >= 8)
		testsPassed += TestBlockRakingReduceIntrinsicInclusive(false, true);
	else
		printf("\nDevice min wave width too low for BlockRakingReduceIntrinsicInclusive\n\n");
	if (m_devInfo.SIMDWidth >= 8)
		testsPassed += TestBlockRakingReduceIntrinsicExclusive(false, true);
	else
		printf("\nDevice min wave width too low for BlockRakingReduceIntrinsicExclusive\n\n");

	testsPassed += TestSharedBrentKungFusedIntrinsicInclusive(false, true);
	testsPassed += TestSharedBrentKungFusedIntrinsicExclusive(false, true);

	if (m_devInfo.SIMDWidth >= 8)
		testsPassed += TestSharedRakingReduceIntrinsicInclusive(false, true);
	else
		printf("\nDevice min wave width too low for SharedRakingReduceIntrinsicInclusive\n\n");
	if (m_devInfo.SIMDWidth >= 8)
		testsPassed += TestSharedRakingReduceIntrinsicExclusive(false, true);
	else
		printf("\nDevice min wave width too low for SharedRakingReduceIntrinsicExclusive\n\n");

	testsPassed += TestTrueBlockInclusiveScan(false, true);
	testsPassed += TestTrueBlockExclusiveScan(false, true);
}

void Survey::InitUtilityShaders()
{
	const std::filesystem::path path = "Shaders/Utility.hlsl";
	m_initOne = new UtilityKernels::InitOne(m_device, m_devInfo, m_compileArguments, path);
	m_clearErrorCount = new UtilityKernels::ClearErrorCount(m_device, m_devInfo, m_compileArguments, path);
	m_validateInclusive = new UtilityKernels::ValidateOneInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_validateExclusive = new UtilityKernels::ValidateOneExclusive(m_device, m_devInfo, m_compileArguments, path);
}

void Survey::InitComputeShaders()
{
	const std::filesystem::path path = "Shaders/Survey.hlsl";

	//Serial
	m_serialInclusive = new SurveyKernels::SerialInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_serialExclusive = new SurveyKernels::SerialExclusive(m_device, m_devInfo, m_compileArguments, path);

	//Wave
	m_waveKoggeStoneInclusive = new SurveyKernels::WaveKoggeStoneInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_waveKoggeStoneExclusive = new SurveyKernels::WaveKoggeStoneExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_waveKoggeStoneShuffleInclusive = new SurveyKernels::WaveKoggeStoneShuffleInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_waveKoggeStoneShuffleExclusive = new SurveyKernels::WaveKoggeStoneShuffleExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_waveKoggeStoneIntrinsicInclusive = new SurveyKernels::WaveKoggeStoneIntrinsicInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_waveKoggeStoneIntrinsicExclusive = new SurveyKernels::WaveKoggeStoneIntrinsicExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_waveRakingReduceInclusive = new SurveyKernels::WaveRakingReduceInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_waveRakingReduceExclusive = new SurveyKernels::WaveRakingReduceExclusive(m_device, m_devInfo, m_compileArguments, path);

	//Block
	m_blockKoggeStoneInclusive = new SurveyKernels::BlockKoggeStoneInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_blockKoggeStoneExclusive = new SurveyKernels::BlockKoggeStoneExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_blockSklanskyInclusive = new SurveyKernels::BlockSklanskyInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_blockSklanskyExclusive = new SurveyKernels::BlockSklanskyExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_blockBrentKungBlellochInclusive = 
		new SurveyKernels::BlockBrentKungBlellochInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_blockBrentKungBlellochExclusive = 
		new SurveyKernels::BlockBrentKungBlellochExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_blockReduceScanInclusive = new SurveyKernels::BlockReduceScanInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_blockReduceScanExclusive = new SurveyKernels::BlockReduceScanExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_blockBrentKungIntrinsicInclusive = new SurveyKernels::BlockBrentKungIntrinsicInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_blockBrentKungIntrinsicExclusive = new SurveyKernels::BlockBrentKungIntrinsicExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_blockBrentKungFusedIntrinsicInclusive 
		= new SurveyKernels::BlockBrentKungFusedIntrinsicInclusive(m_device, m_devInfo, m_compileArguments, path);

	m_blockSklanskyIntrinsicInclusive = new SurveyKernels::BlockSklanskyIntrinsicInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_blockSklanskyIntrinsicInclusiveAlt = 
		new SurveyKernels::BlockSklanskyIntrinsicInclusiveAlt(m_device, m_devInfo, m_compileArguments, path);
	m_blockSklanskyIntrinsicExclusive = new SurveyKernels::BlockSklanskyIntrinsicExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_blockRakingReduceIntrinsicInclusive = new SurveyKernels::BlockRakingReduceIntrinsicInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_blockRakingReduceIntrinsicExclusive = new SurveyKernels::BlockRakingReduceIntrinsicExclusive(m_device, m_devInfo, m_compileArguments, path);

	//Shared
	m_sharedBrentKungFusedIntrinsicInclusive = new SurveyKernels::SharedBrentKungFusedIntrinsicInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_sharedBrentKungFusedIntrinsicExclusive = new SurveyKernels::SharedBrentKungFusedIntrinsicExclusive(m_device, m_devInfo, m_compileArguments, path);

	m_sharedRakingReduceIntrinsicInclusive = new SurveyKernels::SharedRakingReduceIntrinsicInclusive(m_device, m_devInfo, m_compileArguments, path);
	m_sharedRakingReduceIntrinsicExclusive = new SurveyKernels::SharedRakingReduceIntrinsicExclusive(m_device, m_devInfo, m_compileArguments, path);

	//True
	m_trueBlockInclusiveScan = new SurveyKernels::TrueBlockInclusiveScan(m_device, m_devInfo, m_compileArguments, path);
	m_trueBlockExclusiveScan = new SurveyKernels::TrueBlockExclusiveScan(m_device, m_devInfo, m_compileArguments, path);
}

void Survey::UpdateSize(uint32_t size)
{
	if (m_size != size)
	{
		m_size = size;
		DisposeBuffers();
		InitBuffers(m_size);
	}
}

void Survey::DisposeBuffers()
{
	m_scanBuffer = nullptr;
}

void Survey::InitStaticBuffers()
{
	m_errorCountBuffer = CreateBuffer(
		m_device,
		sizeof(uint32_t),
		D3D12_HEAP_TYPE_DEFAULT,
		D3D12_RESOURCE_STATE_COMMON,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	m_validationInfoBuffer = CreateBuffer(
		m_device,
		sizeof(uint32_t),
		D3D12_HEAP_TYPE_DEFAULT,
		D3D12_RESOURCE_STATE_COMMON,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	m_readBackBuffer = CreateBuffer(
		m_device,
		k_maxReadBack * sizeof(uint32_t),
		D3D12_HEAP_TYPE_READBACK,
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_RESOURCE_FLAG_NONE);
}

void Survey::InitBuffers(uint32_t size)
{
	m_scanBuffer = CreateBuffer(
		m_device,
		size * sizeof(uint32_t),
		D3D12_HEAP_TYPE_DEFAULT,
		D3D12_RESOURCE_STATE_COMMON,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
}

void Survey::CreateTestInput()
{
	m_initOne->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_size);
	UAVBarrierSingle(m_cmdList, m_scanBuffer);
}

void Survey::ExecuteCommandList()
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

bool Survey::ValidateInclusive(
	const char* scanName,
	uint32_t testSize,
	bool shouldPrint)
{
	m_clearErrorCount->Dispatch(
		m_cmdList,
		m_errorCountBuffer->GetGPUVirtualAddress());
	UAVBarrierSingle(m_cmdList, m_errorCountBuffer);

	m_validateInclusive->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_errorCountBuffer->GetGPUVirtualAddress(),
		testSize);

	ReadbackPreBarrier(m_cmdList, m_errorCountBuffer);
	m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_errorCountBuffer.get(), 0, sizeof(uint32_t));
	ReadbackPostBarrier(m_cmdList, m_errorCountBuffer);
	ExecuteCommandList();
	std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, 1);
	uint32_t errCount = vecOut[0];

	if (shouldPrint)
	{
		printf(scanName);
		if (errCount)
			printf(" failed at size %u with %u errors. \n", testSize, errCount);
		else
			printf(" passed at size %u. \n", testSize);
	}

	return !errCount;
}

bool Survey::ValidateExclusive(
	const char* scanName,
	uint32_t testSize,
	bool shouldPrint)
{
	m_clearErrorCount->Dispatch(
		m_cmdList,
		m_errorCountBuffer->GetGPUVirtualAddress());
	UAVBarrierSingle(m_cmdList, m_errorCountBuffer);

	m_validateExclusive->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_errorCountBuffer->GetGPUVirtualAddress(),
		testSize);

	ReadbackPreBarrier(m_cmdList, m_errorCountBuffer);
	m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_errorCountBuffer.get(), 0, sizeof(uint32_t));
	ReadbackPostBarrier(m_cmdList, m_errorCountBuffer);
	ExecuteCommandList();
	std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, 1);
	uint32_t errCount = vecOut[0];

	if (shouldPrint)
	{
		printf(scanName);
		if (errCount)
			printf(" failed at size %u with %u errors. \n", testSize, errCount);
		else
			printf(" passed at size %u. \n", testSize);
	}

	return !errCount;
}

//Figure out how big the wave was that performed the scan
//Note that depending on the device, this value can vary between shader executions
uint32_t Survey::ReadBackValidationInfo()
{
	ReadbackPreBarrier(m_cmdList, m_validationInfoBuffer);
	m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_validationInfoBuffer.get(), 0, sizeof(uint32_t));
	ReadbackPostBarrier(m_cmdList, m_validationInfoBuffer);
	ExecuteCommandList();
	std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, sizeof(uint32_t));
	return vecOut[0];
} 

void Survey::ReadBackOutput()
{
	uint64_t readBackSize = m_size < k_maxReadBack ? m_size : k_maxReadBack;
	ReadbackPreBarrier(m_cmdList, m_scanBuffer);
	m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_scanBuffer.get(), 0, readBackSize * sizeof(uint32_t));
	ReadbackPostBarrier(m_cmdList, m_scanBuffer);
	ExecuteCommandList();
	std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, (uint32_t)readBackSize);
	for (uint32_t i = 0; i < vecOut.size(); ++i)
		printf("%u %u \n", i, vecOut[i]);
}

bool Survey::TestScanInclusive(
	const char* scanName,
	SurveyKernelBase* scan,
	const bool& shouldPrint,
	const bool& shouldPrintValidation)
{
	UpdateSize(k_maxWaveSize);
	CreateTestInput();
	scan->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_validationInfoBuffer->GetGPUVirtualAddress(),
		k_maxWaveSize);
	UAVBarrierSingle(m_cmdList, m_scanBuffer);

	if (shouldPrint)
		ReadBackOutput();

	return ValidateInclusive(
		scanName,
		ReadBackValidationInfo(),
		shouldPrintValidation);
}

bool Survey::TestScanInclusive(
	const char* scanName,
	SurveyKernelBase* scan,
	const uint32_t& size,
	const bool& shouldPrint,
	const bool& shouldPrintValidation)
{
	UpdateSize(size);
	CreateTestInput();
	scan->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_validationInfoBuffer->GetGPUVirtualAddress(),
		size);
	UAVBarrierSingle(m_cmdList, m_scanBuffer);

	if (shouldPrint)
		ReadBackOutput();

	return ValidateInclusive(
		scanName,
		size,
		shouldPrintValidation);
}

bool Survey::TestScanExclusive(
	const char* scanName,
	SurveyKernelBase* scan,
	const bool& shouldPrint,
	const bool& shouldPrintValidation)
{
	UpdateSize(k_maxWaveSize);
	CreateTestInput();
	scan->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_validationInfoBuffer->GetGPUVirtualAddress(),
		k_maxWaveSize);
	UAVBarrierSingle(m_cmdList, m_scanBuffer);

	if (shouldPrint)
		ReadBackOutput();

	return ValidateExclusive(
		scanName,
		ReadBackValidationInfo(),
		shouldPrintValidation);
}

bool Survey::TestScanExclusive(
	const char* scanName,
	SurveyKernelBase* scan,
	const uint32_t& size,
	const bool& shouldPrint,
	const bool& shouldPrintValidation)
{
	UpdateSize(size);
	CreateTestInput();
	scan->Dispatch(
		m_cmdList,
		m_scanBuffer->GetGPUVirtualAddress(),
		m_validationInfoBuffer->GetGPUVirtualAddress(),
		size);
	UAVBarrierSingle(m_cmdList, m_scanBuffer);

	if (shouldPrint)
		ReadBackOutput();

	return ValidateExclusive(
		scanName,
		size,
		shouldPrintValidation);
}
