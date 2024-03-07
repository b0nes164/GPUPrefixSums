/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "ChainedScanDecoupledLookback.cuh"
#include "UtilityKernels.cuh"

class ChainedScanDecoupledLookbackDispatcher
{
	const uint32_t k_maxSize;
	const uint32_t k_partitionSize = 3072;
	const uint32_t k_csdlThreads = 256;

	uint32_t* m_scan;
	uint32_t* m_index;
	uint32_t* m_threadBlockReduction;
	uint32_t* m_errCount;

public:
	ChainedScanDecoupledLookbackDispatcher(uint32_t maxSize) :
		k_maxSize(maxSize)
	{
		const uint32_t maxThreadBlocks = divRoundUp(k_maxSize, k_partitionSize);
		cudaMalloc(&m_scan, k_maxSize * sizeof(uint32_t));
		cudaMalloc(&m_index, sizeof(uint32_t));
		cudaMalloc(&m_threadBlockReduction, maxThreadBlocks * sizeof(uint32_t));
		cudaMalloc(&m_errCount, sizeof(uint32_t));
	}

	//Tests input sizes not perfect multiples of the partition tile size,
	//then tests several large inputs.
	void TestAllExclusive()
	{
		if (k_maxSize < (1 << 28))
		{
			printf("This test requires a minimum initialized size of %u. ", 1 << 28);
			printf("Reinitialize the object to at least %u.\n", 1 << 28);
			return;
		}

		printf("Beginning GPUPrefixSums ChainedScanDecoupledLookback exclusive validation test: \n");
		uint32_t testsPassed = 0;
		for (uint32_t i = k_partitionSize; i < k_partitionSize * 2 + 1; ++i)
		{
			InitOne<<<256, 256 >>> (m_scan, i);
			DispatchKernelsExclusive(i);
			if (DispatchValidateExclusive(i))
				testsPassed++;
			else
				printf("\n Test failed at size %u \n", i);

			if (!(i & 255))
				printf(".");
		}
		printf("\n");

		for (uint32_t i = 26; i <= 28; ++i)
		{
			InitOne<<<256, 256 >>> (m_scan, i);
			DispatchKernelsExclusive(1 << i);
			if (DispatchValidateExclusive(i))
				testsPassed++;
			else
				printf("\n Test failed at size %u \n", 1 << i);
		}

		if (testsPassed == k_partitionSize + 3 + 1)
			printf("%u/%u All tests passed.\n\n", testsPassed, testsPassed);
		else
			printf("%u/%u Test failed.\n\n", testsPassed, k_partitionSize + 3 + 1);
	}

	//Tests input sizes not perfect multiples of the partition tile size,
	//then tests several large inputs.
	void TestAllInclusive()
	{
		if (k_maxSize < (1 << 28))
		{
			printf("This test requires a minimum initialized size of %u. ", 1 << 28);
			printf("Reinitialize the object to at least %u.\n", 1 << 28);
			return;
		}

		printf("Beginning GPUPrefixSums ChainedScanDecoupledLookback inclusive validation test: \n");
		uint32_t testsPassed = 0;
		for (uint32_t i = k_partitionSize; i < k_partitionSize * 2 + 1; ++i)
		{
			InitOne <<<256, 256 >>> (m_scan, i);
			DispatchKernelsInclusive(i);
			if (DispatchValidateInclusive(i))
				testsPassed++;
			else
				printf("\n Test failed at size %u \n", i);

			if (!(i & 255))
				printf(".");
		}
		printf("\n");

		for (uint32_t i = 26; i <= 28; ++i)
		{
			InitOne <<<256, 256 >>> (m_scan, i);
			DispatchKernelsInclusive(1 << i);
			if (DispatchValidateInclusive(i))
				testsPassed++;
			else
				printf("\n Test failed at size %u \n", 1 << i);
		}

		if (testsPassed == k_partitionSize + 3 + 1)
			printf("%u/%u All tests passed.\n\n", testsPassed, testsPassed);
		else
			printf("%u/%u Test failed.\n\n", testsPassed, k_partitionSize + 3 + 1);
	}

	void BatchTimingInclusive(uint32_t size, uint32_t batchCount)
	{
		if (size > k_maxSize)
		{
			printf("Error, requested test size exceeds max initialized size. \n");
			return;
		}

		printf("Beginning GPUPrefixSums ChainedScanDecoupledLookback inclusive batch timing test at:\n");
		printf("Size: %u\n", size);
		printf("Test size: %u\n", batchCount);

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		float totalTime = 0.0f;
		for (uint32_t i = 0; i <= batchCount; ++i)
		{
			InitOne <<<256, 256 >>> (m_scan, size);
			cudaDeviceSynchronize();
			cudaEventRecord(start);
			DispatchKernelsInclusive(size);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);

			float millis;
			cudaEventElapsedTime(&millis, start, stop);
			if (i)
				totalTime += millis;

			if ((i & 15) == 0)
				printf(". ");
		}

		printf("\n");
		totalTime /= 1000.0f;
		printf("Total time elapsed: %f\n", totalTime);
		printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size, size / totalTime * batchCount);
	}

	~ChainedScanDecoupledLookbackDispatcher()
	{
		cudaFree(m_scan);
		cudaFree(m_threadBlockReduction);
		cudaFree(m_index);
		cudaFree(m_errCount);
	}

private:
	static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
	{
		return (x + y - 1) / y;
	}

	static inline uint32_t align16(uint32_t x)
	{
		return divRoundUp(x, 4) * 4;
	}

	void ClearMemory(uint32_t threadBlocks)
	{
		cudaMemset(m_index, 0, sizeof(uint32_t));
		cudaMemset(m_threadBlockReduction, 0, threadBlocks * sizeof(uint32_t));
	}

	void DispatchKernelsExclusive(uint32_t size)
	{
		size = align16(size);
		const uint32_t threadBlocks = divRoundUp(size, k_partitionSize);

		ClearMemory(threadBlocks);

		cudaDeviceSynchronize();

		ChainedScanDecoupledLookback::CSDLExclusive <<<threadBlocks, k_csdlThreads >>> (m_scan,
			m_threadBlockReduction, m_index, size);
	}

	void DispatchKernelsInclusive(uint32_t size)
	{
		size = align16(size);
		const uint32_t threadBlocks = divRoundUp(size, k_partitionSize);

		ClearMemory(threadBlocks);

		cudaDeviceSynchronize();

		ChainedScanDecoupledLookback::CSDLInclusive <<<threadBlocks, k_csdlThreads >>> (m_scan,
			m_threadBlockReduction, m_index, size);
	}

	bool DispatchValidateExclusive(uint32_t size)
	{
		uint32_t errCount[1];
		cudaMemset(m_errCount, 0, sizeof(uint32_t));
		cudaDeviceSynchronize();
		ValidateExclusive<<<256,256>>>(m_scan, m_errCount, size);
		cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		return !errCount[0];
	}

	bool DispatchValidateInclusive(uint32_t size)
	{
		uint32_t errCount[1];
		cudaMemset(m_errCount, 0, sizeof(uint32_t));
		cudaDeviceSynchronize();
		ValidateInclusive <<<256, 256>>> (m_scan, m_errCount, size);
		cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		return !errCount[0];
	}
};