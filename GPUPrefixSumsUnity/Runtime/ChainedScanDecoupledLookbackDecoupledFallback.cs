/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/10/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Assertions;

namespace GPUPrefixSums.Runtime
{
    public class ChainedScanDecoupledLookbackDecoupledFallback : GPUPrefixSumsBase
    {
        private readonly int m_kernelInit = -1;
        private readonly int m_kernelInclusive = -1;
        private readonly int m_kernelExclusive = -1;

        private readonly bool m_isValid;

        public ChainedScanDecoupledLookbackDecoupledFallback(
            ComputeShader compute,
            int maxElements,
            ref ComputeBuffer tempBuffer0,
            ref ComputeBuffer tempBuffer1)
        {
            m_cs = compute;
            if (m_cs)
            {
                m_kernelInit = m_cs.FindKernel("InitCSDLDF");
                m_kernelInclusive = m_cs.FindKernel("ChainedScanDecoupledLookbackDecoupledFallbackInclusive");
                m_kernelExclusive = m_cs.FindKernel("ChainedScanDecoupledLookbackDecoupledFallbackExclusive");
            }

            m_isValid = m_kernelInit >= 0 &&
                    m_kernelInclusive >= 0 &&
                    m_kernelExclusive >= 0;

            if (m_isValid)
            {
                if (!m_cs.IsSupported(m_kernelInit) ||
                    !m_cs.IsSupported(m_kernelInclusive) ||
                    !m_cs.IsSupported(m_kernelExclusive))
                {
                    m_isValid = false;
                }
            }

            //Allocate the temporary resources
            //Ensure that the allocation is an aligned size
            m_allocatedSize = AlignFour(maxElements);
            Assert.IsTrue(
                m_isValid &&
                m_allocatedSize < k_maxSize &&
                m_allocatedSize > k_minSize);
            AllocateResources(m_allocatedSize, ref tempBuffer0, ref tempBuffer1);
        }

        private void AllocateResources(
            int allocationSize,
            ref ComputeBuffer threadBlockReductionBuffer,
            ref ComputeBuffer indexBuffer)
        {
            indexBuffer?.Dispose();
            threadBlockReductionBuffer?.Dispose();

            indexBuffer = new ComputeBuffer(1, sizeof(uint));
            threadBlockReductionBuffer = new ComputeBuffer(DivRoundUp(allocationSize, k_partitionSize), sizeof(uint));
        }

        private void SetRootParameters(
            ComputeBuffer _scanBuffer,
            ComputeBuffer _threadBlockReductionBuffer,
            ComputeBuffer _indexBuffer)
        {
            m_cs.SetBuffer(m_kernelInit, "b_threadBlockReduction", _threadBlockReductionBuffer);
            m_cs.SetBuffer(m_kernelInit, "b_index", _indexBuffer);

            m_cs.SetBuffer(m_kernelInclusive, "b_scan", _scanBuffer);
            m_cs.SetBuffer(m_kernelInclusive, "b_threadBlockReduction", _threadBlockReductionBuffer);
            m_cs.SetBuffer(m_kernelInclusive, "b_index", _indexBuffer);

            m_cs.SetBuffer(m_kernelExclusive, "b_scan", _scanBuffer);
            m_cs.SetBuffer(m_kernelExclusive, "b_threadBlockReduction", _threadBlockReductionBuffer);
            m_cs.SetBuffer(m_kernelExclusive, "b_index", _indexBuffer);
        }

        private void SetRootParameters(
            CommandBuffer _cmd,
            ComputeBuffer _scanBuffer,
            ComputeBuffer _threadBlockReductionBuffer,
            ComputeBuffer _indexBuffer)
        {
            _cmd.SetComputeBufferParam(m_cs, m_kernelInit, "b_threadBlockReduction", _threadBlockReductionBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelInit, "b_index", _indexBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelInclusive, "b_scan", _scanBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelInclusive, "b_threadBlockReduction", _threadBlockReductionBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelInclusive, "b_index", _indexBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelExclusive, "b_scan", _scanBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelExclusive, "b_threadBlockReduction", _threadBlockReductionBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelExclusive, "b_index", _indexBuffer);
        }

        public void PrefixSumInclusive(
            int size,
            ComputeBuffer toPrefixSum,
            ComputeBuffer tempBuffer0,
            ComputeBuffer tempBuffer1)
        {
            Assert.IsTrue(
                m_isValid &&
                IsAligned(toPrefixSum.stride) &&
                size < m_allocatedSize &&
                size > k_minSize);

            int threadBlocks = DivRoundUp(AlignFour(size), k_partitionSize);
            SetRootParameters(
                toPrefixSum,
                tempBuffer0,
                tempBuffer1);

            m_cs.SetInt("e_vectorizedSize", VectorizedSize(size));
            m_cs.SetInt("e_threadBlocks", threadBlocks);
            m_cs.Dispatch(m_kernelInit, 256, 1, 1);

            int fullBlocks = threadBlocks / k_maxDispatch;
            if (fullBlocks != 0)
                m_cs.Dispatch(m_kernelInclusive, k_maxDispatch, fullBlocks, 1);

            int partialBlocks = threadBlocks - fullBlocks * k_maxDispatch;
            if (partialBlocks != 0)
                m_cs.Dispatch(m_kernelInclusive, partialBlocks, 1, 1);
        }

        public void PrefixSumExclusive(
            int size,
            ComputeBuffer toPrefixSum,
            ComputeBuffer tempBuffer0,
            ComputeBuffer tempBuffer1)
        {
            Assert.IsTrue(
                m_isValid &&
                IsAligned(toPrefixSum.stride) &&
                size < m_allocatedSize &&
                size > k_minSize);

            int threadBlocks = DivRoundUp(AlignFour(size), k_partitionSize);
            SetRootParameters(
                toPrefixSum,
                tempBuffer0,
                tempBuffer1);

            m_cs.SetInt("e_vectorizedSize", VectorizedSize(size));
            m_cs.SetInt("e_threadBlocks", threadBlocks);
            m_cs.Dispatch(m_kernelInit, 256, 1, 1);

            int fullBlocks = threadBlocks / k_maxDispatch;
            if (fullBlocks != 0)
                m_cs.Dispatch(m_kernelExclusive, k_maxDispatch, fullBlocks, 1);

            int partialBlocks = threadBlocks - fullBlocks * k_maxDispatch;
            if (partialBlocks != 0)
                m_cs.Dispatch(m_kernelExclusive, partialBlocks, 1, 1);
        }

        public void PrefixSumInclusive(
            CommandBuffer cmd,
            int size,
            ComputeBuffer toPrefixSum,
            ComputeBuffer tempBuffer0,
            ComputeBuffer tempBuffer1)
        {
            Assert.IsTrue(
                m_isValid &&
                IsAligned(toPrefixSum.stride) &&
                size < m_allocatedSize &&
                size > k_minSize);

            int vectorizedSize = VectorizedSize(size);
            int threadBlocks = DivRoundUp(AlignFour(size), k_partitionSize);
            SetRootParameters(
                cmd,
                toPrefixSum,
                tempBuffer0,
                tempBuffer1);

            cmd.SetComputeIntParam(m_cs, "e_vectorizedSize", vectorizedSize);
            cmd.SetComputeIntParam(m_cs, "e_threadBlocks", threadBlocks);
            cmd.DispatchCompute(m_cs, m_kernelInit, 256, 1, 1);

            int fullBlocks = threadBlocks / k_maxDispatch;
            if (fullBlocks != 0)
                cmd.DispatchCompute(m_cs, m_kernelInclusive, k_maxDispatch, fullBlocks, 1);

            int partialBlocks = threadBlocks - fullBlocks * k_maxDispatch;
            if (partialBlocks != 0)
                cmd.DispatchCompute(m_cs, m_kernelInclusive, partialBlocks, 1, 1);
        }

        public void PrefixSumExclusive(
            CommandBuffer cmd,
            int size,
            ComputeBuffer toPrefixSum,
            ComputeBuffer tempBuffer0,
            ComputeBuffer tempBuffer1)
        {
            Assert.IsTrue(
                m_isValid &&
                IsAligned(toPrefixSum.stride) &&
                size < m_allocatedSize &&
                size > k_minSize);

            int vectorizedSize = VectorizedSize(size);
            int threadBlocks = DivRoundUp(AlignFour(size), k_partitionSize);
            SetRootParameters(
                cmd,
                toPrefixSum,
                tempBuffer0,
                tempBuffer1);

            cmd.SetComputeIntParam(m_cs, "e_vectorizedSize", vectorizedSize);
            cmd.SetComputeIntParam(m_cs, "e_threadBlocks", threadBlocks);
            cmd.DispatchCompute(m_cs, m_kernelInit, 256, 1, 1);

            int fullBlocks = threadBlocks / k_maxDispatch;
            if (fullBlocks != 0)
                cmd.DispatchCompute(m_cs, m_kernelExclusive, k_maxDispatch, fullBlocks, 1);

            int partialBlocks = threadBlocks - fullBlocks * k_maxDispatch;
            if (partialBlocks != 0)
                cmd.DispatchCompute(m_cs, m_kernelExclusive, partialBlocks, 1, 1);
        }

        ~ChainedScanDecoupledLookbackDecoupledFallback()
        {

        }
    }
}