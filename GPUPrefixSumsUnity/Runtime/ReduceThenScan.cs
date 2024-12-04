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
    public class ReduceThenScan : GPUPrefixSumsBase
    {
        private readonly int m_kernelReduce = -1;
        private readonly int m_kernelScan = -1;
        private readonly int m_kernelInclusive = -1;
        private readonly int m_kernelExclusive = -1;

        private readonly bool m_isValid;

        public ReduceThenScan(
            ComputeShader compute,
            int maxElements,
            ref ComputeBuffer tempBuffer0)
        {
            m_cs = compute;
            if (m_cs)
            {
                m_kernelReduce = m_cs.FindKernel("Reduce");
                m_kernelScan = m_cs.FindKernel("Scan");
                m_kernelInclusive = m_cs.FindKernel("PropagateInclusive");
                m_kernelExclusive = m_cs.FindKernel("PropagateExclusive");
            }

            m_isValid = m_kernelReduce >= 0 &&
                        m_kernelScan >= 0 &&
                        m_kernelInclusive >= 0 &&
                        m_kernelExclusive >= 0;

            if (m_isValid)
            {
                if (!m_cs.IsSupported(m_kernelReduce) ||
                    !m_cs.IsSupported(m_kernelScan) ||
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
            AllocateResources(m_allocatedSize, ref tempBuffer0);

            LocalKeyword m_vulkanKeyword = new LocalKeyword(m_cs, "VULKAN");
            if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Vulkan)
                m_cs.EnableKeyword(m_vulkanKeyword);
            else
                m_cs.DisableKeyword(m_vulkanKeyword);
        }

        private void AllocateResources(
            int allocationSize,
            ref ComputeBuffer threadBlockReductionBuffer)
        {
            threadBlockReductionBuffer?.Dispose();
            threadBlockReductionBuffer = new ComputeBuffer(DivRoundUp(allocationSize, k_partitionSize), sizeof(uint));
        }

        private void SetRootParameters(
            ComputeBuffer _scanInBuffer,
            ComputeBuffer _scanOutBuffer,
            ComputeBuffer _threadBlockReductionBuffer)
        {
            m_cs.SetBuffer(m_kernelReduce, "b_scanIn", _scanInBuffer);
            m_cs.SetBuffer(m_kernelReduce, "b_threadBlockReduction", _threadBlockReductionBuffer);

            m_cs.SetBuffer(m_kernelScan, "b_threadBlockReduction", _threadBlockReductionBuffer);

            m_cs.SetBuffer(m_kernelInclusive, "b_scanIn", _scanInBuffer);
            m_cs.SetBuffer(m_kernelInclusive, "b_scanOut", _scanOutBuffer);
            m_cs.SetBuffer(m_kernelInclusive, "b_threadBlockReduction", _threadBlockReductionBuffer);

            m_cs.SetBuffer(m_kernelExclusive, "b_scanIn", _scanInBuffer);
            m_cs.SetBuffer(m_kernelExclusive, "b_scanOut", _scanOutBuffer);
            m_cs.SetBuffer(m_kernelExclusive, "b_threadBlockReduction", _threadBlockReductionBuffer);
        }

        private void SetRootParameters(
            CommandBuffer _cmd,
            ComputeBuffer _scanInBuffer,
            ComputeBuffer _scanOutBuffer,
            ComputeBuffer _threadBlockReductionBuffer)
        {
            _cmd.SetComputeBufferParam(m_cs, m_kernelReduce, "b_scanIn", _scanInBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelReduce, "b_threadBlockReduction", _threadBlockReductionBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelScan, "b_threadBlockReduction", _threadBlockReductionBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelInclusive, "b_scanIn", _scanInBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelInclusive, "b_scanOut", _scanOutBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelInclusive, "b_threadBlockReduction", _threadBlockReductionBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelExclusive, "b_scanIn", _scanInBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelExclusive, "b_scanOut", _scanOutBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelExclusive, "b_threadBlockReduction", _threadBlockReductionBuffer);
        }

        public void SplitDispatch(
            int kernel,
            int fullBlocks,
            int partialBlocks)
        {
            if (fullBlocks != 0)
            {
                m_cs.SetInt("e_isPartial", k_isNotPartialBitFlag);
                m_cs.Dispatch(kernel, k_maxDispatch, fullBlocks, 1);
            }

            if (partialBlocks != 0)
            {
                m_cs.SetInt("e_isPartial", k_isPartialBitFlag);
                m_cs.SetInt("e_fullDispatches", fullBlocks);
                m_cs.Dispatch(kernel, partialBlocks, 1, 1);
            }
        }

        public void SplitDispatch(
            CommandBuffer cmd,
            int kernel,
            int fullBlocks,
            int partialBlocks)
        {
            if (fullBlocks != 0)
            {
                cmd.SetComputeIntParam(m_cs, "e_isPartial", k_isNotPartialBitFlag);
                cmd.DispatchCompute(m_cs, kernel, k_maxDispatch, fullBlocks, 1);
            }

            if (partialBlocks != 0)
            {
                cmd.SetComputeIntParam(m_cs, "e_isPartial", k_isPartialBitFlag);
                cmd.SetComputeIntParam(m_cs, "e_fullDispatches", fullBlocks);
                cmd.DispatchCompute(m_cs, kernel, partialBlocks, 1, 1);
            }
        }

        public void PrefixSumInclusive(
            int size,
            ComputeBuffer prefixSumIn,
            ComputeBuffer prefixSumOut,
            ComputeBuffer tempBuffer0)
        {
            Assert.IsTrue(
                m_isValid &&
                IsAligned(prefixSumIn.stride) &&
                IsAligned(prefixSumOut.stride) &&
                size < m_allocatedSize &&
                size > k_minSize);

            int threadBlocks = DivRoundUp(AlignFour(size), k_partitionSize);
            SetRootParameters(
                prefixSumIn,
                prefixSumOut,
                tempBuffer0);

            m_cs.SetInt("e_vectorizedSize", VectorizedSize(size));
            m_cs.SetInt("e_threadBlocks", threadBlocks);

            int fullBlocks = threadBlocks / k_maxDispatch;
            int partialBlocks = threadBlocks - fullBlocks * k_maxDispatch;

            SplitDispatch(m_kernelReduce, fullBlocks, partialBlocks);
            m_cs.Dispatch(m_kernelScan, 1, 1, 1);
            SplitDispatch(m_kernelInclusive, fullBlocks, partialBlocks);
        }

        public void PrefixSumExclusive(
            int size,
            ComputeBuffer prefixSumIn,
            ComputeBuffer prefixSumOut,
            ComputeBuffer tempBuffer0)
        {
            Assert.IsTrue(
                m_isValid &&
                IsAligned(prefixSumIn.stride) &&
                IsAligned(prefixSumOut.stride) &&
                size < m_allocatedSize &&
                size > k_minSize);

            int threadBlocks = DivRoundUp(AlignFour(size), k_partitionSize);
            SetRootParameters(
                prefixSumIn,
                prefixSumOut,
                tempBuffer0);

            m_cs.SetInt("e_vectorizedSize", VectorizedSize(size));
            m_cs.SetInt("e_threadBlocks", threadBlocks);

            int fullBlocks = threadBlocks / k_maxDispatch;
            int partialBlocks = threadBlocks - fullBlocks * k_maxDispatch;

            SplitDispatch(m_kernelReduce, fullBlocks, partialBlocks);
            m_cs.Dispatch(m_kernelScan, 1, 1, 1);
            SplitDispatch(m_kernelExclusive, fullBlocks, partialBlocks);
        }

        public void PrefixSumInclusive(
            CommandBuffer cmd,
            int size,
            ComputeBuffer prefixSumIn,
            ComputeBuffer prefixSumOut,
            ComputeBuffer tempBuffer0)
        {
            Assert.IsTrue(
                m_isValid &&
                IsAligned(prefixSumIn.stride) &&
                IsAligned(prefixSumOut.stride) &&
                size < m_allocatedSize &&
                size > k_minSize);

            int vectorizedSize = VectorizedSize(size);
            int threadBlocks = DivRoundUp(AlignFour(size), k_partitionSize);
            SetRootParameters(
                cmd,
                prefixSumIn,
                prefixSumOut,
                tempBuffer0);

            cmd.SetComputeIntParam(m_cs, "e_vectorizedSize", VectorizedSize(size));
            cmd.SetComputeIntParam(m_cs, "e_threadBlocks", threadBlocks);

            int fullBlocks = threadBlocks / k_maxDispatch;
            int partialBlocks = threadBlocks - fullBlocks * k_maxDispatch;
            SplitDispatch(cmd, m_kernelReduce, fullBlocks, partialBlocks);
            cmd.DispatchCompute(m_cs, m_kernelScan, 1, 1, 1);
            SplitDispatch(cmd, m_kernelInclusive, fullBlocks, partialBlocks);
        }

        public void PrefixSumExclusive(
            CommandBuffer cmd,
            int size,
            ComputeBuffer prefixSumIn,
            ComputeBuffer prefixSumOut,
            ComputeBuffer tempBuffer0)
        {
            Assert.IsTrue(
                m_isValid &&
                IsAligned(prefixSumIn.stride) &&
                IsAligned(prefixSumOut.stride) &&
                size < m_allocatedSize &&
                size > k_minSize);

            int vectorizedSize = VectorizedSize(size);
            int threadBlocks = DivRoundUp(AlignFour(size), k_partitionSize);
            SetRootParameters(
                cmd,
                prefixSumIn,
                prefixSumOut,
                tempBuffer0);

            cmd.SetComputeIntParam(m_cs, "e_vectorizedSize", VectorizedSize(size));
            cmd.SetComputeIntParam(m_cs, "e_threadBlocks", threadBlocks);

            int fullBlocks = threadBlocks / k_maxDispatch;
            int partialBlocks = threadBlocks - fullBlocks * k_maxDispatch;
            SplitDispatch(cmd, m_kernelReduce, fullBlocks, partialBlocks);
            cmd.DispatchCompute(m_cs, m_kernelScan, 1, 1, 1);
            SplitDispatch(cmd, m_kernelExclusive, fullBlocks, partialBlocks);
        }
    }
}