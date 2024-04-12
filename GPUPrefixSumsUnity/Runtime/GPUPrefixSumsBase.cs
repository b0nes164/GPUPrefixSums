/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/10/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
using UnityEngine;

namespace GPUPrefixSums.Runtime
{
    public class GPUPrefixSumsBase
    {
        protected const int k_partitionSize = 3072;

        protected const int k_maxDispatch = 65535;
        protected const int k_isNotPartialBitFlag = 0;
        protected const int k_isPartialBitFlag = 1;

        protected const int k_minSize = 1;
        protected const int k_maxSize = (1 << 30) - 128;

        protected ComputeShader m_cs;

        protected int m_allocatedSize;

        protected static int DivRoundUp(int x, int y)
        {
            return (x + y - 1) / y;
        }

        protected static int VectorizedSize(int x)
        {
            return (x + 3) / 4;
        }
        protected static int AlignFour(int x)
        {
            return VectorizedSize(x) * 4;
        }

        protected static bool IsAligned(int x)
        {
            return (x & 15) == 0;
        }
    }
}
