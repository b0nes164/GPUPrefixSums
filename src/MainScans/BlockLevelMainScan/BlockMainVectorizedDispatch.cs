using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockMainVectorizedDispatch : BlockLevelBase
{
    BlockMainVectorizedDispatch()
    {
        threadBlocks = 1;
        mainKernelString = "BlockMainVectorized";
        testKernelString = "BlockMainVectorizedTiming";
        computeShaderString = "BlockMainVectorized";
    }

    public override void UpdatePrefixBuffer(int _size)
    {
        if (prefixSumBuffer != null)
            prefixSumBuffer.Dispose();
        prefixSumBuffer = new ComputeBuffer(Mathf.CeilToInt(_size / 4.0f), sizeof(uint) * 4);
        compute.SetBuffer(k_init, "b_prefixLoad", prefixSumBuffer);
        compute.SetBuffer(k_scan, "b_prefixSum", prefixSumBuffer);
    }

    public override void TestAtSize(int _size, ref int count, string kernelString)
    {
        validationArray = new uint[Mathf.CeilToInt(_size / 4.0f) * 4];
        UpdateSize(_size);
        ResetBuffers();
        DispatchKernels();
        prefixSumBuffer.GetData(validationArray);
        if (ValVector(_size))
            count++;
        else
            Debug.LogError(kernelString + " FAILED AT SIZE: " + _size);
    }

    protected bool ValVector(int _size)
    {
        for (uint i = 0; i < _size; ++i)
        {
            if (validationArray[i] != (i + 1))
                return false;
        }
        return true;
    }
}
