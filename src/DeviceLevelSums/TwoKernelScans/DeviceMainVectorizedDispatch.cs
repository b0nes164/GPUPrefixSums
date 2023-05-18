using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeviceMainVectorizedDispatch : TwoKernelBase
{
    DeviceMainVectorizedDispatch()
    {
        threadBlocks = 256;
        mainKernelString = "DeviceMainVectorizedReduce";
        mainKernelStringB = "DeviceMainVectorizedScan";
        testKernelString = "DeviceMainVectorizedReduceTiming";
        testKernelStringB = "DeviceMainVectorizedScanTiming";
        computeShaderString = "DeviceMainVectorized";
    }

    public override void UpdatePrefixBuffer(int _size)
    {
        if (prefixSumBuffer != null)
            prefixSumBuffer.Dispose();
        prefixSumBuffer = new ComputeBuffer(Mathf.CeilToInt(_size / 4.0f), sizeof(uint) * 4);
        compute.SetBuffer(k_init, "b_prefixLoad", prefixSumBuffer);
        compute.SetBuffer(k_scan, "b_prefixSum", prefixSumBuffer);
        compute.SetBuffer(k_scanB, "b_prefixSum", prefixSumBuffer);
    }

    public override void TestAtSize(int _size, ref int count, string kernelString)
    {
        validationArray = new uint[Mathf.CeilToInt(_size / 4.0f) * 4];
        UpdateSize(_size);
        ResetBuffers();
        DispatchKernels();
        prefixSumBuffer.GetData(validationArray);
        if (ValAndBreak(_size))
            count++;
        else
            Debug.LogError(kernelString + " FAILED AT SIZE: " + _size);
    }

    public override void DebugAtSize(int _size)
    {
        validationArray = new uint[Mathf.CeilToInt(_size / 4.0f) * 4];
        UpdateSize(_size);
        ResetBuffers();
        DebugState();
        prefixSumBuffer.GetData(validationArray);
        if (validateText)
            ValWithText(_size);
        else
            Val(_size);
        UpdateSize(size);
    }

    public virtual bool ValAndBreak(int _size)
    {
        for (uint i = 0; i < _size; ++i)
        {
            if (validationArray[i] != (i + 1))
                return false;
        }
        return true;
    }

    public virtual bool Val(int _size)
    {
        for (uint i = 0; i < _size; ++i)
        {
            if (validationArray[i] != (i + 1))
            {
                Debug.LogError("Sum Failed");
                return false;
            }
        }
        Debug.Log("Sum Passed");
        return true;
    }

    public virtual bool ValWithText(int _size)
    {
        bool isValidated = true;
        int errCount = 0;
        for (uint i = 0; i < _size; ++i)
        {
            if (validationArray[i] != (i + 1))
            {
                isValidated = false;
                if (validateText)
                {
                    Debug.LogError("EXPECTED THE SAME AT INDEX " + i + ": " + (i + 1) + ", " + validationArray[i]);
                    if (quickText)
                    {
                        errCount++;
                        if (errCount > 1024)
                            break;
                    }
                }
            }
        }

        if (isValidated)
            Debug.Log("Sum Passed");
        else
            Debug.LogError("Sum Failed");
        return isValidated;
    }
}
