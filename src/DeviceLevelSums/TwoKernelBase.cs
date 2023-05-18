using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//For Reduce Scan and Scan Propagate
public abstract class TwoKernelBase : DeviceBase
{
    //for the second kernel
    protected int k_scanB;
    protected string testKernelStringB;
    protected string mainKernelStringB;

    public override void UpdateStateBuffer(int _size)
    {
        if (advancedTimingMode)
            stateInitializationArray = new uint[threadBlocks * scanRepeats + 1];
        else
            stateInitializationArray = new uint[threadBlocks + 1];

        for (uint i = 0; i < stateInitializationArray.Length; ++i)
            stateInitializationArray[i] = 0;

        if (stateBuffer != null)
            stateBuffer.Dispose();
        stateBuffer = new ComputeBuffer(stateInitializationArray.Length, sizeof(uint));
        compute.SetBuffer(k_scan, "b_state", stateBuffer);
        compute.SetBuffer(k_scanB, "b_state", stateBuffer);
    }

    public override void UpdatePrefixBuffer(int _size)
    {
        base.UpdatePrefixBuffer(_size);
        compute.SetBuffer(k_scanB, "b_prefixSum", prefixSumBuffer);
    }

    public override void CheckShader()
    {
        try
        {
            if (advancedTimingMode)
            {
                Debug.LogWarning("Warning you have advanced timing mode selected. This should only be enabled if you wish to do extensive timing testing.");
                k_scan = compute.FindKernel(testKernelString);
                k_scanB = compute.FindKernel(testKernelStringB);
            }
            else
            {
                k_scan = compute.FindKernel(mainKernelString);
                k_scanB = compute.FindKernel(mainKernelStringB);
            }
        }
        catch
        {
            Debug.LogError("Kernel(s) not found, most likely you do not have the correct compute shader attached to the game object");
            Debug.LogError("The correct compute shader is" + computeShaderString + ". Exit play mode and attatch to the gameobject, then retry.");
            Debug.LogError("Destroying this object.");
            Destroy(this);
        }
    }

    public override void DispatchKernels()
    {
        base.DispatchKernels();
        compute.Dispatch(k_scanB, threadBlocks, 1, 1);
    }

    public override void DebugState()
    {
        stateValidationArray = new uint[stateBuffer.count];
        DispatchKernels();
        stateBuffer.GetData(stateValidationArray);
        Debug.Log("---------------STATE VALUES---------------");
        for (int i = 0; i < stateValidationArray.Length; ++i)
            Debug.Log(i + ": " + stateValidationArray[i]);

    }
}
