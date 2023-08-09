using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CD_A_Dispatch : DeviceBase
{
    CD_A_Dispatch()
    {
        threadBlocks = 256;
        mainKernelString = "CD_A";
        testKernelString = "CD_A_Timing";
        computeShaderString = "CD_A";
    }

    //Fixed partition count
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
    }
}
