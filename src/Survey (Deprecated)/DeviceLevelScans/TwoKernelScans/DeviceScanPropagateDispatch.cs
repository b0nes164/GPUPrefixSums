using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class DeviceScanPropagateDispatch : TwoKernelBase
{
    DeviceScanPropagateDispatch()
    {
        threadBlocks = 256;
        mainKernelString = "DeviceScan";
        mainKernelStringB = "DevicePropagate";
        testKernelString = "DeviceScanTiming";
        testKernelStringB = "DevicePropagateTiming";
        computeShaderString = "DeviceScanPropagate";
    }

    public override void DispatchKernels()
    {
        compute.Dispatch(k_scan, threadBlocks, 1, 1);
        compute.Dispatch(k_scanB, threadBlocks - 1, 1, 1);
    }
}
