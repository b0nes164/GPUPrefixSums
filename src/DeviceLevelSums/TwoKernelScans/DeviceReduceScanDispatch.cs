using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class DeviceReduceScanDispatch : TwoKernelBase
{
    DeviceReduceScanDispatch()
    {
        threadBlocks = 256;
        mainKernelString = "DeviceReduce";
        mainKernelStringB = "DeviceScan";
        testKernelString = "DeviceReduceTiming";
        testKernelStringB = "DeviceScanTiming";
        computeShaderString = "DeviceReduceScan";
    }
}
