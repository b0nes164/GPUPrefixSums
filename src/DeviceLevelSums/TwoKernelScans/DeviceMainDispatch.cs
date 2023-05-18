using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeviceMainDispatch : TwoKernelBase
{
    DeviceMainDispatch()
    {
        threadBlocks = 256;
        mainKernelString = "DeviceMainReduce";
        mainKernelStringB = "DeviceMainScan";
        testKernelString = "DeviceMainReduceTiming";
        testKernelStringB = "DeviceMainScanTiming";
        computeShaderString = "DeviceMain";
    }
}
