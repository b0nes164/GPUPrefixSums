using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CD_C_Dispatch : DeviceBase
{
    CD_C_Dispatch()
    {
        partitionSize = 8192;
        threadBlocks = 256;
        mainKernelString = "CD_C";
        testKernelString = "CD_C_Timing";
        computeShaderString = "CD_C";
    }
}
