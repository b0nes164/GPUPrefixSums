using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CD_B_Dispatch : DeviceBase
{
    CD_B_Dispatch()
    {
        partitionSize = 16384;
        threadBlocks = 256;
        mainKernelString = "CD_B";
        testKernelString = "CD_B_Timing";
        computeShaderString = "CD_B";
    }
}
