using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CD_D_Dispatch : DeviceBase
{
    CD_D_Dispatch()
    {
        partitionSize = 256;
        threadBlocks = 1024;
        mainKernelString = "CD_D";
        testKernelString = "CD_D_Timing";
        computeShaderString = "CD_D";
    }
}
