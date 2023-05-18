using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CD_Main_Dispatch : DeviceBase
{
    CD_Main_Dispatch()
    {
        partitionSize = 8192;
        threadBlocks = 256;
        mainKernelString = "CD_Main";
        testKernelString = "CD_Main_Timing";
        computeShaderString = "CD_Main";
    }
}