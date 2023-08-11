using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Block_BK_C_Dispatch : BlockLevelBase
{
    Block_BK_C_Dispatch()
    {
        threadBlocks = 1;
        mainKernelString = "Block_BK_C";
        testKernelString = "Block_BK_C_Timing";
        computeShaderString = "Block_BK_C";
    }
}
