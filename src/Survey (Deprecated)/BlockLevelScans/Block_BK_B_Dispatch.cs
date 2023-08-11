using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Block_BK_B_Dispatch : BlockLevelBase
{
    Block_BK_B_Dispatch()
    {
        threadBlocks = 1;
        mainKernelString = "Block_BK_B";
        testKernelString = "Block_BK_B_Timing";
        computeShaderString = "Block_BK_B";
    }
}
