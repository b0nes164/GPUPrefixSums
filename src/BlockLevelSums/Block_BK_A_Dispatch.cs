using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Block_BK_A_Dispatch : BlockLevelBase
{
    Block_BK_A_Dispatch()
    {
        threadBlocks = 1;
        mainKernelString = "Block_BK_A";
        testKernelString = "Block_BK_A_Timing";
        computeShaderString = "Block_BK_A";
    }
}
