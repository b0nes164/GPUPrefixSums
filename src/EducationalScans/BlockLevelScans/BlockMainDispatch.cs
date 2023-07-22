using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockMainDispatch : BlockLevelBase
{
    BlockMainDispatch()
    {
        threadBlocks = 1;
        mainKernelString = "BlockMain";
        testKernelString = "BlockMainTiming";
        computeShaderString = "BlockMainDispatch";
    }
}
