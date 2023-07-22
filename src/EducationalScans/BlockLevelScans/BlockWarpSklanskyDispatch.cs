using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockWarpSklanskyDispatch : BlockLevelBase
{
    BlockWarpSklanskyDispatch()
    {
        threadBlocks = 1;
        mainKernelString = "BlockWarpSklansky";
        testKernelString = "BlockWarpSklanskyTiming";
        computeShaderString = "BlockWarpSklansky";
    }
}
