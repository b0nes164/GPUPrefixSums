using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockRakingReduceDispatch : BlockLevelBase
{
    BlockRakingReduceDispatch()
    {
        threadBlocks = 1;
        mainKernelString = "BlockRakingReduce";
        testKernelString = "BlockRakingReduceTiming";
        computeShaderString = "BlockRakingReduce";
    }
}
