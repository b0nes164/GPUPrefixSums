using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockReduceScanDispatch : BlockLevelBase
{
    BlockReduceScanDispatch()
    {
        threadBlocks = 1;
        mainKernelString = "BlockReduceScan";
        testKernelString = "BlockReduceScanTiming";
        computeShaderString = "BlockReduceScan";
    } 
}
