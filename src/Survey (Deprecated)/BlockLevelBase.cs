using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class BlockLevelBase : ScanBase
{
    protected enum TestType
    {
        //Checks to see if the prefix sum is valid.
        //If validationText is enabled, EVERY index that does not match the correct sum will be printed.
        //However, this will be extremely slow for 32k + errors, so text not advised for large sums.
        ValidateSum,

        //For all other validations, the initial buffer is filled with the value 1.
        //This is because the maximum aggregate value of the prefix sum is 2^30, and
        //thus larger values would quickly limit the maximum buffer size testable.
        //This test allows you to verify that the sum is valid for a monotonically increasing sequence, albeit at a size of 2^15.
        ValidateSumMonotonic,

        //Test a specific buffer size using the "SpecificSize" field. 
        DebugAtSize,

        //Runs the test for the desired number of iterations, without timing or validating,
        TortureTest,

        //Runs the sum for the desired number of iterations, and prints out the speed of the sum.
        //Note, this is *NOT* the true time of the algorithm because it includes the readback time.
        //This is purely for relative measurement of different algorithms. Read the testing methodology for more information.
        TimingTest,

        //Records the raw timing data in a csv file.
        RecordTimingData,

        //Validates for every power of 2 in the given range.
        ValidatePowersOfTwo,

        //Validates the sum on non-powers of two
        ValidateAllOffSizes,
    }

    [SerializeField]
    protected TestType testType;

    [Range(minSize, maxSize)]
    public int sizeExponent;

    [Range(1, 40)]
    public int scanRepeats;

    [SerializeField]
    protected bool advancedTimingMode;

    protected const int k_init = 0;
    protected int k_scan;

    protected const int minSize = 10;
    protected const int maxSize = 28;

    protected int repeats = 0;

    protected int partitionSize;
    protected int threadBlocks;
    protected string testKernelString;
    protected string mainKernelString;
    protected string computeShaderString;

    public virtual void Start()
    {
        CheckShader();

        size = 1 << sizeExponent;
        repeats = scanRepeats;
        UpdateSize(size);
        UpdateRepeats(repeats);

        breaker = true;

        Debug.Log(computeShaderString + ": init Complete.");
    }

    public virtual void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (breaker)
            {
                if (size != (1 << sizeExponent))
                {
                    size = 1 << sizeExponent;
                    UpdateSize(size);
                }

                if (repeats != scanRepeats)
                {
                    repeats = scanRepeats;
                    UpdateRepeats(repeats);
                }
                Dispatcher();
            }
            else
            {
                Debug.LogWarning("Please allow the current test to complete before attempting any other tests.");
            }
        }
    }

    public override void UpdateSize(int _size)
    {
        compute.SetInt("e_size", _size);
        UpdatePrefixBuffer(_size);
    }

    public virtual void UpdateRepeats(int _repeats)
    {
        compute.SetInt("e_repeats", _repeats);
    }

    public override void UpdatePrefixBuffer(int _size)
    {
        if (prefixSumBuffer != null)
            prefixSumBuffer.Dispose();
        prefixSumBuffer = new ComputeBuffer(_size, sizeof(uint));
        compute.SetBuffer(k_init, "b_prefixSum", prefixSumBuffer);
        compute.SetBuffer(k_scan, "b_prefixSum", prefixSumBuffer);
    }

    public override void ResetBuffers()
    {
        compute.Dispatch(k_init, 256, 1, 1);
    }

    public override void CheckShader()
    {
        try
        {
            if (advancedTimingMode)
            {
                Debug.LogWarning("Warning you have advanced timing mode selected. This enables the looping version of the scan.");
                k_scan = compute.FindKernel(testKernelString);
            }
            else
                k_scan = compute.FindKernel(mainKernelString);
        }
        catch
        {
            Debug.LogError("Kernel(s) not found, most likely you do not have the correct compute shader attached to the game object");
            Debug.LogError("The correct compute shader is" + computeShaderString + ". Exit play mode and attatch to the gameobject, then retry.");
            Debug.LogError("Destroying this object.");
            Destroy(this);
        }
    }

    public override void DispatchKernels()
    {
        compute.Dispatch(k_scan, threadBlocks, 1, 1);
    }

    public virtual void Dispatcher()
    {
        ResetBuffers();

        switch (testType)
        {
            case TestType.ValidateSum:
                StartCoroutine(ValidateSum());
                break;
            case TestType.ValidateSumMonotonic:
                ValidateSumMonotonic();
                break;
            case TestType.DebugAtSize:
                DebugAtSize(specificSize);
                break;
            case TestType.TortureTest:
                TortureTest();
                break;
            case TestType.TimingTest:
                StartCoroutine(TimingRoutine());
                break;
            case TestType.ValidatePowersOfTwo:
                StartCoroutine(ValidatePowersOfTwo(minSize, maxSize, computeShaderString, false));
                break;
            case TestType.ValidateAllOffSizes:
                StartCoroutine(ValidateAllOffSizes(computeShaderString));
                break;
            case TestType.RecordTimingData:
                StartCoroutine(RecordTimingData(computeShaderString, "Scan Repeats", "" + scanRepeats));
                break;
            default:
                break;
        }
    }

    public virtual void ResetBuffersMonotonic(ref uint[] temp)
    {
        prefixSumBuffer.SetData(temp);
    }

    public virtual void ValidateSumMonotonic()
    {
        bool validated = true;
        validationArray = new uint[1 << 15];
        uint[] temp = new uint[1 << 15];
        for (uint i = 0; i < temp.Length; ++i)
            temp[i] = i;
        UpdateSize(1 << 15);
        ResetBuffersMonotonic(ref temp);
        for (int j = 0; j < kernelIterations; ++j)
        {
            DispatchKernels();
            prefixSumBuffer.GetData(validationArray);
            int errCount = 0;
            uint total = 0;
            for (uint i = 1; i < (1 << 15); ++i)
            {
                total += i;
                if (validationArray[i] != total)
                {
                    validated = false;
                    if (validateText)
                    {
                        Debug.LogError("EXPECTED THE SAME AT INDEX " + i + ": " + total + ", " + validationArray[i]);
                        if (quickText)
                        {
                            errCount++;
                            if (errCount > 1024)
                                break;
                        }
                    }
                }
            }
            if (validated)
                ResetBuffersMonotonic(ref temp);
            else
                break;
        }

        if (validated)
            Debug.Log("Prefix Sum Monotonic passed");
        else
            Debug.LogError("Prefix Sum Monotonic failed");
        UpdateSize(size);
    }

    public virtual void OnDestroy()
    {
        if (prefixSumBuffer != null)
            prefixSumBuffer.Dispose();
    }

}
