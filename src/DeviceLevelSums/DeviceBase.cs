using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

public abstract class DeviceBase : ScanBase
{
    protected enum TestType
    {
        //Checks to see if the prefix sum is valid.
        //If validationText is enabled, EVERY index that does not match the correct sum will be printed.
        //However, this will be extremely slow for 32k + errors, so text not advised for large sums.
        ValidateSum,

        //Instead of using a monotonic sequence like we did with the block level scans, we create a sequence of 
        //randomized numbers guarunteed to not exceed 2^30. Because the validation is perfomed CPU side, we generate
        //the numbers on the cpu, then send them to the GPU. The only issue with this test is that its not very useful for
        //debugging besides indicating that a problem is there.
        ValidateSumRandom,

        //Test a specific buffer size using the "SpecificSize" field. 
        DebugAtSize,

        //Prints only the state flags buffer
        DebugState,

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

    protected const int minSize = 21;
    protected const int maxSize = 28;

    protected const int k_init = 0;
    protected int k_scan;

    protected int repeats = 0;

    protected ComputeBuffer stateBuffer;
    protected uint[] stateValidationArray;
    protected uint[] stateInitializationArray;

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
        UpdateStateBuffer(_size);
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

    public virtual void UpdateStateBuffer(int _size)
    {
        if (advancedTimingMode)
            stateInitializationArray = new uint[Mathf.CeilToInt(_size * 1.0f / partitionSize) * scanRepeats + 1];
        else
            stateInitializationArray = new uint[Mathf.CeilToInt(_size * 1.0f / partitionSize) + 1];

        for (uint i = 0; i < stateInitializationArray.Length; ++i)
            stateInitializationArray[i] = 0;

        if (stateBuffer != null)
            stateBuffer.Dispose();
        stateBuffer = new ComputeBuffer(stateInitializationArray.Length, sizeof(uint));
        compute.SetBuffer(k_scan, "b_state", stateBuffer);
    }

    public override void ResetBuffers()
    {
        stateBuffer.SetData(stateInitializationArray);
        compute.Dispatch(k_init, threadBlocks, 1, 1);
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
            case TestType.ValidateSumRandom:
                ValidateSumRandom();
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
            case TestType.DebugState:
                DebugState();
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

    public virtual void ResetBuffersRandom(ref uint[] temp)
    {
        stateBuffer.SetData(stateInitializationArray);
        prefixSumBuffer.SetData(temp);
    }

    public virtual void ValidateSumRandom()
    {
        bool validated = true;
        System.Random random = new System.Random();
        validationArray = new uint[1 << 21];
        uint[] temp = new uint[1 << 21];
        for (uint i = 0; i < temp.Length; ++i)
            temp[i] = (uint)random.Next(0, 512);

        UpdateSize(1 << 21);
        ResetBuffersRandom(ref temp);
        for (int j = 0; j < kernelIterations; ++j)
        {
            DispatchKernels();
            prefixSumBuffer.GetData(validationArray);
            int errCount = 0;
            uint total = 0;
            for (int i = 0; i < (1 << 21); ++i)
            {
                total += temp[i];
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
                ResetBuffersRandom(ref temp);
            else
                break;
        }

        if (validated)
            Debug.Log("Prefix Sum Random passed");
        else
            Debug.LogError("Prefix Sum Random failed");
        UpdateSize(size);
    }


    public virtual void DebugState()
    {
        stateValidationArray = new uint[stateBuffer.count];
        DispatchKernels();
        stateBuffer.GetData(stateValidationArray);

        Debug.Log("---------------STATE VALUES---------------");
        int i = 0;
        for (; i < stateValidationArray.Length - 1; ++i)
            Debug.Log(i + ": " + (stateValidationArray[i] >> 2));
        Debug.Log(i + ": " + stateValidationArray[i]);
    }

    public override void DebugAtSize(int _size)
    {
        validationArray = new uint[_size];
        UpdateSize(_size);
        ResetBuffers();
        DebugState();
        prefixSumBuffer.GetData(validationArray);
        if (validateText)
            ValWithText();
        else
            Val();
        UpdateSize(size);
    }

    public virtual void DebugAll()
    {
        DebugState();
        validationArray = new uint[prefixSumBuffer.count];
        prefixSumBuffer.GetData(validationArray);

        Debug.Log("---------------PREFIX VALUES---------------");
        for (int i = 0; i < validationArray.Length; ++i)
            Debug.Log(i + ": " + validationArray[i]);
    }

    public virtual void OnDestroy()
    {
        if(prefixSumBuffer != null)
            prefixSumBuffer.Dispose();

        if(stateBuffer != null)
            stateBuffer.Dispose();
    }
}
