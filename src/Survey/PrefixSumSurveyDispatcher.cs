using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class PrefixSumDispatcher : ScanBase
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

        //Prints out the entire contents of the prefix sum buffer. Not suggested at large sizes due to lag.
        Debug,

        //Test a specific buffer size using the "SpecificSize" field. 
        DebugAtSize,

        //Runs the sum for the desired number of iterations, and prints out the speed of the sum.
        //Note, this is *NOT* the true time of the algorithm because it includes the readback time.
        //This is purely for relative measurement of different algorithms. Read the testing methodology for more information.
        TimingTest,

        //Records the raw timing data in a csv file.
        RecordTimingData,

        //Validates for every power of 2 in the given range.
        ValidatePowersOfTwo,

        //Runs the validate monontonic test on ALL scans contained in the survey
        ValidateMonotonicAllScans,

        //Runs the validate powers of two test on ALL scans contained in the survey
        ValidatePowersOfTwoAllScans
    }

    private enum ScanType
    {
        Serial,
        KoggeStoneWarp,
        WarpIntrinsic,
        RakingReduce,
        RadixRakingReduce,
        KoggeStone,
        Sklansky,
        BrentKung,
        BrentKungBlelloch,
        ReduceScan, 
        RadixBrentKung,
        RadixSklansky,
        BrentKungLarge,
        BrentKungBlellochLarge,
        BrentKungLargeUnrolled,
        ReduceScanLarge, 
        RadixBrentKungLarge,
        RadixBrentKungFused,
        RadixSklanskyLarge,
        RadixSklanskyAdvanced
    }

    private int[] maxSizes = new int[20] 
        { 10, 5, 5, 10, 10, 10, 10, 10, 10, 10,
          10, 10, 20, 25, 25, 18, 25, 25, 25, 25 };

    [SerializeField]
    private ScanType scanType;

    [SerializeField]
    private TestType testType;

    [Range(minSlider, maxSlider)]
    public int sizeExponent;

    private const int minSlider = 5;
    private const int maxSlider = 25;

    private const string computeShaderString = "PrefixSumSurvey";
    private const int k_init = 0;

    protected int k_scan;
    protected int greatestKernelIndex;

    void Start()
    {
        CheckShader();

        size = 1 << sizeExponent;
        UpdateSize(size);

        k_scan = compute.FindKernel(scanType.ToString());

        breaker = true;

        Debug.Log("Init complete");
    }

    void Update()
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

                k_scan = compute.FindKernel(scanType.ToString());

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

    public override void UpdatePrefixBuffer(int _size)
    {
        if (prefixSumBuffer != null)
            prefixSumBuffer.Dispose();
        prefixSumBuffer = new ComputeBuffer(_size, sizeof(uint));
        for (int i = 0; i <= greatestKernelIndex; ++i)
            compute.SetBuffer(i, "prefixSumBuffer", prefixSumBuffer);
    }

    public override void ResetBuffers()
    {
        compute.Dispatch(k_init, 32, 1, 1);
    }

    public override void DispatchKernels()
    {
        compute.Dispatch(k_scan, 1, 1, 1);
    }

    public override void CheckShader()
    {
        try
        {
            greatestKernelIndex = 0;
            foreach (ScanType scan in Enum.GetValues(typeof(ScanType)))
            {
                compute.FindKernel(scan.ToString());
                greatestKernelIndex++;
            }
        }
        catch
        {
            Debug.LogError("Kernel(s) not found, most likely you do not have the correct compute shader attached to the game object");
            Debug.LogError("The correct compute shader is " + computeShaderString + ". Exit play mode and attatch to the gameobject, then retry.");
            Debug.LogError("Destroying this object.");
            Destroy(this);
        }
    }

    void Dispatcher()
    {
        if (size <= (1 << maxSizes[(int)scanType]))
        {
            ResetBuffers();

            switch (testType)
            {
                case TestType.ValidateSum:
                    StartCoroutine(ValidateSum());
                    break;
                case TestType.ValidateSumMonotonic:
                    StartCoroutine(ValidateSumMonotonic(maxSizes[(int)scanType], scanType.ToString()));
                    break;
                case TestType.Debug:
                    DebugText();
                    break;
                case TestType.TimingTest:
                    StartCoroutine(TimingRoutine());
                    break;
                case TestType.RecordTimingData:
                    StartCoroutine(RecordTimingData(scanType.ToString(), "", ""));
                    break;
                case TestType.ValidatePowersOfTwo:
                    StartCoroutine(ValidatePowersOfTwo(minSlider, maxSizes[(int)scanType], scanType.ToString(), false));
                    break;
                case TestType.ValidateMonotonicAllScans:
                    StartCoroutine(ValidateMonotonicAllScans());
                    break;
                case TestType.ValidatePowersOfTwoAllScans:
                    StartCoroutine(ValidatePowersOfTwoAllScans());
                    break;
                default:
                    break;
            }
        }
        else
        {
            Debug.LogWarning("Size too big");
        }
        
    }

    public virtual void ResetBuffersMonotonic(ref uint[] temp)
    {
        prefixSumBuffer.SetData(temp);
    }

    public virtual IEnumerator ValidateSumMonotonic(int max, string kernelString)
    {
        breaker = false;

        int tempSize = 1 << (max > 15 ? 15 : max);
        bool validated = true;
        validationArray = new uint[tempSize];
        uint[] temp = new uint[tempSize];
        for (uint i = 0; i < temp.Length; ++i)
            temp[i] = i;
        UpdateSize(tempSize);
        ResetBuffersMonotonic(ref temp);
        for (int j = 0; j < kernelIterations; ++j)
        {
            DispatchKernels();
            prefixSumBuffer.GetData(validationArray);
            int errCount = 0;
            uint total = 0;
            for (uint i = 1; i < tempSize; ++i)
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
            yield return new WaitForSeconds(.25f);  //To prevent unity from crashing
        }

        if (validated)
            Debug.Log("Prefix Sum Monotonic " + kernelString + ": passed");
        else
            Debug.LogError("Prefix Sum Monotonic " + kernelString + ": failed");
        UpdateSize(size);

        breaker = true;
    }

    public virtual void DebugText()
    {
        DispatchKernels();
        validationArray = new uint[size];
        prefixSumBuffer.GetData(validationArray);
        for (int i = 0; i < validationArray.Length; ++i)
            Debug.Log(i + ": " + validationArray[i]);
    }

    public virtual IEnumerator ValidateMonotonicAllScans()
    {
        Debug.Log("Begginning monotonic sum validation for all scans: ");
        foreach (ScanType scan in Enum.GetValues(typeof(ScanType)))
        {
            k_scan = compute.FindKernel(scan.ToString());
            yield return ValidateSumMonotonic(maxSizes[(int)scan], scan.ToString());
        }
        k_scan = compute.FindKernel(scanType.ToString());
        Debug.Log("Test complete.");
    }

    public virtual IEnumerator ValidatePowersOfTwoAllScans()
    {
        Debug.Log("Begginning powers of two validation for all scans: ");
        foreach (ScanType scan in Enum.GetValues(typeof(ScanType)))
        {
            k_scan = compute.FindKernel(scan.ToString());
            yield return ValidatePowersOfTwo(minSlider, maxSizes[(int)scan], scan.ToString(), false);
        }
        k_scan = compute.FindKernel(scanType.ToString());
        Debug.Log("Test complete.");
    }
    private void OnDestroy()
    {
        prefixSumBuffer.Dispose();
    }
}
