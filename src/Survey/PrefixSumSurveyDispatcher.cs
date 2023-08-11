using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class PrefixSumSurveyDispatcher : MonoBehaviour
{
    protected enum TestType
    {
        //Checks to see if the prefix sum is valid.
        //If validationText is enabled, EVERY index that does not match the correct sum will be printed.
        //However, this will be extremely slow for 32k + errors, so text not advised for large sums.
        ValidatePrefixSum,

        //For all other validations, the initial buffer is filled with the value 1.
        //This is because the maximum aggregate value of the prefix sum is 2^30, and
        //thus larger values would quickly limit the maximum buffer size testable.
        //This test allows you to verify that the sum is valid for a monotonically increasing sequence, albeit at a size of 2^15.
        ValidateSumMonotonic,

        //Prints out the entire contents of the prefix sum buffer. Not suggested at large sizes due to lag.
        Debug,

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

    [SerializeField]
    private ComputeShader compute;

    [SerializeField]
    private ScanType scanType;

    [SerializeField]
    private TestType testType;

    [Range(minSlider, maxSlider)]
    public int sizeExponent;

    [SerializeField]
    private bool printValidationText;

    [SerializeField]
    private bool quickText;


    private int[] maxSizes = new int[20] 
        { 10, 5, 5, 10, 10, 10, 10, 10, 10, 10,
          10, 10, 20, 25, 25, 18, 25, 25, 25, 25 };

    private const int minSlider = 5;
    private const int maxSlider = 25;

    private const string computeShaderString = "PrefixSumSurvey";
    private const int k_init = 0;

    private int k_scan;
    private int size;
    private bool breaker;
    private int validationCount;
    private int greatestKernelIndex;
    private uint[] validationArray;

    private ComputeBuffer prefixSumBuffer;

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

    public void UpdateSize(int _size)
    {
        compute.SetInt("e_size", _size);
        UpdatePrefixBuffer(_size);
    }

    public void UpdatePrefixBuffer(int _size)
    {
        if (prefixSumBuffer != null)
            prefixSumBuffer.Dispose();
        prefixSumBuffer = new ComputeBuffer(_size, sizeof(uint));
        for (int i = 0; i <= greatestKernelIndex; ++i)
            compute.SetBuffer(i, "prefixSumBuffer", prefixSumBuffer);
    }

    public void ResetBuffers()
    {
        compute.Dispatch(k_init, 256, 1, 1);
    }

    public void DispatchKernels()
    {
        compute.Dispatch(k_scan, 1, 1, 1);
    }

    public void CheckShader()
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
                case TestType.ValidatePrefixSum:
                    StartCoroutine(ValidatePrefixSum(scanType));
                    break;
                case TestType.ValidatePowersOfTwo:
                    StartCoroutine(ValidatePowersOfTwo(minSlider, maxSizes[(int)scanType], scanType.ToString(), true));
                    break;
                case TestType.ValidateSumMonotonic:
                    StartCoroutine(ValidateSumMonotonic(maxSizes[(int)scanType], scanType.ToString()));
                    break;
                case TestType.Debug:
                    DebugText();
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

    private IEnumerator ValidatePrefixSum(ScanType scan)
    {
        breaker = false;

        validationArray = new uint[Mathf.CeilToInt(size / 4.0f) * 4];
        DispatchKernels();
        prefixSumBuffer.GetData(validationArray);
        yield return new WaitForSeconds(.1f);   //To prevent unity from crashig
        if (printValidationText ? ValWithText(size) : Validate(size))
            Debug.Log(scan.ToString() + " Passed");
        else
            Debug.LogError("Sum Failed at size: " + size);

        breaker = true;
    }

    private IEnumerator ValidatePowersOfTwo(int _min, int _max, string kernelString, bool beginningText)
    {
        breaker = false;
        if(beginningText)
            Debug.Log("BEGINNING VALIDATE POWERS OF TWO.");

        validationCount = 0;
        for (int s = minSlider; s <= _max; ++s)
            yield return TestAtSize(1 << s, kernelString);

        if (validationCount == _max + 1 - _min)
            Debug.Log(kernelString + " [" + validationCount + "/" + (_max + 1 - _min) + "] ALL TESTS PASSED");
        else
            Debug.Log(kernelString + " FAILED. [" + validationCount + "/" + (_max + 1 - _min) + "] PASSED");

        UpdateSize(size);
        breaker = true;
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
        prefixSumBuffer.SetData(temp);
        yield return new WaitForSeconds(.1f);  //To prevent unity from crashing

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
                if (printValidationText)
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

    private IEnumerator TestAtSize(int _size, string kernelString)
    {
        UpdateSize(_size);
        ResetBuffers();
        DispatchKernels();
        yield return new WaitForSeconds(.1f);  //To prevent unity from crashing
        validationArray = new uint[Mathf.CeilToInt(_size / 4.0f) * 4];
        prefixSumBuffer.GetData(validationArray);
        if (printValidationText ? ValWithText(_size) : Validate(_size))
            validationCount++;
        else
            Debug.LogError(kernelString + " FAILED AT SIZE: " + _size);
    }

    private bool Validate(int _size)
    {
        for (uint i = 0; i < _size; ++i)
        {
            if (validationArray[i] != (i + 1))
                return false;
        }
        return true;
    }

    private bool ValWithText(int _size)
    {
        bool isValidated = true;
        int errCount = 0;
        for (uint i = 0; i < _size; ++i)
        {
            if (validationArray[i] != (i + 1))
            {
                isValidated = false;
                if (printValidationText)
                {
                    Debug.LogError("EXPECTED THE SAME AT INDEX " + i + ": " + (i + 1) + ", " + validationArray[i]);
                    if (quickText)
                    {
                        errCount++;
                        if (errCount > 1024)
                            break;
                    }
                }
            }
        }

        return isValidated;
    }
    private void OnDestroy()
    {
        prefixSumBuffer.Dispose();
    }
}
