using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class ChainedScanWithDecoupledLookbackExclusive : MonoBehaviour
{
    private enum TestType
    {
        //Checks to see if the prefix sum is valid.
        ValidatePrefixSum,

        //Validates prefix sum on an input of random numbers instead of the default of input of all elements initialized to one.
        //Tends to be slower because validation is performed on the CPU instead of GPU
        ValidatePrefixSumRandom,

        //Prints out the prefix sum, gets slow very fast for large inputs
        DebugPrefixSum,

        //Prints the values of the state and index buffer. Use to ensure that the lookback is functioning properly.
        DebugState,

        //Times the execution of the prefix sum kernel. Read testing methodology for more information.
        TimingTest,

        //Validates the prefix sum on input of size for every power of two from 21 to 28.
        ValidatePowersOfTwo,

        //Validates the prefix sum for non-powers of two.
        ValidateAllOffSizes,
    }

    [SerializeField]
    private ComputeShader compute;

    [SerializeField]
    private TestType testType;

    [Range(minSize, maxSize)]
    public int inputSize;

    private const int minSize = 8191;
    private const int maxSize = 268435456;

    private const int k_init = 0;
    private int k_scan;
    private const int k_validate = 2;

    private int size;
    private bool breaker;

    private ComputeBuffer prefixSumBuffer;
    private ComputeBuffer stateBuffer;
    private ComputeBuffer timingBuffer;
    private ComputeBuffer indexBuffer;

    private const int partitionSize = 8192;
    private const string computeShaderString = "ChainedDecoupledExclusive";

    private void Start()
    {
        CheckShader();
        size = inputSize;
        UpdateSize(size);
        UpdateTimingBuffer();
        UpdateIndexBuffer();
        breaker = true;
        Debug.Log(computeShaderString + ": init complete.");
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (breaker)
            {
                if (size != inputSize)
                {
                    size = inputSize;
                    UpdateSize(size);
                }

                ResetBuffers();

                Dispatcher();
            }
            else
            {
                Debug.LogWarning("Please allow the current test to complete before attempting any other tests.");
            }
        }
    }

    private void Dispatcher()
    {
        ResetBuffers();

        switch (testType)
        {
            case TestType.ValidatePrefixSum:
                StartCoroutine(ValidatePrefixSum());
                break;
            case TestType.ValidatePrefixSumRandom:
                StartCoroutine(ValidateRandom());
                break;
            case TestType.DebugPrefixSum:
                StartCoroutine(DebugPrefixSum());
                break;
            case TestType.DebugState:
                StartCoroutine(DebugState());
                break;
            case TestType.TimingTest:
                StartCoroutine(TimingTest());
                break;
            case TestType.ValidatePowersOfTwo:
                StartCoroutine(ValidatePowersOfTwo());
                break;
            case TestType.ValidateAllOffSizes:
                StartCoroutine(ValidateAllOffsizes());
                break;
            default:
                Debug.LogWarning("Test type not found");
                break;
        }
    }

    private void CheckShader()
    {
        //scan kernel functions as identifier for compute shader
        try
        {
            k_scan = compute.FindKernel(computeShaderString);
        }
        catch
        {
            Debug.LogError("Kernel(s) not found, most likely you do not have the correct compute shader attached to the game object");
            Debug.LogError("The correct compute shader is" + computeShaderString + ". Exit play mode and attatch to the gameobject, then retry.");
            Debug.LogError("Destroying this object.");
            Destroy(this);
        }
    }

    private void UpdateSize(int _size)
    {
        compute.SetInt("e_size", _size);
        UpdatePrefixBuffer(_size);
        UpdateStateBuffer(_size);
    }

    private void UpdatePrefixBuffer(int _size)
    {
        if (prefixSumBuffer != null)
            prefixSumBuffer.Dispose();

        prefixSumBuffer = new ComputeBuffer(divRoundUp(_size, 4), sizeof(uint) * 4);
        compute.SetBuffer(k_init, "b_prefixLoad", prefixSumBuffer);
        compute.SetBuffer(k_scan, "b_prefixSum", prefixSumBuffer);
        compute.SetBuffer(k_validate, "b_prefixSum", prefixSumBuffer);
    }

    private void UpdateStateBuffer(int _size)
    {
        if (stateBuffer != null)
            stateBuffer.Dispose();
        stateBuffer = new ComputeBuffer(divRoundUp(_size, partitionSize), sizeof(uint));
        compute.SetBuffer(k_init, "b_state", stateBuffer);
        compute.SetBuffer(k_scan, "b_state", stateBuffer);
    }

    private void UpdateTimingBuffer()
    {
        timingBuffer = new ComputeBuffer(1, sizeof(uint));
        compute.SetBuffer(k_scan, "b_timing", timingBuffer);
    }

    private void UpdateIndexBuffer()
    {
        indexBuffer = new ComputeBuffer(1, sizeof(uint));
        compute.SetBuffer(k_init, "b_index", indexBuffer);
        compute.SetBuffer(k_scan, "b_index", indexBuffer);
    }

    private void ResetBuffers()
    {
        compute.Dispatch(k_init, 256, 1, 1);
    }

    private void DispatchKernels(int _size)
    {
        compute.Dispatch(k_scan, divRoundUp(_size, partitionSize), 1, 1);
    }

    private IEnumerator ValidatePrefixSum()
    {
        breaker = false;

        if (TestAtSizeIndirect(size))
            Debug.Log("Test passed at size " + size);
        else
            Debug.LogError("Test failed at size " + size);
        yield return new WaitForSeconds(.1f);   //To prevent unity from crashing when reading back data GPU -> CPU

        breaker = true;
    }

    private IEnumerator ValidateRandom()
    {
        breaker = false;

        //intialize the buffer to random values
        System.Random rand = new System.Random((int)(Time.realtimeSinceStartup * 1000000.0f));
        uint[] temp = new uint[prefixSumBuffer.count * 4];
        for (int i = 0; i < temp.Length; ++i)
            temp[i] = (uint)rand.Next(1, 2);
        prefixSumBuffer.SetData(temp);

        bool isValidated = true;
        uint[] validationArray = new uint[temp.Length];
        DispatchKernels(size);
        prefixSumBuffer.GetData(validationArray);
        int errCount = 0;
        uint total = 0;

        for (int i = 0; i < size; ++i)
        {
            if (validationArray[i] != total)
            {
                isValidated = false;
                Debug.LogError("EXPECTED THE SAME AT INDEX " + i + ": " + total + ", " + validationArray[i]);

                errCount++;
                if (errCount > 1024)
                    break;
            }

            total += temp[i];
        }
        yield return new WaitForSeconds(.1f);

        if (isValidated)
            Debug.Log("Prefix Sum Random passed");
        else
            Debug.LogError("Prefix Sum Random failed");

        breaker = true;
    }

    private IEnumerator DebugPrefixSum()
    {
        breaker = false;

        uint[] outputArr = new uint[prefixSumBuffer.count * 4];

        DispatchKernels(size);
        prefixSumBuffer.GetData(outputArr);

        Debug.Log("---------------PREFIX VALUES---------------");
        for (int i = 0; i < size; ++i)
            Debug.Log(i + ": " + outputArr[i]);

        yield return new WaitForSeconds(.1f);

        breaker = true;
    }

    private IEnumerator DebugState()
    {
        breaker = false;

        uint[] outputArray = new uint[stateBuffer.count];
        uint[] tempArray = new uint[1];

        DispatchKernels(size);
        stateBuffer.GetData(outputArray);
        indexBuffer.GetData(tempArray);

        Debug.Log("---------------STATE VALUES---------------");
        for (int i = 0; i < outputArray.Length; ++i)
            Debug.Log(i + ": " + (outputArray[i] >> 2));

        Debug.Log("---------------INDEX VALUE----------------");
        Debug.Log(tempArray[0]);

        yield return new WaitForSeconds(.1f);   //To prevent unity from crashing

        breaker = true;
    }

    private IEnumerator TimingTest()
    {
        breaker = false;

        //make sure the init kernel is done
        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(stateBuffer);
        yield return new WaitUntil(() => request.done);

        float time = Time.realtimeSinceStartup;
        DispatchKernels(size);
        request = AsyncGPUReadback.Request(timingBuffer);
        yield return new WaitUntil(() => request.done);
        time = Time.realtimeSinceStartup - time;

        Debug.Log("Raw Time: " + time + " secs");
        Debug.Log("Estimated Speed: " + (size / time) + " keys/sec");

        breaker = true;
    }

    public virtual IEnumerator ValidatePowersOfTwo()
    {
        breaker = false;

        Debug.Log("BEGINNING VALIDATE POWERS OF TWO.");

        int validationCount = 0;
        for (int s = 21; s <= 28; ++s)
            yield return validationCount += TestAtSizeIndirect(1 << s) ? 1 : 0;

        if (validationCount == 8)
            Debug.Log(computeShaderString + " [" + validationCount + "/ 8]. ALL TESTS PASSED");
        else
            Debug.Log(computeShaderString + " FAILED. [" + validationCount + "/ 8] PASSED");

        UpdateSize(size);

        breaker = true;
    }

    public virtual IEnumerator ValidateAllOffsizes()
    {
        breaker = false;

        Debug.Log("Beginning Validate All Off Sizes.");

        int validationCount = 0;
        for (int i = 1; i <= partitionSize; ++i)
        {
            yield return validationCount += TestAtSizeIndirect((1 << 16) + i) ? 1 : 0;
            if ((i & 31) == 0)
                Debug.Log("Running");
        }

        if (validationCount == partitionSize)
            Debug.Log("[" + validationCount + "/" + partitionSize + "]. ALL TESTS PASSED");
        else
            Debug.LogError("[" + validationCount + "/" + partitionSize + "] TESTS PASSED");

        UpdateSize(size);

        breaker = true;
    }

    private bool TestAtSizeIndirect(int _size)
    {
        if (size != _size)
            UpdateSize(_size);
        ResetBuffers();
        DispatchKernels(_size);
        return ValidateIndirect(_size);
    }

    private bool ValidateIndirect(int _size)
    {
        bool isValid = true;
        uint[] t = new uint[1] { 0 };
        ComputeBuffer validate = new ComputeBuffer(1, sizeof(uint));
        validate.SetData(t);

        ComputeBuffer errorAppend = new ComputeBuffer(1024, sizeof(uint) * 3, ComputeBufferType.Append);
        errorAppend.SetCounterValue(0);

        compute.SetBuffer(k_validate, "b_validate", validate);
        compute.SetBuffer(k_validate, "b_error", errorAppend);
        compute.Dispatch(k_validate, divRoundUp(_size, 8192), 1, 1);

        validate.GetData(t);

        if (t[0] > 0)
        {
            Vector3Int[] err = new Vector3Int[1024];
            errorAppend.GetData(err);

            t[0] = t[0] < 1024 ? t[0] : 1024;
            for (int i = 0; i < t[0]; ++i)
                Debug.LogError("EXPECTED THE SAME AT INDEX " + err[i].x + ": " + err[i].y + ", " + err[i].z);

            isValid = false;
        }

        validate.Dispose();
        errorAppend.Dispose();

        return isValid;
    }

    private int divRoundUp(int dividend, int divisor)
    {
        return (dividend + divisor - 1) / divisor;
    }

    private void OnDestroy()
    {
        if (prefixSumBuffer != null)
            prefixSumBuffer.Dispose();
        if (stateBuffer != null)
            stateBuffer.Dispose();
        if (timingBuffer != null)
            timingBuffer.Dispose();
        if (indexBuffer != null)
            indexBuffer.Dispose();
    }
}