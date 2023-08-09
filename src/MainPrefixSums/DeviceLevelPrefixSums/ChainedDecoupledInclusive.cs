using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class ChainedDecoupledInclusive : MonoBehaviour
{
    private enum TestType
    {
        //Checks to see if the prefix sum is valid.
        //If validationText is enabled, EVERY index that does not match the correct sum will be printed.
        //It is recommended to enable quick text which will limit the number errors printed to 1024,
        //because printing many errors can be quite slow.
        ValidatePrefixSum,

        //Validates prefix sum on an input of random numbers instead of the default of input of all elements initialized to one.
        ValidatePrefixSumRandom,

        //Prints 
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

    [SerializeField]
    private bool printValidationText;

    [SerializeField]
    private bool quickText;

    private const int minSize = 8192;
    private const int maxSize = 268435456;

    private const int k_init = 0;
    private int k_scan;

    private int size;
    private int validationCount;
    private bool breaker;

    private ComputeBuffer prefixSumBuffer;
    private ComputeBuffer stateBuffer;
    private ComputeBuffer timingBuffer;
    private ComputeBuffer indexBuffer;

    private int partitionSize;
    private int threadBlocks;
    private string computeShaderString;

    private uint[] validationArray;

    ChainedDecoupledInclusive()
    {
        partitionSize = 8192;
        threadBlocks = 256;
        computeShaderString = "ChainedDecoupledInclusive";
    }

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

        prefixSumBuffer = new ComputeBuffer(Mathf.CeilToInt(_size / 4.0f), sizeof(uint) * 4);
        compute.SetBuffer(k_init, "b_prefixLoad", prefixSumBuffer);
        compute.SetBuffer(k_scan, "b_prefixSum", prefixSumBuffer);
    }

    private void UpdateStateBuffer(int _size)
    {
        if (stateBuffer != null)
            stateBuffer.Dispose();
        stateBuffer = new ComputeBuffer(_size / partitionSize, sizeof(uint));
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
        compute.Dispatch(k_init, threadBlocks, 1, 1);
    }

    private void DispatchKernels()
    {
        compute.Dispatch(k_scan, threadBlocks, 1, 1);
    }

    private IEnumerator ValidatePrefixSum()
    {
        breaker = false;

        validationArray = new uint[Mathf.CeilToInt(size / 4.0f) * 4];
        DispatchKernels();
        prefixSumBuffer.GetData(validationArray);
        yield return new WaitForSeconds(.1f);   //To prevent unity from crashig
        if (printValidationText ? ValWithText(size) : Validate(size))
            Debug.Log("Sum Passed");
        else
            Debug.LogError("Sum Failed at size: " + size);

        breaker = true;
    }

    private IEnumerator ValidateRandom()
    {
        breaker = false;

        //intialize the buffer to random values
        System.Random rand = new System.Random((int)(Time.realtimeSinceStartup * 1000000.0f));
        uint[] temp = new uint[Mathf.CeilToInt(size / 4.0f) * 4];
        int max = (1 << 30) / temp.Length;
        for (uint i = 0; i < temp.Length; ++i)
            temp[i] = (uint)rand.Next(1, max);
        prefixSumBuffer.SetData(temp);

        bool isValidated = true;
        validationArray = new uint[temp.Length];
        DispatchKernels();
        prefixSumBuffer.GetData(validationArray);
        int errCount = 0;
        uint total = 0;

        for (int i = 0; i < size; ++i)
        {
            total += temp[i];

            if (validationArray[i] != total)
            {
                isValidated = false;
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
        yield return new WaitForSeconds(.1f);

        if (isValidated)
            Debug.Log("Prefix Sum Random passed");
        else
            Debug.LogError("Prefix Sum Random failed");
        UpdateSize(size);

        breaker = true;
    }

    private IEnumerator DebugPrefixSum()
    {
        breaker = false;

        validationArray = new uint[Mathf.CeilToInt(size / 4.0f) * 4];
        DispatchKernels();
        prefixSumBuffer.GetData(validationArray);
        yield return new WaitForSeconds(.1f);   //To prevent unity from crashing

        int limit = quickText ? 1024 : size;
        for (int i = 0; i < limit; ++i)
            Debug.Log(validationArray[i]);

        breaker = true;
    }

    private IEnumerator DebugState()
    {
        breaker = false;

        validationArray = new uint[stateBuffer.count];
        uint[] tempArray = new uint[1];

        DispatchKernels();
        stateBuffer.GetData(validationArray);
        indexBuffer.GetData(tempArray);

        Debug.Log("---------------STATE VALUES---------------");
        for (int i = 0; i < validationArray.Length; ++i)
            Debug.Log(i + ": " + (validationArray[i] >> 2));

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
        DispatchKernels();
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

        validationCount = 0;
        for (int s = 21; s <= 28; ++s)
            yield return TestAtSize(1 << s);

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
        Debug.Log("Beginning Validate All Off Sizes. This may take a while.");

        validationCount = 0;
        for (int i = 1; i <= partitionSize; ++i)
        {
            yield return TestAtSize((1 << 16) + i);
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

    private IEnumerator TestAtSize(int _size)
    {
        UpdateSize(_size);
        ResetBuffers();
        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(stateBuffer);
        yield return new WaitUntil(() => request.done);

        DispatchKernels();
        validationArray = new uint[Mathf.CeilToInt(_size / 4.0f) * 4];
        prefixSumBuffer.GetData(validationArray);
        if (printValidationText ? ValWithText(_size) : Validate(_size))
            validationCount++;
        else
            Debug.LogError(computeShaderString + " FAILED AT SIZE: " + _size);
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