using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

public class ChainedScanWithDecoupledLookbackTiming : MonoBehaviour
{
    private enum TestType
    {
        //Checks to see if the prefix sum is valid.
        ValidatePrefixSum,

        //Validates prefix sum on an input of random numbers instead of the default of input of all elements initialized to one.
        ValidatePrefixSumRandom,

        //Prints 
        DebugPrefixSum,

        //Prints the values of the state and index buffer. Use to ensure that the lookback is functioning properly.
        DebugState,

        //Times execution of the kernel at loopRepeats number of repititions in the kernel
        TimingTest,

        //Performs testIterations numbers of kernel executions, using loopRepeats number of repititions in the kernel. It then prints the results to a csv file.
        RecordTimingData,
    }

    [SerializeField]
    private ComputeShader compute;

    [SerializeField]
    private TestType testType;

    [Range(1, 30)]
    public int loopRepeats;

    [Range(1, 1000)]
    public int testIterations;

    [SerializeField]
    private bool printValidationText;

    [SerializeField]
    private bool quickText;

    private const int k_init = 0;
    private int k_scan;

    private int size;
    private int reps;
    private bool breaker;

    private ComputeBuffer prefixSumBuffer;
    private ComputeBuffer stateBuffer;
    private ComputeBuffer timingBuffer;
    private ComputeBuffer indexBuffer;

    private int partitionSize;
    private int threadBlocks;
    private string computeShaderString;

    private uint[] validationArray;

    ChainedScanWithDecoupledLookbackTiming()
    {
        partitionSize = 8192;
        threadBlocks = 2048;
        computeShaderString = "ChainedDecoupledScanTiming";
    }

    private void Start()
    {
        CheckShader();
        size = 1 << 28;
        reps = loopRepeats;
        UpdateSize(size);
        UpdateRepeats(reps);
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
                if (reps != loopRepeats)
                {
                    reps = loopRepeats;
                    UpdateRepeats(reps);
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
            case TestType.RecordTimingData:
                StartCoroutine(RecordTimingData());
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

            Debug.LogWarning("This is the timing version of the prefix sum. The input size has been locked to 2^28, and some testing functionality has been disabled.");
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
        UpdateStateBuffer(_size, reps);
    }

    private void UpdateRepeats(int _repeats)
    {
        compute.SetInt("e_repeats", _repeats);
        UpdateStateBuffer(size, _repeats);
    }

    private void UpdatePrefixBuffer(int _size)
    {
        if (prefixSumBuffer != null)
            prefixSumBuffer.Dispose();

        prefixSumBuffer = new ComputeBuffer(Mathf.CeilToInt(_size / 4.0f), sizeof(uint) * 4);
        compute.SetBuffer(k_init, "b_prefixLoad", prefixSumBuffer);
        compute.SetBuffer(k_scan, "b_prefixSum", prefixSumBuffer);
    }

    private void UpdateStateBuffer(int _size, int _repeats)
    {
        if (stateBuffer != null)
            stateBuffer.Dispose();
        stateBuffer = new ComputeBuffer(_size / partitionSize * _repeats + 1, sizeof(uint));
        compute.SetBuffer(k_init, "b_state", stateBuffer);
        compute.SetBuffer(k_scan, "b_state", stateBuffer);
    }

    private void UpdateTimingBuffer()
    {
        timingBuffer = new ComputeBuffer(1, sizeof(uint));
        compute.SetBuffer(k_init, "b_timing", timingBuffer);
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
        Debug.Log("Estimated Speed: " + (size / time * reps) + " keys/sec");

        breaker = true;
    }

    private IEnumerator RecordTimingData()
    {
        breaker = false;
        List<string> csv = new List<string>();

        for (int loopRepeats = 1; loopRepeats <= 40; ++loopRepeats)
        {
            UpdateRepeats(loopRepeats);

            for (int i = 0; i < 1000; ++i)
            {
                ResetBuffers();
                AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(timingBuffer);
                yield return new WaitUntil(() => request.done);

                float time = Time.realtimeSinceStartup;
                DispatchKernels();
                request = AsyncGPUReadback.Request(timingBuffer);
                yield return new WaitUntil(() => request.done);
                time = Time.realtimeSinceStartup - time;
                csv.Add(loopRepeats + ", " + time);

                if (i % 10 == 0)
                    Debug.Log("Running");
            }
        }

        StreamWriter sWriter = new StreamWriter("CSDL.csv");
        sWriter.WriteLine("Loop Repititions, Total Time");
        foreach (string s in csv)
            sWriter.WriteLine(s);
        sWriter.Close();

        UpdateRepeats(reps);
        Debug.Log("Done");
        breaker = true;
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