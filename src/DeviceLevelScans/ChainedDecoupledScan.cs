using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class ChainedDecoupledScan : MonoBehaviour
{
    public enum TestType
    {
        //Checks to see if the prefix sum is valid.
        //If validationText is enabled, indexes that do not match the correct sum will be printed.
        //However, this will be extremely slow for 16k + errors, so text not advised for large sums.
        ValidateSum,

        //Prints the entire prefix sum buffer, as well as the state flags buffer.
        //As with validationText, this will be slow for sums of size greater than 16k ish.
        DebugAll,

        //Prints only the state flags buffer
        DebugState,

        //Runs the test for the desired number of iterations, without timing or validating,
        TortureTest,

        //Runs the sum for the desired number of iterations, and prints out the speed of the sum.
        TimingTest,

        //Validates for every possible power of 2 sum size. Note, it only tests up to 2^27 because
        //although the algorithm is performing correctly (which you can validate with individual test)
        //Unity does not like scheduling multiple sequential dispatches on large buffers, and tends to crash
        ValidateAll
    };
    public enum ScanType
    {
        ChainedDecoupledScanA,
        ChainedDecoupledScanB,
        ChainedDecoupledScanC,
        ChainedDecoupledScanAtomic,
    };


    [SerializeField]
    ComputeShader compute;

    [SerializeField]
    public ScanType scanType;

    [SerializeField]
    public TestType testType;

    [Range(10, 28)]
    public int sizeExponent;

    [Range(1, 500)]
    public int testIterations;
    
    [SerializeField]
    private bool validateText;

    private int k_init;
    private int k_initPartDesc;
    private int k_chainedScan;

    private const int DEFAULT_BLOCKS = 32;
    private int threadBlocks;
    private int groupSize;
    private int partitionSize;
    private int partitions;
    private int initPartitionsBlocks;
    private int size; 

    private ComputeBuffer prefixSumBuffer;
    private ComputeBuffer stateBuffer;

    private uint[] validationArray;
    private uint[] stateValidationArray;

    void Start()
    {
        size = 1 << sizeExponent;
        k_init = compute.FindKernel("Init");
        k_initPartDesc = compute.FindKernel("InitPartitionDescriptors");

        switch (scanType)
        {
            case ScanType.ChainedDecoupledScanA:
                FindProperKernel(scanType.ToString(), "CD_A");
                InitA();
                break;
            case ScanType.ChainedDecoupledScanB:
                FindProperKernel(scanType.ToString(), "CD_B");
                InitB();
                break;
            case ScanType.ChainedDecoupledScanC:
                FindProperKernel(scanType.ToString(), "CD_C");
                InitC();
                break;
            case ScanType.ChainedDecoupledScanAtomic:
                FindProperKernel(scanType.ToString(), "CD_Atomic");
                InitAtomic();
                break;
            default:
                Debug.LogError("This should not happen, destroying this object");
                Destroy(this);
                break;
        }
        
        Debug.Log(scanType.ToString() + ": init Complete.");
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Dispatcher();
        }
    }

    void InitA()
    {
        partitionSize = 2048;
        groupSize = 1024;
        threadBlocks = 256;
        partitions = Mathf.CeilToInt(size * 1.0f / partitionSize);
        initPartitionsBlocks = Mathf.CeilToInt((partitions * 3 + 1) * 1.0f / groupSize);
        BoilerPlateInits();
    }

    void InitB()
    {
        partitionSize = 1024;
        groupSize = 1024;       //irrelevant
        threadBlocks = 256;
        partitions = Mathf.CeilToInt(size * 1.0f / partitionSize);
        initPartitionsBlocks = Mathf.CeilToInt((partitions * 3 + 1) * 1.0f / groupSize);
        BoilerPlateInits();
    }

    void InitC()
    {
        partitionSize = 1024;
        groupSize = 1024;
        threadBlocks = Mathf.CeilToInt(size * 1.0f / partitionSize);
        partitions = threadBlocks;
        initPartitionsBlocks = Mathf.CeilToInt((partitions * 3 + 1) * 1.0f / groupSize);
        BoilerPlateInits();
    }

    void InitAtomic()
    {
        partitionSize = 0; //irrelevant in this case
        groupSize = 1024;
        threadBlocks = 256;
        partitions = threadBlocks;         
        initPartitionsBlocks = Mathf.CeilToInt((partitions * 3 + 1) * 1.0f / groupSize);
        BoilerPlateInits();
    }

    void BoilerPlateInits()
    {
        compute.SetInt("e_size", size);
        compute.SetInt("e_partitions", partitions);

        prefixSumBuffer = new ComputeBuffer(size, sizeof(uint));
        stateBuffer = new ComputeBuffer(partitions * 3 + 1, sizeof(uint));

        compute.SetBuffer(k_init, "b_prefixSum", prefixSumBuffer);
        compute.Dispatch(k_init, DEFAULT_BLOCKS, 1, 1);

        compute.SetBuffer(k_initPartDesc, "b_state", stateBuffer);
        compute.Dispatch(k_initPartDesc, initPartitionsBlocks, 1, 1);

        compute.SetBuffer(k_chainedScan, "b_prefixSum", prefixSumBuffer);
        compute.SetBuffer(k_chainedScan, "b_state", stateBuffer);
    }

    void FindProperKernel(string kernelName, string kernelFileName)
    {
        try
        {
            k_chainedScan = compute.FindKernel(kernelName);
        }
        catch
        {
            Debug.LogError(kernelName + " kernel not found, most likely you do not have the correct compute shader attached to the game object");
            Debug.LogError("The correct compute shader is " + kernelFileName + ". Exit play mode and attatch to the gameobject, then retry.");
            Debug.LogError("Destroying this object.");
            Destroy(this);
        }
    }
    void Dispatcher()
    {
        switch (testType)
        {
            case TestType.ValidateSum:
                ValidateSum();
                break;
            case TestType.DebugAll:
                DebugAll();
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
            case TestType.ValidateAll:
                ValidateAll();
                break;
            default:
                break;
        }
    }

    void ValidateSum()
    {
        bool validated = true;
        validationArray = new uint[size];
        for (uint j = 0; j < testIterations; j++)
        {
            compute.Dispatch(k_chainedScan, threadBlocks, 1, 1);
            prefixSumBuffer.GetData(validationArray);
            for (uint i = 0; i < size; i++)
            {
                if (validationArray[i] != (i + 1))
                {
                    validated = false;
                    if (validateText)
                        Debug.Log("EXPECTED THE SAME AT INDEX " + i + ": " + (i + 1) + ", " + validationArray[i]);
                }
            }

            if (!validated)
                break;
            else
            {
                compute.Dispatch(k_init, DEFAULT_BLOCKS, 1, 1);
                compute.Dispatch(k_initPartDesc, initPartitionsBlocks, 1, 1);
            }
        }

        compute.Dispatch(k_init, DEFAULT_BLOCKS, 1, 1);
        compute.Dispatch(k_initPartDesc, initPartitionsBlocks, 1, 1);

        if (validated)
            Debug.Log("Prefix Sum passed");
        else
            Debug.LogError("Prefix Sum failed");
    }

    private void DebugAll()
    {
        compute.Dispatch(k_chainedScan, threadBlocks, 1, 1);
        validationArray = new uint[prefixSumBuffer.count];
        stateValidationArray = new uint[stateBuffer.count];
        prefixSumBuffer.GetData(validationArray);
        stateBuffer.GetData(stateValidationArray);

        for (int i = 0; i < validationArray.Length; ++i)
            Debug.Log(i + ": " + validationArray[i]);

        Debug.Log("---------------STATE VALUES---------------");
        for(int i = 0; i < stateValidationArray.Length; ++i)
            Debug.Log(i + ": " + stateValidationArray[i]);
    }

    private void DebugState()
    {
        stateValidationArray = new uint[stateBuffer.count];
        compute.Dispatch(k_chainedScan, threadBlocks, 1, 1);
        stateBuffer.GetData(stateValidationArray);
        Debug.Log("---------------STATE VALUES---------------");
        for (int i = 0; i < stateValidationArray.Length; ++i)
            Debug.Log(i + ": " + stateValidationArray[i]);

        compute.Dispatch(k_init, DEFAULT_BLOCKS, 1, 1);
        compute.Dispatch(k_initPartDesc, initPartitionsBlocks, 1, 1);
    }

    private void TortureTest()
    {
        for (int i = 0; i < testIterations; ++i)
        {
            compute.Dispatch(k_initPartDesc, initPartitionsBlocks, 1, 1);
            compute.Dispatch(k_chainedScan, threadBlocks, 1, 1);
            Debug.Log("Running " + i);
        }

        validationArray = new uint[prefixSumBuffer.count];
        prefixSumBuffer.GetData(validationArray);
        Debug.Log(validationArray[size - 1]);
    }

    private IEnumerator TimingRoutine()
    {
        float totalTime = 0;
        for (int i = 0; i < testIterations; ++i)
        {
            float time = Time.realtimeSinceStartup;
            compute.Dispatch(k_initPartDesc, initPartitionsBlocks, 1, 1);
            compute.Dispatch(k_chainedScan, threadBlocks, 1, 1);
            AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(prefixSumBuffer);
            yield return new WaitUntil(() => request.done);
            totalTime += Time.realtimeSinceStartup - time;
            compute.Dispatch(k_init, DEFAULT_BLOCKS, 1, 1);

            if (i == size / testIterations)
                Debug.Log("Running");
        }
        Debug.Log("Done");
        Debug.Log("PrefixSum average time " + size * (testIterations / totalTime) + " elements/sec");
    }

    private void ValidateAll()
    {
        Debug.Log("BEGINNING VALIDATE ALL.");

        int validCount = 0;
        int start = 10;
        int max = scanType != ScanType.ChainedDecoupledScanC ? 28 : 25;

        for (int s = start; s <= max; s++)
        {
            size = 1 << s;
            validationArray = new uint[1 << s];

            prefixSumBuffer.Dispose();
            stateBuffer.Dispose();

            switch (scanType)
            {
                case ScanType.ChainedDecoupledScanA:
                    InitA();
                    break;
                case ScanType.ChainedDecoupledScanB:
                    InitB();
                    break;
                case ScanType.ChainedDecoupledScanC:
                    InitC();
                    break;
                case ScanType.ChainedDecoupledScanAtomic:
                    InitAtomic();
                    break;
                default:
                    Debug.LogError("This should not happen, destroying this object");
                    Destroy(this);
                    break;
            }

            compute.Dispatch(k_chainedScan, threadBlocks, 1, 1);
            prefixSumBuffer.GetData(validationArray);

            if (Val(ref validationArray))
                validCount++;
            else
                Debug.LogError(scanType.ToString() + " FAILED AT SIZE: " + (1 << s));
        }

        if (validCount == max + 1 - start)
            Debug.Log(scanType.ToString() + " [" + validCount + "/" + (max + 1 - start) + "] ALL TESTS PASSED");
        else
            Debug.Log(scanType.ToString() + " FAILED. [" + validCount + "/" + (max + 1 - start) + "] PASSED");
    }

    private bool Val(ref uint[] vArray)
    {
        for (int i = 0; i < size; ++i)
            if (vArray[i] != (i + 1))
                return false;
        return true;
    }

    private void OnDestroy()
    {
        prefixSumBuffer.Dispose();
        stateBuffer.Dispose();
    }
}
