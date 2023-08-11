using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

public class MemCopyDispatch : MonoBehaviour
{
    private enum TestType
    {
        //Performs a simple test at loopRepeats number of repititions in the kernel
        TimingTest,

        //Performs testIterations numbers of kernel executions, using loopRepeats number of repititions in the kernel. It then prints the results to a csv file.
        RecordTimingData,
    }

    [SerializeField]
    public ComputeShader compute;

    [SerializeField]
    private TestType testType;

    [Range(10, 28)]
    public int sizeExponent;

    [Range(1, 40)]
    public int loopRepeats;

    [Range(1, 1000)]
    public int testIterations;

    private const int k_memCpy = 0;
    private const int THREAD_BLOCKS = 512;

    private ComputeBuffer bufferA;
    private ComputeBuffer bufferB;
    private ComputeBuffer timingBuffer;

    private bool breaker;
    private int reps;

    void Start()
    {
        breaker = true;
        reps = loopRepeats;
        compute.SetInt("e_size", 1 << sizeExponent);
        compute.SetInt("e_repeats", loopRepeats);

        bufferA = new ComputeBuffer(1 << (sizeExponent - 2), sizeof(uint) << 2);
        bufferB = new ComputeBuffer(1 << (sizeExponent - 2), sizeof(uint) << 2);
        timingBuffer = new ComputeBuffer(THREAD_BLOCKS, sizeof(uint));

        compute.SetBuffer(k_memCpy, "bufferA", bufferA);
        compute.SetBuffer(k_memCpy, "bufferB", bufferB);
        compute.SetBuffer(k_memCpy, "timingBuffer", timingBuffer);
        
        Debug.Log("Init Complete");
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
                    compute.SetInt("e_repeats", loopRepeats);
                }

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
        switch (testType)
        {
            case TestType.TimingTest:
                StartCoroutine(TimingRoutine());
                break;
            case TestType.RecordTimingData:
                StartCoroutine(RecordTimingData());
                break;
            default:
                Debug.LogWarning("Test type not found");
                break;
        }
    }


    private void DispatchKernels()
    {
        compute.Dispatch(k_memCpy, THREAD_BLOCKS, 1, 1);
    }

    private IEnumerator TimingRoutine()
    {
        breaker = false;
        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(timingBuffer);
        yield return new WaitUntil(() => request.done);

        float time = Time.realtimeSinceStartup;
        DispatchKernels();
        request = AsyncGPUReadback.Request(timingBuffer);
        yield return new WaitUntil(() => request.done);
        time = Time.realtimeSinceStartup - time;

        Debug.Log("Raw Time: " + time);
        Debug.Log("Speed: " + ((1 << sizeExponent) / time * loopRepeats) + " keys/s");
        breaker = true;
    }

    private IEnumerator RecordTimingData()
    {
        breaker = false;
        List<string> csv = new List<string>();

        for (int loopRepeats = 1; loopRepeats <= 40; ++loopRepeats)
        {
            compute.SetInt("e_repeats", loopRepeats);

            for (int i = 0; i < testIterations; ++i)
            {
                float time = Time.realtimeSinceStartup;
                DispatchKernels();
                AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(timingBuffer);
                yield return new WaitUntil(() => request.done);
                time = Time.realtimeSinceStartup - time;
                csv.Add(loopRepeats + ", " + time);

                if (i % 10 == 0)
                    Debug.Log("Running");
            }
        }

        StreamWriter sWriter = new StreamWriter("MemCpy.csv");
        sWriter.WriteLine("Loop Repititions, Total Time");
        foreach (string s in csv)
            sWriter.WriteLine(s);
        sWriter.Close();

        Debug.Log("Done");
        breaker = true;
    }

    private void OnDestroy()
    {
        if(bufferA != null)
            bufferA.Dispose();
        if (bufferB != null)
            bufferB.Dispose();
        if (timingBuffer != null)
            timingBuffer.Dispose();
    }
}
