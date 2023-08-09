using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class CD_SPEED_Dispatch : DeviceBase
{
    CD_SPEED_Dispatch()
    {
        partitionSize = 8192;
        threadBlocks = 256;
        mainKernelString = "CD_SPEED";
        testKernelString = "CD_SPEED_Timing";
        computeShaderString = "CD_SPEED";
    }
    public override void Start()
    {
        Debug.LogWarning("Warning, this scan is designed specifically for testing the speed at buffer size 2^28.");
        Debug.LogWarning("Changing buffer sizes, and some other functionality is disabled.");

        CheckShader();

        size = 1 << 28;
        UpdateSize(size);

        repeats = scanRepeats;
        UpdateRepeats(repeats);

        breaker = true;

        Debug.Log(computeShaderString + ": init Complete.");
    }

    public override void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (breaker)
            {
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
        UpdateStateBuffer(_size);
        UpdatePrefixBuffer(_size);
    }

    public override void Dispatcher()
    {
        ResetBuffers();

        switch (testType)
        {
            case TestType.ValidateSum:
                StartCoroutine(ValidateSum());
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
            case TestType.RecordTimingData:
                StartCoroutine(RecordTimingData(computeShaderString, "Scan Repeats", "" + scanRepeats));
                break;
            default:
                Debug.LogWarning("This scan does not support this test.");
                break;
        }
    }
}
