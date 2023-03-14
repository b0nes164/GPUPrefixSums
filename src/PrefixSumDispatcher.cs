using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class PrefixSumDispatcher : MonoBehaviour
{
    private enum TestType
    {
        //Checks to see if the prefix sum is valid.
        //If validationText is enabled, indexes that do not match the correct sum will be printed.
        //However, this will be extremely slow for 16k + errors, so its not really advised.
        ValidateSum,

        //Prints the entire array output
        //As with validationText, this will be slow for sums of size greater than 16k ish.
        DebugText,

        //Runs the test for the desired number of iterations, without timing or validating,
        TortureTest,

        //Runs the sum for the desired number of iterations, and prints out the speed of the sum.
        TimingTest,

        //Validates every test, for every possible power of 2 sum size. Note, it only tests up to 2^27 because
        //although the algorithm is performing correctly (which you can validate with individual test)
        //Unity does not like scheduling multiple sequential dispatches on large buffers, and will crash.
        ValidateAll
    };

    private enum ScanType
    {
        serial,
        koggeStoneWarp,
        warpIntrinsic,
        koggeStone,
        sklansky,
        brentKung,
        reduceScan,
        radixBrentKung,
        radixSklansky,
        brentKungLarge,
        brentKungLargeUnrolled,
        reduceScanLarge,
        radixBrentKungLarge,
        radixReduceLarge,
        radixSklasnkyLarge,
    }

    [SerializeField]
    private ComputeShader compute;

    [SerializeField]
    private ScanType scanType;

    [SerializeField]
    private TestType testType;

    [Range(4, 28)]
    public int sizeExponent; //My unity version crashes when I try to allocate a buffer larger than 2^28, so I've set this as the max

    [Range(1, 500)]
    public int testIterations;

    [SerializeField]
    private bool validateText;

    private const int k_init = 0;
    private int size;
    private ComputeBuffer prefixSumBuffer;
    private (int, ScanType) scan;

    private uint[] validationArray;
    private bool breaker = true;

    void Start()
    {
        size = 1 << sizeExponent;
        compute.SetInt("e_size", size);
        prefixSumBuffer = new ComputeBuffer(size, sizeof(uint));
        compute.SetBuffer(k_init, "prefixSumBuffer", prefixSumBuffer);

        Debug.Log("Init Complete. Press space to initiate desired test.");
    }

    void Update()
    {
        if (breaker)
        {
            if (scan.Item2 != scanType)
                scan = ((int)scanType + 1, scanType);

            if (size != (1 << sizeExponent))
            {
                size = 1 << sizeExponent;
                compute.SetInt("e_size", size);
                prefixSumBuffer.Dispose();
                prefixSumBuffer = new ComputeBuffer(size, sizeof(uint));
                compute.SetBuffer(k_init, "prefixSumBuffer", prefixSumBuffer);
            }
        }

        if (Input.GetKeyDown(KeyCode.Space))
            Dispatcher();
    }

    void Dispatcher()
    {
        if (scan.Item1 < 3 && sizeExponent > 5)
            Debug.LogWarning("Selected sum size is too big for the selected scan. Please change the scan to one labeled 'large' or change the sum size.");
        else
        {
            if (scan.Item1 < 10 && sizeExponent > 10)
                Debug.LogWarning("Selected sum size is too big for the selected scan. Please change the scan to one labeled 'large' or change the sum size.");
            else
            {
                if (scan.Item1 == 12 && sizeExponent > 18)
                {
                    Debug.LogWarning("Because we are using shared memory to hold the intermediates and HLSL's constraint on shared memory, the maximum size of Reduce Scan");
                    Debug.LogWarning("is effectively 8192 * (2 ^ spillFactor). By default the spillFactor is 4, yielding a max dispatch size of 131072 or 2^17");
                    Debug.LogWarning("Please change the sum size.");
                }
                else
                {
                    switch (testType)
                    {
                        case TestType.ValidateSum:
                            ValidateSum();
                            break;
                        case TestType.DebugText:
                            DebugText();
                            break;
                        case TestType.TimingTest:
                            StartCoroutine(TimingRoutine(testIterations, size, scan.Item1));
                            break;
                        case TestType.TortureTest:
                            TortureTest();
                            break;
                        case TestType.ValidateAll:
                            ValidateAll();
                            break;
                    }
                }
            }
        }
    }

    void ValidateSum()
    {
        bool validated = true;
        validationArray = new uint[size];
        compute.Dispatch(k_init, 32, 1, 1);
        compute.SetBuffer(scan.Item1, "prefixSumBuffer", prefixSumBuffer);
        Debug.Log("Beginning " + testIterations + " of validation test for scan type " + scan.Item2.ToString() + ". Sum size: " + size + ".");

        for (uint j = 0; j < testIterations; j++)
        {
            compute.Dispatch(scan.Item1, 1, 1, 1);
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
                compute.Dispatch(k_init, 1, 1, 1);
        }

        if (validated)
            Debug.Log("Prefix Sum passed");
        else
            Debug.LogError("Prefix Sum failed");
    }

    private void DebugText()
    {
        compute.Dispatch(k_init, 32, 1, 1);
        Debug.Log("Beginning debug text for scan type " + scan.Item2.ToString() + ". Sum size: " + size + ".");
        compute.SetBuffer(scan.Item1, "prefixSumBuffer", prefixSumBuffer);
        compute.Dispatch(scan.Item1, 1, 1, 1);
        validationArray = new uint[size];
        prefixSumBuffer.GetData(validationArray);
        for (int i = 0; i < validationArray.Length; i++)
            Debug.Log(i + ": " +validationArray[i]);
    }

    private IEnumerator TimingRoutine(int _testIterations, int _size, int kernel)
    {
        if (breaker)
        {
            breaker = false;
            float totalTime = 0;
            compute.SetBuffer(kernel, "prefixSumBuffer", prefixSumBuffer);
            compute.Dispatch(k_init, 32, 1, 1);
            Debug.Log("Beginning " + _testIterations + " of timing test for scan type " + scan.Item2.ToString() + ". Sum size: " + _size + ".");

            for (int i = 0; i < _testIterations; ++i)
            {
                float time = Time.realtimeSinceStartup;
                compute.Dispatch(kernel, 1, 1, 1);
                AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(prefixSumBuffer);
                yield return new WaitUntil(() => request.done);
                totalTime += Time.realtimeSinceStartup - time;
                compute.Dispatch(k_init, 32, 1, 1);
            }
            Debug.Log("Done");
            Debug.Log("PrefixSum average time " + _size * (_testIterations / totalTime) + " elements/sec");
            breaker = true;
        }
        else
            Debug.LogWarning("Please alow the previous timing test to finish before initiating a new one.");
    }

    private void TortureTest()
    {
        Debug.Log("Beginning " + testIterations + " of torture test for scan type " + scan.Item2.ToString() + ". Sum size: " + size + ".");
        compute.SetBuffer(scan.Item1, "prefixSumBuffer", prefixSumBuffer);
        for (int i = 0; i < testIterations; ++i)
        {
            compute.Dispatch(k_init, 32, 1, 1);
            compute.Dispatch(scan.Item1, 1, 1, 1);
        }
        Debug.Log("Complete");
    }

    private void ValidateAll()
    {
        Debug.Log("BEGINNING VALIDATE ALL." );
        int scanCount = System.Enum.GetValues(typeof(ScanType)).Length;

        for (int i = 0; i < scanCount; ++i)
        {
            Debug.Log("------------------" + ((ScanType)i).ToString() + "------------------");

            validationArray = new uint[1 << 10];
            prefixSumBuffer.Dispose();
            prefixSumBuffer = new ComputeBuffer(1 << 10, sizeof(uint));
            compute.SetBuffer(k_init, "prefixSumBuffer", prefixSumBuffer);
            compute.SetBuffer(i + 1, "prefixSumBuffer", prefixSumBuffer);

            int start, max;
            int validCount = 0;
            switch (i < 3 ? 0 : i < 9 ? 1 : i == 11 ? 2 : 3)
            {
                case 0:
                    start = 1;
                    max = 5;
                    break;
                case 1:
                    start = 5;
                    max = 10;
                    break;
                case 2:
                    start = 5;
                    max = 17;
                    break;
                case 3:
                    start = 5;
                    max = 27;
                    break;
                default:
                    start = 0;
                    max = -1;
                    break;
            }

            for (int s = start; s <= max; ++s)
            {
                if (s > 20)
                {
                    validationArray = new uint[1 << s];
                    prefixSumBuffer.Dispose();
                    prefixSumBuffer = new ComputeBuffer(1 << s, sizeof(uint));
                    compute.SetBuffer(k_init, "prefixSumBuffer", prefixSumBuffer);
                    compute.SetBuffer(i + 1, "prefixSumBuffer", prefixSumBuffer);
                }
                else
                {
                    if (s == 11)
                    {
                        validationArray = new uint[1 << 20];
                        prefixSumBuffer.Dispose();
                        prefixSumBuffer = new ComputeBuffer(1 << 20, sizeof(uint));
                        compute.SetBuffer(k_init, "prefixSumBuffer", prefixSumBuffer);
                        compute.SetBuffer(i + 1, "prefixSumBuffer", prefixSumBuffer);
                    }
                }
                
                if (Val(i + 1, ref validationArray, s))
                    validCount++;
                else
                    Debug.LogError(((ScanType)i).ToString() + " FAILED AT SIZE: " + (1 << s));
            }

            if (validCount == max + 1 - start)
                Debug.Log(((ScanType)i).ToString() + " [" + validCount + "/" + (max + 1 - start) + "] ALL TESTS PASSED");
            else
                Debug.Log(((ScanType)i).ToString() + " FAILED. [" + validCount + "/" + (max + 1 - start) + "] PASSED");
        }
    }

    private bool Val(int kernel, ref uint[] vArray, int s)
    {
        compute.SetInt("e_size", 1 << s);
        compute.Dispatch(k_init, 32, 1, 1);
        compute.Dispatch(kernel, 1, 1, 1);
        prefixSumBuffer.GetData(vArray);

        for (int i = 0; i < (1 << s); ++i)
            if (vArray[i] != (i + 1))
                return false;
        return true;
    }

    private void OnDestroy()
    {
        prefixSumBuffer.Dispose();
    }
}
