using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

public abstract class ScanBase : MonoBehaviour
{
    [SerializeField]
    protected ComputeShader compute;

    [Range(1, 500)]
    public int kernelIterations;

    [SerializeField]
    protected int specificSize;

    [SerializeField]
    protected bool validateText;

    [SerializeField]
    protected bool quickText;

    protected int size;
    protected bool breaker;
    protected ComputeBuffer prefixSumBuffer;

    protected uint[] validationArray;

    public abstract void UpdateSize(int _size);

    public abstract void UpdatePrefixBuffer(int _size);

    public abstract void ResetBuffers();

    public abstract void DispatchKernels();

    public abstract void CheckShader();

    public virtual IEnumerator ValidateSum()
    {
        breaker = false;
        validationArray = new uint[size];
        for (int j = 0; j < kernelIterations; ++j)
        {
            DispatchKernels();
            prefixSumBuffer.GetData(validationArray);
            if (ValWithText())
                ResetBuffers();
            else
                break;

            yield return new WaitForSeconds(.25f);  //To prevent unity from crashing
            if (j % 10 == 1)
                Debug.Log("Running");
        }
        breaker = true;
    }

    public virtual void DebugAtSize(int _size)
    {
        validationArray = new uint[_size];
        UpdateSize(_size);
        ResetBuffers();
        DispatchKernels();
        prefixSumBuffer.GetData(validationArray);
        if (validateText)
            ValWithText();
        else
            Val();
        UpdateSize(size);
    }

    public virtual void TortureTest()
    {
        for (int i = 0; i < kernelIterations; ++i)
        {
            DispatchKernels();
            ResetBuffers();
            if (i % 10 == 0)
                Debug.Log("Running");
        }
        Debug.Log("Torture complete. Lol.");
    }

    public virtual IEnumerator TimingRoutine()
    {
        breaker = false;
        float totalTime = 0;
        Debug.LogWarning("Please note that this is the time with the readback delay included. This is *NOT* the actual speed of the algorithm.");
        Debug.LogWarning("Rather, this should be used for relative comparisons between algorithms.");
        for (int i = 0; i < kernelIterations; ++i)
        {
            float time = Time.realtimeSinceStartup;
            DispatchKernels();
            AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(prefixSumBuffer);
            yield return new WaitUntil(() => request.done);
            totalTime += Time.realtimeSinceStartup - time;
            ResetBuffers();
            yield return new WaitForSeconds(.5f);  //To prevent unity from crashing
            if (i % 10 == 0)
                Debug.Log("Running");
        }

        Debug.Log("Raw Value: " + totalTime);
        Debug.Log("Round trip average time: " + kernelIterations * (size / totalTime) + " elements/sec");
        breaker = true;
    }

    public virtual IEnumerator RecordTimingData(string kernelString, string collumnHeader, string collumnLabel)
    {
        breaker = false;
        List<string> csv = new List<string>();
        for (int i = 0; i < kernelIterations; ++i)
        {
            float time = Time.realtimeSinceStartup;
            DispatchKernels();
            AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(prefixSumBuffer);
            yield return new WaitUntil(() => request.done);
            time = Time.realtimeSinceStartup - time;
            csv.Add(collumnLabel + ", " + time);
            ResetBuffers();
            
            yield return new WaitForSeconds(.5f);  //To prevent unity from crashing
            if (i % 10 == 0)
                Debug.Log("Running");
        }

        StreamWriter sWriter = new StreamWriter(kernelString + ".csv");
        sWriter.WriteLine(collumnHeader + ", Total Time");
        foreach (string s in csv)
            sWriter.WriteLine(s);
        sWriter.Close();

        Debug.Log("Done");
        breaker = true;
    }

    public virtual IEnumerator ValidatePowersOfTwo(int _min, int _max, string kernelString, bool beginningText)
    {
        breaker = false;

        if(beginningText)
            Debug.Log("BEGINNING VALIDATE POWERS OF TWO.");

        int validCount = 0;
        for (int s = _min; s <= _max; ++s)
        {
            TestAtSize(1 << s, ref validCount, kernelString);
            yield return new WaitForSeconds(.25f);
        }

        if (validCount == _max + 1 - _min)
            Debug.Log(kernelString + " [" + validCount + "/" + (_max + 1 - _min) + "] ALL TESTS PASSED");
        else
            Debug.Log(kernelString + " FAILED. [" + validCount + "/" + (_max + 1 - _min) + "] PASSED");

        UpdateSize(size);
        breaker = true;
    }

    public virtual IEnumerator ValidateAllOffSizes(string kernelString)
    {
        breaker = false;
        Debug.Log("Beginning Validate All Off Sizes. This may take a while.");
        int validCount = 0;
        for (int i = 1; i <= 1024; ++i)
        {
            TestAtSize((1 << 21) + i, ref validCount, kernelString);
            yield return new WaitForSeconds(.05f);
            if ((i & 31) == 0)
                Debug.Log("Running");
        }

        for (int i = 1; i <= 8192; ++i)
        {
            TestAtSize((1 << 21) + (i << 8), ref validCount, kernelString);
            yield return new WaitForSeconds(.05f);
            if ((i & 31) == 0)
                Debug.Log("Running");
        }

        for (int i = 1; i <= 8; ++i)
        {
            TestAtSize((1 << 21) + (i << 18) + 31, ref validCount, kernelString);
            yield return new WaitForSeconds(.05f);

            TestAtSize((1 << 21) + (i << 18) + 33, ref validCount, kernelString);
            yield return new WaitForSeconds(.05f);
        }

        if (validCount == 1024 + 8192 + 16)
            Debug.Log("[" + validCount + "/8208] ALL TESTS PASSED");
        else
            Debug.Log("TEST FAILED. [" + validCount + "/8208] PASSED");

        UpdateSize(size);
        breaker = true;
    }

    public virtual void TestAtSize(int _size, ref int count, string kernelString)
    {
        validationArray = new uint[_size];
        UpdateSize(_size);
        ResetBuffers();
        DispatchKernels();
        prefixSumBuffer.GetData(validationArray);
        if (ValAndBreak())
            count++;
        else
            Debug.LogError(kernelString + " FAILED AT SIZE: " + _size);
    }

    public virtual bool ValAndBreak()
    {
        for (uint i = 0; i < validationArray.Length; ++i)
        {
            if (validationArray[i] != (i + 1))
                return false;
        }
        return true;
    }

    public virtual bool Val()
    {
        for (uint i = 0; i < validationArray.Length; ++i)
        {
            if (validationArray[i] != (i + 1))
            {
                Debug.LogError("Sum Failed");
                return false;
            }
        }
        Debug.Log("Sum Passed");
        return true;
    }

    public virtual bool ValWithText()
    {
        bool isValidated = true;
        int errCount = 0;
        for (uint i = 0; i < validationArray.Length; ++i)
        {
            if (validationArray[i] != (i + 1))
            {
                isValidated = false;
                if (validateText)
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

        if (isValidated)
            Debug.Log("Sum Passed");
        else
            Debug.LogError("Sum Failed");
        return isValidated;
    }
}

