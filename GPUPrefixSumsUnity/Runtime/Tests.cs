/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/10/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
using System.Collections;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

namespace GPUPrefixSums.Runtime
{
    public class Tests : MonoBehaviour
    {
        [SerializeField]
        ComputeShader csdl;

        [SerializeField]
        ComputeShader rts;

        [SerializeField]
        ComputeShader m_util;

        private const int k_partitionSize = 3072;
        private const int k_maxDispatch = 65535;

        private ComputeBuffer scan;
        private ComputeBuffer scanValidation;
        private ComputeBuffer threadBlockReduction;
        private ComputeBuffer index;
        private ComputeBuffer errCount;

        private CommandBuffer m_cmd;

        private ChainedScanDecoupledLookback m_csdl;
        private ReduceThenScan m_rts;

        private bool m_isValid;
        private int m_kernelInitOne = -1;
        private int m_kernelInitRandom = -1;
        private int m_kernelClearErrors = -1;
        private int m_kernelValidateOneInclusive = -1;
        private int m_kernelValidateOneExclusive = -1;
        private int m_kernelValidateRandomInclusive = -1;
        private int m_kernelValidateRandomExclusive = -1;

        void Start()
        {
            if (m_util)
            {
                m_kernelInitOne = m_util.FindKernel("InitOne");
                m_kernelInitRandom = m_util.FindKernel("InitRandom");
                m_kernelClearErrors = m_util.FindKernel("ClearErrorCount");
                m_kernelValidateOneInclusive = m_util.FindKernel("ValidateOneInclusive");
                m_kernelValidateOneExclusive = m_util.FindKernel("ValidateOneExclusive");
                m_kernelValidateRandomInclusive = m_util.FindKernel("ValidateRandomInclusive");
                m_kernelValidateRandomExclusive = m_util.FindKernel("ValidateRandomExclusive");
            }

            m_isValid = m_kernelInitOne >= 0 &&
                        m_kernelInitRandom >= 0 &&
                        m_kernelClearErrors >= 0 &&
                        m_kernelValidateOneInclusive >= 0 &&
                        m_kernelValidateOneExclusive >= 0 &&
                        m_kernelValidateRandomInclusive >= 0 &&
                        m_kernelValidateRandomExclusive >= 0;

            if (m_isValid)
            {
                if (!m_util.IsSupported(m_kernelInitOne) ||
                   !m_util.IsSupported(m_kernelInitRandom) ||
                   !m_util.IsSupported(m_kernelClearErrors) ||
                   !m_util.IsSupported(m_kernelValidateOneInclusive) ||
                   !m_util.IsSupported(m_kernelValidateOneExclusive) ||
                   !m_util.IsSupported(m_kernelValidateRandomInclusive) ||
                   !m_util.IsSupported(m_kernelValidateRandomExclusive))
                {
                    m_isValid = false;
                }
            }
            Assert.IsTrue(m_isValid);

            Initialize();
            StartCoroutine(TestAll());
        }

        private void Initialize()
        {
            m_cmd = new CommandBuffer();

            scan = new ComputeBuffer(1 << 26, sizeof(uint) * 4);
            errCount = new ComputeBuffer(1, sizeof(uint));
            scanValidation = new ComputeBuffer(1 << 18, sizeof(uint) * 4);
        }

        private void InitCSDL()
        {
            m_csdl = new ChainedScanDecoupledLookback(
                csdl,
                1 << 28,
                ref threadBlockReduction,
                ref index);
        }

        private void InitRTS()
        {
            m_rts = new ReduceThenScan(
                rts,
                1 << 28,
                ref threadBlockReduction);
        }

        private void PreScan(int _testSize, bool isRandom)
        {
            m_util.SetInt("e_vectorizedSize", VectorizedSize(_testSize));

            if (isRandom)
            {
                m_util.SetBuffer(m_kernelInitRandom, "b_scan", scan);
                m_util.SetBuffer(m_kernelInitRandom, "b_scanValidation", scanValidation);
                m_util.Dispatch(m_kernelInitRandom, 256, 1, 1);
            }
            else
            {
                m_util.SetBuffer(m_kernelInitOne, "b_scan", scan);
                m_util.Dispatch(m_kernelInitOne, 256, 1, 1);
            }
        }

        private bool PostScan(bool isRandom, bool isInclusive, bool shouldPrint)
        {
            m_util.SetBuffer(m_kernelClearErrors, "b_errorCount", errCount);
            m_util.Dispatch(m_kernelClearErrors, 1, 1, 1);

            if (isRandom)
            {
                if (isInclusive)
                {
                    m_util.SetBuffer(m_kernelValidateRandomInclusive, "b_errorCount", errCount);
                    m_util.SetBuffer(m_kernelValidateRandomInclusive, "b_scan", scan);
                    m_util.SetBuffer(m_kernelValidateRandomInclusive, "b_scanValidation", scanValidation);
                    m_util.Dispatch(m_kernelValidateRandomInclusive, 1, 1, 1);
                }
                else
                {
                    m_util.SetBuffer(m_kernelValidateRandomExclusive, "b_errorCount", errCount);
                    m_util.SetBuffer(m_kernelValidateRandomExclusive, "b_scan", scan);
                    m_util.SetBuffer(m_kernelValidateRandomExclusive, "b_scanValidation", scanValidation);
                    m_util.Dispatch(m_kernelValidateRandomExclusive, 1, 1, 1);
                }
            }
            else
            {
                if (isInclusive)
                {
                    m_util.SetBuffer(m_kernelValidateOneInclusive, "b_errorCount", errCount);
                    m_util.SetBuffer(m_kernelValidateOneInclusive, "b_scan", scan);
                    m_util.Dispatch(m_kernelValidateOneInclusive, 256, 1, 1);
                }
                else
                {
                    m_util.SetBuffer(m_kernelValidateOneExclusive, "b_errorCount", errCount);
                    m_util.SetBuffer(m_kernelValidateOneExclusive, "b_scan", scan);
                    m_util.Dispatch(m_kernelValidateOneExclusive, 256, 1, 1);
                }
            }

            uint[] errors = new uint[1];
            errCount.GetData(errors);

            if (errors[0] == 0)
            {
                if (shouldPrint)
                    Debug.Log("Test passed");
                return true;
            }
            else
            {
                if (shouldPrint)
                    Debug.LogError("Test Failed: " + errors[0] + " errors.");
                return false;
            }
        }

        private bool CSDLTestOneInclusive(int testSize)
        {
            PreScan(testSize, false);
            m_csdl.PrefixSumInclusive(
                testSize,
                scan,
                threadBlockReduction,
                index);
            return PostScan(false, true, false);
        }

        private bool CSDLTestOneExclusive(int testSize)
        {
            PreScan(testSize, false);
            m_csdl.PrefixSumExclusive(
                testSize,
                scan,
                threadBlockReduction,
                index);
            return PostScan(false, false, false);
        }

        private bool CSDLTestRandomInclusive(int testSize)
        {
            PreScan(testSize, true);
            m_csdl.PrefixSumInclusive(
                testSize,
                scan,
                threadBlockReduction,
                index);
            return PostScan(true, true, false);
        }

        private bool CSDLTestRandomExclusive(int testSize)
        {
            PreScan(testSize, true);
            m_csdl.PrefixSumExclusive(
                testSize,
                scan,
                threadBlockReduction,
                index);
            return PostScan(true, false, false);
        }

        private bool CSDLTestOneInclusiveCmd(int testSize)
        {
            PreScan(testSize, false);
            m_cmd.Clear();
            m_csdl.PrefixSumInclusive(
                m_cmd,
                testSize,
                scan,
                threadBlockReduction,
                index);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostScan(false, true, false);
        }

        private bool CSDLTestOneExclusiveCmd(int testSize)
        {
            PreScan(testSize, false);
            m_cmd.Clear();
            m_csdl.PrefixSumExclusive(
                m_cmd,
                testSize,
                scan,
                threadBlockReduction,
                index);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostScan(false, false, false);
        }

        private bool CSDLTestRandomInclusiveCmd(int testSize)
        {
            PreScan(testSize, true);
            m_cmd.Clear();
            m_csdl.PrefixSumInclusive(
                m_cmd,
                testSize,
                scan,
                threadBlockReduction,
                index);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostScan(true, true, false);
        }

        private bool CSDLTestRandomExclusiveCmd(int testSize)
        {
            PreScan(testSize, true);
            m_cmd.Clear();
            m_csdl.PrefixSumExclusive(
                m_cmd,
                testSize,
                scan,
                threadBlockReduction,
                index);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostScan(true, false, false);
        }

        private bool RTSTestOneInclusive(int testSize)
        {
            PreScan(testSize, false);
            m_rts.PrefixSumInclusive(
                testSize,
                scan,
                threadBlockReduction);
            return PostScan(false, true, false);
        }

        private bool RTSTestOneExclusive(int testSize)
        {
            PreScan(testSize, false);
            m_rts.PrefixSumExclusive(
                testSize,
                scan,
                threadBlockReduction);
            return PostScan(false, false, false);
        }

        private bool RTSTestRandomInclusive(int testSize)
        {
            PreScan(testSize, true);
            m_rts.PrefixSumInclusive(
                testSize,
                scan,
                threadBlockReduction);
            return PostScan(true, true, false);
        }

        private bool RTSTestRandomExclusive(int testSize)
        {
            PreScan(testSize, true);
            m_rts.PrefixSumExclusive(
                testSize,
                scan,
                threadBlockReduction);
            return PostScan(true, false, false);
        }

        private bool RTSTestOneInclusiveCmd(int testSize)
        {
            PreScan(testSize, false);
            m_cmd.Clear();
            m_rts.PrefixSumInclusive(
                m_cmd,
                testSize,
                scan,
                threadBlockReduction);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostScan(false, true, false);
        }

        private bool RTSTestOneExclusiveCmd(int testSize)
        {
            PreScan(testSize, false);
            m_cmd.Clear();
            m_rts.PrefixSumExclusive(
                m_cmd,
                testSize,
                scan,
                threadBlockReduction);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostScan(false, false, false);
        }

        private bool RTSTestRandomInclusiveCmd(int testSize)
        {
            PreScan(testSize, true);
            m_cmd.Clear();
            m_rts.PrefixSumInclusive(
                m_cmd,
                testSize,
                scan,
                threadBlockReduction);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostScan(true, true, false);
        }

        private bool RTSTestRandomExclusiveCmd(int testSize)
        {
            PreScan(testSize, true);
            m_cmd.Clear();
            m_rts.PrefixSumExclusive(
                m_cmd,
                testSize,
                scan,
                threadBlockReduction);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostScan(true, false, false);
        }

        private static void PrintTestResults(
            int passed,
            int expected,
            string testName)
        {
            if (passed == expected)
                Debug.Log(expected + " / " + expected + " All " + testName + " tests passed.");
            else
                Debug.LogError(passed + " / " + expected + " " + testName + " test failed.");
        }

        private IEnumerator CSDLTest()
        {
            int totalCSDLTestsPassed = 0;

            int csdlInclusiveOnePassed = 0;
            int csdlExclusiveOnePassed = 0;
            int csdlInclusiveRandomPassed = 0;
            int csdlExclusiveRandomPassed = 0;

            int csdlInclusiveOnePassedCmd = 0;
            int csdlExclusiveOnePassedCmd = 0;
            int csdlInclusiveRandomPassedCmd = 0;
            int csdlExclusiveRandomPassedCmd = 0;

            int csdlInclusiveOneSplitPassed = 0;
            int csdlExclusiveOneSplitPassed = 0;
            int csdlInclusiveOneSplitPassedCmd = 0;
            int csdlExclusiveOneSplitPassedCmd = 0;

            const int passStart1 = 1 << 25;
            const int passEnd1 = (1 << 25) + k_partitionSize;

            const int passStart2 = k_partitionSize;
            const int passEnd2 = k_partitionSize * 2;

            const int passStart3 = k_partitionSize * (k_maxDispatch + 1) - 1024;
            const int passEnd3 = passStart3 + 1024;

            const int expected = k_partitionSize / 4;
            const int splitExpected = 256;
            const int totalExpected = (expected * 8) + (splitExpected * 4);

            //One
            for (int i = passStart1; i < passEnd1; i += 4)
            {
                yield return csdlInclusiveOnePassed +=
                    CSDLTestOneInclusive(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlInclusiveOnePassed;
            PrintTestResults(csdlInclusiveOnePassed, expected, "CSDL One Inclusive");

            for (int i = passStart1; i < passEnd1; i += 4)
            {
                yield return csdlExclusiveOnePassed +=
                    CSDLTestOneExclusive(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlExclusiveOnePassed;
            PrintTestResults(csdlExclusiveOnePassed, expected, "CSDL One Exclusive");

            for (int i = passStart1; i < passEnd1; i += 4)
            {
                yield return csdlInclusiveOnePassedCmd +=
                    CSDLTestOneInclusiveCmd(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlInclusiveOnePassedCmd;
            PrintTestResults(csdlInclusiveOnePassedCmd, expected, "CSDL One Inclusive Cmd");

            for (int i = passStart1; i < passEnd1; i += 4)
            {
                yield return csdlExclusiveOnePassedCmd +=
                    CSDLTestOneExclusiveCmd(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlExclusiveOnePassedCmd;
            PrintTestResults(csdlExclusiveOnePassedCmd, expected, "CSDL One Exclusive Cmd");

            //Random
            for (int i = passStart2; i < passEnd2; i += 4)
            {
                yield return csdlInclusiveRandomPassed +=
                    CSDLTestRandomInclusive(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlInclusiveRandomPassed;
            PrintTestResults(csdlInclusiveRandomPassed, expected, "CSDL Random Inclusive");

            for (int i = passStart2; i < passEnd2; i += 4)
            {
                yield return csdlExclusiveRandomPassed +=
                    CSDLTestRandomExclusive(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlExclusiveRandomPassed;
            PrintTestResults(csdlExclusiveRandomPassed, expected, "CSDL Random Exclusive");

            for (int i = passStart2; i < passEnd2; i += 4)
            {
                yield return csdlInclusiveRandomPassedCmd +=
                    CSDLTestRandomInclusiveCmd(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlInclusiveRandomPassedCmd;
            PrintTestResults(csdlInclusiveRandomPassedCmd, expected, "CSDL Random Inclusive Cmd");

            for (int i = passStart2; i < passEnd2; i += 4)
            {
                yield return csdlExclusiveRandomPassedCmd +=
                    CSDLTestRandomExclusiveCmd(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlExclusiveRandomPassedCmd;
            PrintTestResults(csdlExclusiveRandomPassedCmd, expected, "CSDL Random Exclusive Cmd");

            //Very Large/Split testing
            for (int i = passStart3; i < passEnd3; i += 4)
            {
                yield return csdlInclusiveOneSplitPassed +=
                    CSDLTestOneInclusive(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlInclusiveOneSplitPassed;
            PrintTestResults(csdlInclusiveOneSplitPassed, splitExpected, "CSDL Split Inclusive");

            for (int i = passStart3; i < passEnd3; i += 4)
            {
                yield return csdlExclusiveOneSplitPassed +=
                    CSDLTestOneExclusive(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlExclusiveOneSplitPassed;
            PrintTestResults(csdlExclusiveOneSplitPassed, splitExpected, "CSDL Split Exclusive");

            for (int i = passStart3; i < passEnd3; i += 4)
            {
                yield return csdlInclusiveOneSplitPassedCmd +=
                    CSDLTestOneInclusiveCmd(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlInclusiveOneSplitPassedCmd;
            PrintTestResults(csdlInclusiveOneSplitPassedCmd, splitExpected, "CSDL Split Inclusive Cmd");

            for (int i = passStart3; i < passEnd3; i += 4)
            {
                yield return csdlExclusiveOneSplitPassedCmd +=
                    CSDLTestOneExclusiveCmd(i) ? 1 : 0;
            }
            totalCSDLTestsPassed += csdlExclusiveOneSplitPassedCmd;
            PrintTestResults(csdlExclusiveOneSplitPassedCmd, splitExpected, "CSDL Split Exclusive Cmd");

            Debug.Log("CSDL TESTS COMPLETED");
            if (totalCSDLTestsPassed == totalExpected)
                Debug.Log(totalExpected + " / " + totalExpected + " ALL CSDL TESTS PASSED");
            else
                Debug.LogError(totalCSDLTestsPassed + " / " + totalExpected + " CSDL TEST FAILED");

            //Sanity check
            /*uint[] sanity = new uint[scan.count];
            scan.GetData(sanity);

            for (int i = scan.count - 1024; i < scan.count; ++i)
                Debug.Log(sanity[i]);*/
        }

        private IEnumerator RTSTest()
        {
            int totalRTSTestsPassed = 0;

            int rtsInclusiveOnePassed = 0;
            int rtsExclusiveOnePassed = 0;
            int rtsInclusiveRandomPassed = 0;
            int rtsExclusiveRandomPassed = 0;

            int rtsInclusiveOnePassedCmd = 0;
            int rtsExclusiveOnePassedCmd = 0;
            int rtsInclusiveRandomPassedCmd = 0;
            int rtsExclusiveRandomPassedCmd = 0;

            int rtsInclusiveOneSplitPassed = 0;
            int rtsExclusiveOneSplitPassed = 0;
            int rtsInclusiveOneSplitPassedCmd = 0;
            int rtsExclusiveOneSplitPassedCmd = 0;

            const int passStart1 = 1 << 25;
            const int passEnd1 = passStart1 + k_partitionSize;

            const int passStart2 = k_partitionSize;
            const int passEnd2 = passStart2 + k_partitionSize;

            const int passStart3 = k_partitionSize * (k_maxDispatch + 1) - 1024;
            const int passEnd3 = passStart3 + 1024;

            const int expected = k_partitionSize / 4;
            const int splitExpected = 256;
            const int totalExpected = (expected * 8) + (splitExpected * 4);

            //One
            for (int i = passStart1; i < passEnd1; i += 4)
            {
                yield return rtsInclusiveOnePassed +=
                    RTSTestOneInclusive(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsInclusiveOnePassed;
            PrintTestResults(rtsInclusiveOnePassed, expected, "RTS One Inclusive");

            for (int i = passStart1; i < passEnd1; i += 4)
            {
                yield return rtsExclusiveOnePassed +=
                    RTSTestOneExclusive(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsExclusiveOnePassed;
            PrintTestResults(rtsExclusiveOnePassed, expected, "RTS One Exclusive");

            for (int i = passStart1; i < passEnd1; i += 4)
            {
                yield return rtsInclusiveOnePassedCmd +=
                    RTSTestOneInclusiveCmd(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsInclusiveOnePassedCmd;
            PrintTestResults(rtsInclusiveOnePassedCmd, expected, "RTS One Inclusive Cmd");

            for (int i = passStart1; i < passEnd1; i += 4)
            {
                yield return rtsExclusiveOnePassedCmd +=
                    RTSTestOneExclusiveCmd(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsExclusiveOnePassedCmd;
            PrintTestResults(rtsExclusiveOnePassedCmd, expected, "RTS One Exclusive Cmd");

            //Random
            for (int i = passStart2; i < passEnd2; i += 4)
            {
                yield return rtsInclusiveRandomPassed +=
                    RTSTestRandomInclusive(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsInclusiveRandomPassed;
            PrintTestResults(rtsInclusiveRandomPassed, expected, "RTS Random Inclusive");

            for (int i = passStart2; i < passEnd2; i += 4)
            {
                yield return rtsExclusiveRandomPassed +=
                    RTSTestRandomExclusive(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsExclusiveRandomPassed;
            PrintTestResults(rtsExclusiveRandomPassed, expected, "RTS Random Exclusive");

            for (int i = passStart2; i < passEnd2; i += 4)
            {
                yield return rtsInclusiveRandomPassedCmd +=
                    RTSTestRandomInclusiveCmd(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsInclusiveRandomPassedCmd;
            PrintTestResults(rtsInclusiveRandomPassedCmd, expected, "RTS Random Inclusive Cmd");

            for (int i = passStart2; i < passEnd2; i += 4)
            {
                yield return rtsExclusiveRandomPassedCmd +=
                    RTSTestRandomExclusiveCmd(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsExclusiveRandomPassedCmd;
            PrintTestResults(rtsExclusiveRandomPassedCmd, expected, "RTS Random Exclusive Cmd");

            //Very Large/Split testing
            for (int i = passStart3; i < passEnd3; i += 4)
            {
                yield return rtsInclusiveOneSplitPassed +=
                    RTSTestOneInclusive(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsInclusiveOneSplitPassed;
            PrintTestResults(rtsInclusiveOneSplitPassed, splitExpected, "RTS Split Inclusive");

            for (int i = passStart3; i < passEnd3; i += 4)
            {
                yield return rtsExclusiveOneSplitPassed +=
                    RTSTestOneExclusive(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsExclusiveOneSplitPassed;
            PrintTestResults(rtsExclusiveOneSplitPassed, splitExpected, "RTS Split Exclusive");

            for (int i = passStart3; i < passEnd3; i += 4)
            {
                yield return rtsInclusiveOneSplitPassedCmd +=
                    RTSTestOneInclusiveCmd(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsInclusiveOneSplitPassedCmd;
            PrintTestResults(rtsInclusiveOneSplitPassedCmd, splitExpected, "RTS Split Inclusive Cmd");

            for (int i = passStart3; i < passEnd3; i += 4)
            {
                yield return rtsExclusiveOneSplitPassedCmd +=
                    RTSTestOneExclusiveCmd(i) ? 1 : 0;
            }
            totalRTSTestsPassed += rtsExclusiveOneSplitPassedCmd;
            PrintTestResults(rtsExclusiveOneSplitPassedCmd, splitExpected, "RTS Split Exclusive Cmd");

            Debug.Log("RTS TESTS COMPLETED");
            if (totalRTSTestsPassed == totalExpected)
                Debug.Log(totalExpected + " / " + totalExpected + " ALL RTS TESTS PASSED");
            else
                Debug.LogError(totalRTSTestsPassed + " / " + totalExpected + " RTS TEST FAILED");

            //Sanity check
            /*uint[] sanity = new uint[scan.count];
            scan.GetData(sanity);

            for (int i = scan.count - 1024; i < scan.count; ++i)
                Debug.Log(sanity[i]);*/
        }

        private IEnumerator TestAll()
        {
            Debug.Log("Beginning Chained Scan Decoupled Lookback Test All");
            InitCSDL();
            yield return StartCoroutine(CSDLTest());

            Debug.Log("Beginning Reduce Then Scan Test All");
            InitRTS();
            yield return StartCoroutine(RTSTest());
        }

        static int VectorizedSize(int x)
        {
            return (x + 3) / 4;
        }
        private void OnDestroy()
        {
            scan?.Dispose();
            scanValidation?.Dispose();
            threadBlockReduction?.Dispose();
            index?.Dispose();
            errCount?.Dispose();
        }
    }
}