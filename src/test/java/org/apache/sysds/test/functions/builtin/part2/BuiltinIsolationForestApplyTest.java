/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.builtin.part2;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public class BuiltinIsolationForestApplyTest extends AutomatedTestBase {
	private final static String TEST_NAME = "outlierByIsolationForestApply";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinIsolationForestApplyTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int rows = 50;
	private final static int cols = 3;
	private final static int n_trees = 10;
	private final static int subsampling_size = 20;
	private final static int seed = 42;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,
			new String[]{"scores"}));
	}

	@Test
	public void testIsolationForestApplyCP() {
		runIsolationForestApplyTest(false, ExecType.CP);
	}

	@Test
	public void testIsolationForestApplySP() {
		runIsolationForestApplyTest(false, ExecType.SPARK);
	}

	@Test
	public void testIsolationForestApplyWithOutliersCP() {
		runIsolationForestApplyTest(true, ExecType.CP);
	}

	@Test
	public void testIsolationForestApplyWithOutliersSP() {
		runIsolationForestApplyTest(true, ExecType.SPARK);
	}

	private void runIsolationForestApplyTest(boolean withOutliers, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			// First, train a model
			fullDMLScriptName = HOME + "outlierByIsolationForest.dml";
			programArgs = new String[]{"-nvargs",
				"X=" + input("A"),
				"n_trees=" + n_trees,
				"subsampling_size=" + subsampling_size,
				"seed=" + seed,
				"model_output=" + output("model"),
				"subsampling_size_output=" + output("subsampling_size")};

			// Generate training data
			double[][] A = getRandomMatrix(rows, cols, -2, 2, 0.7, seed);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);

			// Now, apply the model to test data
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			int testRows = withOutliers ? 10 : 20;
			double[][] testData = new double[testRows][cols];

			if (withOutliers) {
				// Half normal, half outliers
				for (int i = 0; i < 5; i++) {
					for (int j = 0; j < cols; j++) {
						testData[i][j] = (Math.random() - 0.5) * 4; // Normal
					}
				}
				for (int i = 5; i < 10; i++) {
					for (int j = 0; j < cols; j++) {
						testData[i][j] = 8 + Math.random() * 4; // Outliers
					}
				}
			} else {
				// All normal data
				for (int i = 0; i < testRows; i++) {
					for (int j = 0; j < cols; j++) {
						testData[i][j] = (Math.random() - 0.5) * 4;
					}
				}
			}

			writeInputMatrixWithMTD("X", testData, true);

			programArgs = new String[]{"-nvargs",
				"X=" + input("X"),
				"model=" + output("model"),
				"subsampling_size=" + subsampling_size,
				"output=" + output("scores")};

			runTest(true, false, null, -1);

			// Verify scores
			HashMap<CellIndex, Double> scores = readDMLMatrixFromOutputDir("scores");
			Assert.assertNotNull("Scores should not be null", scores);
			Assert.assertEquals("Should have score for each test sample",
				testRows, scores.size());

			// Verify score properties
			for (CellIndex idx : scores.keySet()) {
				double score = scores.get(idx);
				// Scores should be between 0 and 1
				Assert.assertTrue("Score should be >= 0", score >= 0);
				Assert.assertTrue("Score should be <= 1", score <= 1);
			}

			if (withOutliers) {
				// Verify that outliers have higher scores
				double avgNormalScore = 0;
				double avgOutlierScore = 0;

				for (int i = 1; i <= 5; i++) {
					avgNormalScore += scores.get(new CellIndex(i, 1));
				}
				avgNormalScore /= 5;

				for (int i = 6; i <= 10; i++) {
					avgOutlierScore += scores.get(new CellIndex(i, 1));
				}
				avgOutlierScore /= 5;

				// Outliers should generally have higher scores
				// (This might occasionally fail due to randomness, but should be reliable with fixed seed)
				Assert.assertTrue("Outliers should have higher average score than normal points: " +
					"normal=" + avgNormalScore + ", outlier=" + avgOutlierScore,
					avgOutlierScore > avgNormalScore);
			}
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
