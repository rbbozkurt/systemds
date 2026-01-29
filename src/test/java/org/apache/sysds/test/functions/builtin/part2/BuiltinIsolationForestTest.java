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

public class BuiltinIsolationForestTest extends AutomatedTestBase {
	private final static String TEST_NAME = "outlierByIsolationForest";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinIsolationForestTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int rows = 100;
	private final static int cols = 3;
	private final static int n_trees = 10;
	private final static int subsampling_size = 20;
	private final static int seed = 42;

	@Override
	public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,
			new String[]{"model", "subsampling_size"}));
	}

	@Test
	public void testIsolationForestCP() {
		runIsolationForestTest(ExecType.CP);
	}

	@Test
	public void testIsolationForestSP() {
		runIsolationForestTest(ExecType.SPARK);
	}

	@Test
	public void testIsolationForestWithOutliersCP() {
		runIsolationForestWithOutliersTest(ExecType.CP);
	}

	@Test
	public void testIsolationForestWithOutliersSP() {
		runIsolationForestWithOutliersTest(ExecType.SPARK);
	}

	private void runIsolationForestTest(ExecType instType) {
		ExecMode platformOld = setExecMode(instType);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-nvargs",
				"X=" + input("A"),
				"n_trees=" + n_trees,
				"subsampling_size=" + subsampling_size,
				"seed=" + seed,
				"model_output=" + output("model"),
				"subsampling_size_output=" + output("subsampling_size")};

			// Generate normal data (no outliers)
			double[][] A = getRandomMatrix(rows, cols, -5, 5, 0.7, seed);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);

			// Verify model was created
			HashMap<CellIndex, Double> model = readDMLMatrixFromOutputDir("model");
			Assert.assertNotNull("Model should not be null", model);
			Assert.assertTrue("Model should have entries", model.size() > 0);

			// Verify subsampling size was stored correctly
			HashMap<CellIndex, Double> subsamplingSize = readDMLScalarFromOutputDir("subsampling_size");
			Assert.assertEquals("Subsampling size should match",
				(double) subsampling_size,
				subsamplingSize.get(new CellIndex(1, 1)),
				eps);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private void runIsolationForestWithOutliersTest(ExecType instType) {
		ExecMode platformOld = setExecMode(instType);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-nvargs",
				"X=" + input("A"),
				"n_trees=" + n_trees,
				"subsampling_size=" + subsampling_size,
				"seed=" + seed,
				"model_output=" + output("model"),
				"subsampling_size_output=" + output("subsampling_size")};

			// Generate data with clear outliers
			// Most data is around 0, outliers are far away
			double[][] A = new double[rows][cols];
			for (int i = 0; i < rows - 5; i++) {
				for (int j = 0; j < cols; j++) {
					// Normal data: mean=0, range=[-2, 2]
					A[i][j] = (Math.random() - 0.5) * 4;
				}
			}
			// Add outliers: far from normal data
			for (int i = rows - 5; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					// Outliers: mean=10, range=[8, 12]
					A[i][j] = 8 + Math.random() * 4;
				}
			}

			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);

			// Verify model structure
			HashMap<CellIndex, Double> model = readDMLMatrixFromOutputDir("model");
			Assert.assertNotNull("Model should not be null", model);
			Assert.assertTrue("Model should have entries", model.size() > 0);

			// Verify model has n_trees rows
			int maxRow = 0;
			for (CellIndex idx : model.keySet()) {
				maxRow = Math.max(maxRow, idx.row);
			}
			Assert.assertEquals("Model should have n_trees rows", n_trees, maxRow);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
