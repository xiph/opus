// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the Chromium LICENSE file.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define SKIP_CONFIG_H

#ifndef CUSTOM_MODES
#define CUSTOM_MODES
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define CELT_C
#include "pitch.h"

#if !defined(FIXED_POINT)
#if defined(OPUS_X86_MAY_HAVE_SSE)
#include "x86/pitch_sse.c"
#define SIMD_TYPE               "SSE"
#define xcorr_kernel_simd       xcorr_kernel_sse
#define dual_inner_prod_simd    dual_inner_prod_sse
#define celt_inner_prod_simd    celt_inner_prod_sse
#define comb_filter_const_simd  comb_filter_const_sse
#else
#pragma message "No SIMD implementation found, using C definitions"
#define SIMD_TYPE               "C"
#define xcorr_kernel_simd       xcorr_kernel_c
#define dual_inner_prod_simd    dual_inner_prod_simd_c
#define celt_inner_prod_simd    celt_inner_prod_c
#define comb_filter_const_simd  comb_filter_const_c
#endif
#else
#error "This test works only when FIXED_POINT is NOT enabled"
#endif

#define PITCH_TEST_FAIL	0
#define PITCH_TEST_PASS	1

// Generates random float numbers between min and max.
// Split [min, max] interval into 1/step values.
static inline float randf(float min, float max, float step) {
	long range = (max - min) / step;
	long rint = rand() % (range + 1);

	return min + rint * step;
}

// Returns current time in seconds.
static inline double seconds() {
	struct timeval now;
	gettimeofday(&now, 0);
	return now.tv_sec + now.tv_usec * (1.0 / 1000000.0);
}

// Computes duration of function call. And accumulates in time variable.
#define TIME_FUNCTION(function, time) do {  \
		double start = seconds();           \
		(function);                         \
		*time += seconds() - start;         \
} while (0)

// Typedef and struct for handling individual tests.
typedef int (*test_fn)(const int, const float, const float, const float, double *, double *);
struct pitch_test {
	const char *name;
	test_fn function;
};

static int test_xcorr_kernel(const int data_size, const float min_range, const float max_range,
		const float step, double *time_c, double *time_simd) {
	int status = PITCH_TEST_PASS;
	float sum_c[4] = {0,}, sum_simd[4] = {0,};
	float x_c[data_size], y_c[data_size + 3];
	float x_simd[data_size], y_simd[data_size + 3];
	double rmse = 0.0;
	int i;

	// Generate random input data.
	for (i = 0; i < data_size; i++) {
		x_c[i] = randf(min_range, max_range, step);
		y_c[i] = randf(min_range, max_range, step);
	}
	for (;i < data_size + 3; i++) {
		y_c[i] = randf(min_range, max_range, step);
	}

	memcpy(x_simd, x_c, sizeof(float) * data_size);
	memcpy(y_simd, y_c, sizeof(float) * (data_size + 3));

	// Run C and SIMD function.
	TIME_FUNCTION(xcorr_kernel_c(x_c, y_c, sum_c, data_size), time_c);
	TIME_FUNCTION(xcorr_kernel_simd(x_simd, y_simd, sum_simd, data_size), time_simd);

	// Check error.
	for (i = 0; i < 4; i++) {
		rmse += (double)(sum_c[i] - sum_simd[i]) * (double)(sum_c[i] - sum_simd[i]);
	}
	rmse = sqrt(rmse / 4);
	if (rmse > step) {
		printf("Warning RMSE for %s in [%.6f, %.6f] = %.6lf\n",
				__FUNCTION__, min_range, max_range, rmse);
		status = PITCH_TEST_FAIL;
	}
	return status;
}

static int test_dual_inner_prod(const int data_size, const float min_range, const float max_range,
		const float step, double *time_c, double *time_simd) {
	int status = PITCH_TEST_PASS;
	float x_c[data_size], y_c[data_size], z_c[data_size];
	float x_simd[data_size], y_simd[data_size], z_simd[data_size];
	float xy1_c, xy2_c;
	float xy1_simd, xy2_simd;
	double rmse = 0.0;
	int i;

	// Generate random input data.
	for (i = 0; i < data_size; i++) {
		x_c[i] = randf(min_range, max_range, step);
		y_c[i] = randf(min_range, max_range, step);
		z_c[i] = randf(min_range, max_range, step);
	}

	memcpy(x_simd, x_c, sizeof(float) * data_size);
	memcpy(y_simd, y_c, sizeof(float) * data_size);
	memcpy(z_simd, z_c, sizeof(float) * data_size);

	// Run C and SIMD function.
	TIME_FUNCTION(dual_inner_prod_c(x_c, y_c, z_c, data_size, &xy1_c, &xy2_c), time_c);
	TIME_FUNCTION(dual_inner_prod_simd(x_simd, y_simd, z_simd, data_size, &xy1_simd, &xy2_simd), time_simd);

	// Check error.
	rmse += (double)(xy1_c - xy1_simd) * (double)(xy1_c - xy1_simd) +
			(double)(xy2_c - xy2_simd) * (double)(xy2_c - xy2_simd);

	rmse = sqrt(rmse / 2);
	if (rmse > step) {
		printf("Warning RMSE for %s in [%.6f, %.6f] = %.6lf\n",
				__FUNCTION__, min_range, max_range, rmse);
		status = PITCH_TEST_FAIL;
	}
	return status;
}

static int test_celt_inner_prod(const int data_size, const float min_range, const float max_range,
		const float step, double *time_c, double *time_simd) {
	int status = PITCH_TEST_PASS;
	float x_c[data_size], y_c[data_size];
	float x_simd[data_size], y_simd[data_size];
	float f_c;
	float f_simd;
	double rmse = 0.0;
	int i;

	// Generate random input data.
	for (i = 0; i < data_size; i++) {
		x_c[i] = randf(min_range, max_range, step);
		y_c[i] = randf(min_range, max_range, step);
	}

	memcpy(x_simd, x_c, sizeof(float) * data_size);
	memcpy(y_simd, y_c, sizeof(float) * data_size);

	// Run C and SIMD function.
	TIME_FUNCTION((f_c = celt_inner_prod_c(x_c, y_c, data_size)), time_c);
	TIME_FUNCTION((f_simd = celt_inner_prod_simd(x_simd, y_simd, data_size)), time_simd);

	// Check error.
	rmse += (double)(f_c - f_simd) * (double)(f_c - f_simd);

	rmse = sqrt(rmse);
	if (rmse > step) {
		printf("Warning RMSE for %s in [%.6f, %.6f] = %.6lf\n",
				__FUNCTION__, min_range, max_range, rmse);
		status = PITCH_TEST_FAIL;
	}
	return status;
}

// Function copied from celt.c.
static void comb_filter_const_c(opus_val32 *y, opus_val32 *x, int T, int N,
		opus_val16 g10, opus_val16 g11, opus_val16 g12) {
	opus_val32 x0, x1, x2, x3, x4;
	int i;
	x4 = x[-T-2];
	x3 = x[-T-1];
	x2 = x[-T];
	x1 = x[-T+1];
	for (i = 0; i < N; i++)
	{
		x0 = x[i - T + 2];
		y[i] = x[i]
		         + MULT16_32_Q15(g10, x2)
		         + MULT16_32_Q15(g11, ADD32(x1, x3))
		         + MULT16_32_Q15(g12, ADD32(x0, x4));
		x4 = x3;
		x3 = x2;
		x2 = x1;
		x1 = x0;
	}

}

static int test_comb_filter_const(const int data_size, const float min_range, const float max_range,
		const float step, double *time_c, double *time_simd) {
	int status = PITCH_TEST_PASS;
	float x_c[data_size], y_c[data_size];
	float x_simd[data_size], y_simd[data_size];
	double rmse = 0.0;
	int i;
	int N = data_size / 2;

	// Generate random input data.
	for (i = 0; i < data_size; i++) {
		x_c[i] = randf(min_range, max_range, step);
		y_c[i] = randf(min_range, max_range, step);
	}

	memcpy(x_simd, x_c, sizeof(float) * data_size);
	memcpy(y_simd, y_c, sizeof(float) * data_size);

	// Run C and SIMD function.
	TIME_FUNCTION(comb_filter_const_c(y_c + N, x_c + N, N, N,
			0.333f, 0.123f, 0.020f), time_c);
	TIME_FUNCTION(comb_filter_const_simd(y_simd + N, x_simd + N, N, N,
			0.333f, 0.123f, 0.020f), time_simd);

	// Check error.
	for (i = 0; i < data_size; i++) {
		rmse += (double)(y_c[i] - y_simd[i]) * (double)(y_c[i] - y_simd[i]);
	}
	rmse = sqrt(rmse / data_size);

	if (rmse > step) {
		printf("Warning RMSE for %s in [%.6f, %.6f] = %.6lf\n",
				__FUNCTION__, min_range, max_range, rmse);
		status = PITCH_TEST_FAIL;
	}
	return status;
}

static const struct pitch_test pitch_tests[] = {
		{"test_xcorr_kernel", test_xcorr_kernel},
		{"test_dual_inner_prod", test_dual_inner_prod},
		{"test_celt_inner_prod", test_celt_inner_prod},
		{"test_comb_filter_const", test_comb_filter_const}
};

static test_fn load_symbol(const char *s) {
	int i;
	int test_nr = sizeof(pitch_tests) / sizeof(pitch_tests[0]);

	for (i = 0; i < test_nr; i++) {
		if (strcmp(s, pitch_tests[i].name) == 0) {
			return pitch_tests[i].function;
		}
	}

	return NULL;
}

void print_usage(const char *exe) {
	int i;
	int test_nr = sizeof(pitch_tests) / sizeof(pitch_tests[0]);

	printf("Usage:\n\t%s <test_function> -i <iterations(optional)> "
			"-s <data_size(optional)> -e <error(optional)>\n", exe);
	printf("Available tests:\n");

	for (i = 0; i < test_nr; i++) {
		printf("\t%s\n", pitch_tests[i].name);
	}

	printf("\n");
}

int main(int argc, const char **argv) {
	int data_size = 1024;
	int iterations = 1;
	double time_c = 0.0, time_simd = 0.0;
	int i;
	int pass = 0;
	float step = 0.0001;

	const float min_range = -1, max_range = 1;
	const char *test_name = NULL;
	test_fn test = NULL;

	srand(0);

	if (argc < 2) {
		print_usage(argv[0]);
		// Running all tests with default values for "make check".
		int test_nr = sizeof(pitch_tests) / sizeof(pitch_tests[0]);
		for (i = 0; i < test_nr; i++) {
			pass += pitch_tests[i].function(data_size, min_range, max_range, step, &time_c, &time_simd);
		}
		return test_nr - pass;
	}
	else {
		test_name = argv[1];
		test = (test_fn)load_symbol(test_name);
		if (test == NULL) {
			fprintf(stderr, "Unable to find symbol: %s, exiting\n\n", test_name);
			print_usage(argv[0]);
			return 0;
		}
	}

	argv++;
	argc--;

	while (argc > 1) {
		if (strcmp(argv[1], "-i") == 0) {
			iterations = atoi(argv[2]);
		}
		else if (strcmp(argv[1], "-s") == 0) {
			data_size = atoi(argv[2]);
		}
		else if (strcmp(argv[1], "-e") == 0) {
			step = atof(argv[2]);
		}
		(--argc, ++argv);
	}

	for (i = 0; i < iterations; i++) {
		pass += test(data_size, min_range, max_range, step, &time_c, &time_simd);
	}

	// Print summary.
	printf("Results for %s, with %d iterations and %d data size:\n",
			test_name, iterations, data_size);
	printf("%.6lf seconds (avg: %.7lf)\tC time\n", time_c, time_c / iterations);
	printf("%.6lf seconds (avg: %.7lf)\t%s time\n", time_simd, time_simd / iterations, SIMD_TYPE);
	printf("%.6lf x\tSpeedup\n", time_c / time_simd);
	printf("%d/%d passed\n\n", pass, iterations);

	return iterations - pass;
}
