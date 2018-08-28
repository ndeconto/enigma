#ifndef ENIGMA_H
#define ENIGMA_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FAILURE 1

/* 
 * if the cipher text is too short, it's of course more difficult to decrypt it
 * but it's not easier to decrypt an infinitely long text than a medium text
 * However, if the text is too long, it does not fit in fast GPU caches and 
 * performances decrease. So we limit the considered input 
*/
#define MAX_INPUT_SIZE 	65536

/* maximum number of rotors that can be used */
#define MAX_ROTORS		6
/* maximum size of the rotor set */
#define MAX_ROTOR_SET_SIZE 8
#define CONST_MEM_SIZE 65536
#define ROTOR_DATA_SIZE (26 * MAX_ROTOR_SET_SIZE) 
#define CHOSEN_MEM_SIZE (CONST_MEM_SIZE - ROTOR_DATA_SIZE)

#define BLOCK_SIZE 		1024
/*
 * if there are too many rotors (6 or more), we cannot use one grid describing
 * all possible keys because it does not fit in memory (memory space ~ 1 GB,
 * but 26**6 * 6! = 222 GB)
 */
#define MAX_DIM_GRID 	10000

#define DETECTION_THRESHOLD 0.067

typedef unsigned long long int uint64_t;


extern __global__ void decrypt_kernel(const char* devCipherText, float* IC, int n, int N);
extern __host__ void precomputationKeyToInt(char* chosenRotorsMemory, int n, int N);

extern __constant__ char cChosenMemory[];

#endif