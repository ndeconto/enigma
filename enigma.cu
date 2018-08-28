#include "enigma.h"


/* see https://en.wikipedia.org/wiki/Enigma_rotor_details#Rotor_wiring_tables */
/* wiring translated in range [0, 25] instead of 'A'...'Z' */
__device__ int rotors[][26] = {
	{4, 10, 12, 5, 11, 6, 3, 16, 21, 25, 13, 19, 14, 22, 24, 7, 23, 20, 18, 15, 0, 8, 1, 17, 2, 9},
	{0, 9, 3, 10, 18, 8, 17, 20, 23, 1, 11, 7, 22, 19, 12, 2, 16, 6, 25, 13, 15, 24, 5, 21, 14, 4},
	{1, 3, 5, 7, 9, 11, 2, 15, 17, 19, 23, 21, 25, 13, 24, 4, 8, 22, 6, 0, 10, 12, 20, 18, 16, 14},
	{4, 18, 14, 21, 15, 25, 9, 0, 24, 16, 20, 8, 17, 7, 23, 11, 13, 5, 19, 6, 10, 3, 2, 12, 22, 1},
	{21, 25, 1, 17, 6, 8, 19, 24, 20, 15, 18, 3, 13, 7, 11, 23, 0, 22, 12, 9, 16, 14, 5, 4, 2, 10},
	{9, 15, 6, 21, 14, 20, 12, 5, 24, 16, 1, 4, 13, 7, 25, 17, 3, 10, 0, 18, 23, 11, 8, 2, 19, 22},
	{13, 25, 9, 7, 6, 17, 2, 23, 12, 24, 18, 22, 1, 14, 20, 5, 0, 8, 21, 11, 15, 4, 10, 16, 3, 19},
	{5, 10, 16, 7, 19, 11, 23, 14, 2, 1, 9, 18, 15, 3, 25, 17, 0, 12, 4, 22, 13, 8, 20, 24, 6, 21} 
};

/* TODO: is there a memory alignment problem using char instead of int? */
__constant__ char cChosenMemory[CHOSEN_MEM_SIZE];

__device__ unsigned int powerOf26[] = {
	1, 26, 676, 17576, 456976, 11881376, 308915776
};

/* productArray[N][n] = N * (N - 1) * ... * (N - n + 1) */ 
__device__ __constant__ unsigned short productArray[][MAX_ROTORS + 1] = {
	{1, 0, 0, 0, 0, 0, 0},
	{1, 1, 0, 0, 0, 0, 0},
	{1, 2, 2, 0, 0, 0, 0},
	{1, 3, 6, 6, 0, 0, 0},
	{1, 4, 12, 24, 24, 0, 0},
	{1, 5, 20, 60, 120, 120, 0},
	{1, 6, 30, 120, 360, 720, 720},
	{1, 7, 42, 210, 840, 2520, 5040},
	{1, 8, 56, 336, 1680, 6720, 20160}
};

/* In order to better handle parallelism, each possible key is associated to an
 *	integer. 
 */
__device__ __host__ void intToKey(uint64_t keyNumber, char* chosenRotors,
								char* rotorOffset, char n, char N) {
	
	//unsigned int r = keyNumber % powerOf26[n];	
	for (char k = 0; k < n; k++){
		//rotorOffset[k] = r % powerOf26[n - k] / powerOf26[n - k - 1] ;
	}
	//keyNumber /= powerOf26[n];
	for (char k = 0; k < n; k++){
		chosenRotors[k] = keyNumber % productArray[N - k][n - k] / productArray[N - k - 1][n - k - 1];
	}
}								


__host__ void precomputationKeyToInt(char* chosenRotorsMemory, int n, int N){
	char* perm = (char*) malloc(n * sizeof(char));
	for (int i = 0; i < n; i++) perm[i] = -1;
	char* used = (char*) calloc(N, sizeof(char));
	int i, j = 0, k = 0;
	while (k >= 0){
		for (i = perm[k] + 1; i < N; i++){
			if (used[i]) continue;
			if (perm[k] != -1) used[perm[k]] = 0;
			perm[k] = i;
			used[i] = 1;
			k++;
			break;
		}
		if (k == n) {
			memcpy(chosenRotorsMemory + j * n, perm, n * sizeof(char));
			j++;
			k--;
		}
		else if (i == N) {
			used[perm[k]] = 0;
			perm[k] = -1;
			k--;
		}
	}
	free(perm);
	free(used);
}


__host__ uint64_t keyToInt(char* chosenRotors, char* rotorOffset, char n,
							char N){
	uint64_t r = 0;
	for (char k = 0; k < n; k++){
		r += rotorOffset[k];
		r *= 26;
	}
	
	
	//use precomputation here!!!
	for (char k = n - 1; k >= 0; k++){
		//TODO: r += chosenRotors;
	}
	
	return 0; //bidon
	
}

/*
 * keyIndexOffset: tests the key whose number is in range
		[keyIndexOffset, keyIndexOffset + blockDim * dimGrid - 1]
 * IC: index of coincidence
 * n: number of rotors in the enigma machine
 * N: size of the rotor set 
 */
__global__ void decrypt_kernel(uint64_t keyIndexOffset, 
								const char* devCipherText, float* IC,
								char n, char N){
	extern __shared__ char s[];
	char* chosenRotors = s;
	char* rotorOffset = s + n;
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t keyNumber = i + keyIndexOffset;
	
	/* first, we compute the key parameters (i.e. chosen rotors,
	 * and the associated offsets) from the key number
	 */
	 //intToKey(keyNumber, chosenRotors, rotorOffset);

}