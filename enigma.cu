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

/* TODO: is there a memory alignment problem using uint8_t instead of int? */
uint8_t* chosenMemory;
__constant__ uint8_t cChosenMemory[CHOSEN_MEM_SIZE];
unsigned int powerOf26[] = {1, 26, 676, 17576, 456976, 11881376, 308915776};
__constant__ unsigned int cPowerOf26[MAX_ROTORS + 1];

/* In order to better handle parallelism, each possible key is associated to an
 *	integer. 
 */
__host__ void intToKeyHost(uint64_t keyNumber, uint8_t n, 
								uint8_t** chosenRotors, uint8_t* rotorOffset) {
	
	unsigned int r = keyNumber % powerOf26[n];	
	for (uint8_t k = 0; k < n; k++){
		rotorOffset[k] = r % powerOf26[n - k] / powerOf26[n - k - 1] ;
	}
	*chosenRotors = chosenMemory + keyNumber / powerOf26[n] * n;
}								

__device__ void intToKeyDev(uint64_t keyNumber, uint8_t n, 
								uint8_t** chosenRotors, uint8_t* rotorOffset) {
	
	unsigned int r = keyNumber % cPowerOf26[n];	
	for (uint8_t k = 0; k < n; k++){
		rotorOffset[k] = r % cPowerOf26[n - k] / cPowerOf26[n - k - 1] ;
	}
	*chosenRotors = cChosenMemory + keyNumber / cPowerOf26[n];
}								



__host__ void precomputationIntToKey(uint8_t* chosenRotorsMemory, int n, int N){
	uint8_t* perm = (uint8_t*) malloc(n * sizeof(uint8_t));
	for (int i = 0; i < n; i++) perm[i] = 0xff;
	uint8_t* used = (uint8_t*) calloc(N, sizeof(uint8_t));
	int i, j = 0, k = 0;
	while (k >= 0){
		for (i = (perm[k] == 0xff ? 0 : perm[k] + 1); i < N; i++){
			if (used[i]) continue;
			if (perm[k] != 0xff) used[perm[k]] = 0;
			perm[k] = i;
			used[i] = 1;
			k++;
			break;
		}
		if (k == n) {
			memcpy(chosenRotorsMemory + j * n, perm, n * sizeof(uint8_t));
			j++;
			k--;
		}
		else if (i == N) {
			used[perm[k]] = 0;
			perm[k] = 0xff;
			k--;
		}
	}
	free(perm);
	free(used);
}


__host__ void printKey(uint64_t key, uint8_t n) {
	uint8_t offset[MAX_ROTORS];
	uint8_t *rotors;
	intToKeyHost(key, n, &rotors, offset);
	printf("Key %lld: ", key);
	for (int i = 0; i < n; i++) printf("%d ", (int) rotors[i]);
	printf("|| ");
	for (int i = 0; i < n; i++) printf("%d ", (int) offset[i]);
	printf("\n");
}
	

__host__ uint64_t keyToInt(uint8_t* chosenRotors, uint8_t* rotorOffset, uint8_t n,
							uint8_t N){
	uint64_t r = 0;
	for (uint8_t k = 0; k < n; k++){
		r += rotorOffset[k];
		r *= 26;
	}
	
	
	//use precomputation here!!!
	for (uint8_t k = n - 1; k >= 0; k++){
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
								const uint8_t* devCipherText, float* IC,
								uint8_t n, uint8_t N){
	extern __shared__ uint8_t s[];
	uint8_t* chosenRotors = s;
	uint8_t* rotorOffset = s + n;
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t keyNumber = i + keyIndexOffset;
	
	/* first, we compute the key parameters (i.e. chosen rotors,
	 * and the associated offsets) from the key number
	 */
	 //intToKey(keyNumber, chosenRotors, rotorOffset);

}