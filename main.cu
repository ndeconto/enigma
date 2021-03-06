#include <time.h>
#include <deque>

#include "preprocessing.h"
#include "enigma.h"


int main(int argc, char* argv[]){
	if (argc != 3){
		printf("Wrong number of arguments\n");
		printf("cipher text must be in stdin\n");
		printf("Usage: enigma number_of_rotors size_of_rotorset < textfile_to_decrypt\n");
		printf("Example: enigma 3 4 < ciphertext.txt\n");
		return 0;
	}
	
	cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return FAILURE;
    }
	
	uint8_t* cipherText, *clearText;
	uint8_t *devCipherText, *devClearText;
	float *IC, *devIC;
	int cipherLength = loadInput(&cipherText);
	clearText = (uint8_t*) malloc(sizeof(uint8_t) * cipherLength);
	printf("Cipher text is: \n");
	printText(cipherText, cipherLength);
	printf("Cipher length after preprocessing: %d\n", cipherLength);
	
	printf("\n===== ROTOR PART =====\n");
	fflush(stdout);
	int nbRotors = strtol(argv[1], NULL, 10); 
	int rotorSetSize = strtol(argv[2], NULL, 10);
	if (nbRotors > rotorSetSize){
		printf("Wrong arguments: number of rotors must be less or equal"
				"than the size of the rotor set\n");
		return 0;
	}
	if (nbRotors > MAX_ROTORS) {
		printf("Error: Too many rotors\n"
				"Maximum number of rotors is %d\n", nbRotors);
		return 1;
	}
	if (rotorSetSize > MAX_ROTOR_SET_SIZE) {
		printf("Error: rotor set too large\n"
				"Maximum set size is %d", MAX_ROTOR_SET_SIZE);
		return 1;
	}
	
	/* we will compute the index of coincidence for each possible key */
	//TODO: use managed memory
	IC = (float*) malloc(MAX_DIM_GRID * BLOCK_SIZE * sizeof(float));
	/* in order to avoid branching in kernel, we need a bit more than cipherLength bytes */
	if (cudaMalloc((void**) &devCipherText, cipherLength * sizeof(uint8_t)) != cudaSuccess
		|| cudaMalloc((void**) &devClearText, cipherLength * sizeof(uint8_t)) != cudaSuccess 
		|| cudaMalloc((void**) &devIC, MAX_DIM_GRID * BLOCK_SIZE * sizeof(float)) != cudaSuccess) {
			printf("cudaMalloc failed! "
				"Maybe MAX_CIPHER_LENGTH or MAX_DIM_GRID is too large?\n");
		return FAILURE;
    }
	cudaMemcpy(devCipherText, cipherText, cipherLength, cudaMemcpyHostToDevice);
	
	/* precomputation to better handle key indexes */
	//TODO make sure a memory alignement problem is not decreasing performance
	int precomputationSize = (factorial(rotorSetSize) 
							/ factorial(rotorSetSize - nbRotors));
	chosenMemory = (uint8_t*)malloc(precomputationSize * nbRotors * sizeof(uint8_t));
	precomputationIntToKey(chosenMemory, nbRotors, rotorSetSize);
	//test
	//for (int i = 0; i < precomputationSize; i++) {for(int j = 0; j < nbRotors; j++) printf("%d ", chosenMemory[i * nbRotors + j]); printf("\n");}
	cudaMemcpyToSymbol(cChosenMemory, chosenMemory, 
							precomputationSize * nbRotors * sizeof(uint8_t));
	cudaMemcpyToSymbol(cPowerOf26, powerOf26, (nbRotors + 1) * sizeof(unsigned int));
	initRotors<<<1, 1>>>();
	
	uint64_t possibleKeys = NB_OF_REFLEC * (factorial(rotorSetSize)
								/ factorial(rotorSetSize - nbRotors) 
								*  pow(26, nbRotors));
	printf("There are %lld possible keys... "
			"Trying to find the good one...\n", possibleKeys);
	//for (int i = 0; i < possibleKeys; i++) printKey(i, nbRotors);
	if (DEBUG) printKey(DEBUG_ID, nbRotors);
	int nbSteps = ceil(possibleKeys / ((float) BLOCK_SIZE) / MAX_DIM_GRID);
	for (int i = 0; i < nbSteps; i++){
		int dimGrid = min((int) MAX_DIM_GRID, (int) (possibleKeys - i * KEYS_PER_STEP) / BLOCK_SIZE + 1);
		printf("Step %d out of %d: dimGrid = %d blocks of %d threads\n", 
					i + 1, nbSteps, dimGrid, BLOCK_SIZE);
		clock_t start_t = clock();
		decryptKernel<<<dimGrid, BLOCK_SIZE, cipherLength * sizeof(uint8_t)>>>
			(i * KEYS_PER_STEP, possibleKeys, cipherLength, devCipherText, devIC, nbRotors);
		cudaMemcpy(IC, devIC, dimGrid * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		clock_t end_t = clock();
		printf("Time elapsed: %.3f seconds\n", (float) (end_t - start_t) / CLOCKS_PER_SEC); 
		printf("Last error: %s\n", cudaGetErrorName(cudaGetLastError()));
		
		std::deque<uint64_t> possibleKeys;
		for (int j = 0; j < dimGrid * BLOCK_SIZE; j++){
			if (IC[j] > DETECTION_THRESHOLD){
				printf("Possible key has been found! (IC = %.3f)\n\t", IC[j]);
				uint64_t key = j + i * BLOCK_SIZE * MAX_DIM_GRID;				
				printKey(key, nbRotors);
				possibleKeys.push_back(key);
				enigmaCipher<<<1, 1>>>(devCipherText, devClearText, cipherLength, key, nbRotors);
				cudaMemcpy(clearText, devClearText, cipherLength * sizeof(uint8_t), cudaMemcpyDeviceToHost);
				printf("Corresponding text is:\n");
				printText(clearText, cipherLength);
			}
		}
	}
	
	cudaFree(devCipherText);
	cudaFree(devClearText);
	free(chosenMemory);
	free(cipherText);
	free(clearText);
	
	return 0;
}