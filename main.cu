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
	
	char* cipherText;
	uint8_t *devCipherText;
	float *IC, *devIC;
	int cipherLength = loadInput(&cipherText);
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
	if (cudaMalloc((void**) &devCipherText, cipherLength * sizeof(uint8_t)) != cudaSuccess 
		|| cudaMalloc((void**) &devIC, MAX_DIM_GRID * BLOCK_SIZE * sizeof(float)) != cudaSuccess) {
			printf("cudaMalloc failed! "
				"Maybe MAX_CIPHER_LENGTH or MAX_DIM_GRID is too big?\n");
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
	for (int i = 0; i < precomputationSize; i++) {for(int j = 0; j < nbRotors; j++) printf("%d ", chosenMemory[i * nbRotors + j]); printf("\n");}
	cudaMemcpyToSymbol(cChosenMemory, chosenMemory, 
										precomputationSize * nbRotors);
	
	uint64_t possibleKeys = (factorial(rotorSetSize)
								/ factorial(rotorSetSize - nbRotors) 
								*  pow(26, nbRotors));
	printf("There are %lld possible keys... "
			"Trying to find the good one...\n", possibleKeys);
	for (int i = 0; i < possibleKeys; i++) printKey(i, nbRotors);
	int nbSteps = ceil(possibleKeys / ((float) BLOCK_SIZE) / MAX_DIM_GRID);
	for (int i = 0; i < nbSteps; i++){
		int dimGrid = min((int) MAX_DIM_GRID, (int) possibleKeys - i * BLOCK_SIZE * MAX_DIM_GRID);
		/*decrypt_kernel<<<dimGrid, BLOCK_SIZE, 2 * rotorSetSize * sizeof(char)>>>
				(i, devCipherText, devIC, nbRotors, rotorSetSize);*/
		cudaMemcpy(IC, devIC, dimGrid * BLOCK_SIZE, cudaMemcpyDeviceToHost);
		for (int j = 0; j < dimGrid * BLOCK_SIZE; j++){
			if (IC[j] > DETECTION_THRESHOLD){
				printf("Key has been found! Key id is %d\n", 
							j + i * BLOCK_SIZE * MAX_DIM_GRID);
				//TODO s'en servir pour décoder !!!
			}
		}
	}
	
	cudaFree(devCipherText);
	free(chosenMemory);
	free(cipherText);
	
	return 0;
}