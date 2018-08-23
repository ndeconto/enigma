#include "enigma.h"

/*
 * IC: 
 * n: number of rotors in the enigma machine
 * N: size of the rotor set 
 */
__global__ void decrypt_kernel(const char* devCipherText, float* IC, int n, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;


}