#include "preprocessing.h"

/* 
 * reads and preprocess stdin, stores the result in text
 * all non alphabetic characters are removed
 * ASCII is not used, but 0 for A, ... 25 for Z.
 * returns the length of preprocessed text
 */ 
int loadInput(uint8_t** text){
	*text = (uint8_t*) malloc(sizeof(uint8_t) * MAX_INPUT_SIZE);
	char line[MAX_INPUT_SIZE];
	int i, pos = 0;
	
	//TODO a debug car pas la doc de fgets sans internet
	while (fgets(line, MAX_INPUT_SIZE, stdin) != NULL){
		for (i = 0; i < MAX_INPUT_SIZE && line[i] != '\n' && line[i] != '\0'; i++){
			line[i] = toupper(line[i]);
			if (line[i] < 'A' || line[i] > 'Z') continue;
			(*text)[pos++] = line[i] - 'A';
		}
	}
	printf("\n[+] preprocessing done\n");
	return pos;
}

int factorial(int n){
	if (n <= 1) return 1;
	return n * factorial(n - 1);
}