#ifndef PREPROCESSING_HEADER
#define PREPROCESSING_HEADER

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

/* 
 * if the cipher text is too short, it's of course more difficult to decrypt it
 * but it's not easier to decrypt an infinitely long text than a mediu text
 * However, if the text is too long, it does not fit in fast GPU caches and 
 * performances decrease. So we limit the considered input 
*/
#define MAX_INPUT_SIZE 65536

#ifdef __cplusplus
extern "C" {
#endif

extern int loadInput(char** text);
extern int factorial(int n);

#ifdef __cplusplus
}
#endif

#endif