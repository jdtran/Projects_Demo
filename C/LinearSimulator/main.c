#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "Report.h"

#define MAX_STRING 100
#define DECIMAL 10
#define ARGS_ALLOWED 7

/*typedef struct {
   int id;       // ID of the Cell reporting
   int step;     // Time for which report is being made
   double value; // Value of the cell at time "step"
} Report;*/

int ExitReturn(int success) {
   return success;
}

void ReportMe() {

}

void NeighborReport() {

}

int ParseArgs(char *arg, Report *myself, Report *others) {
   int value, len, ndx;
   char command[MAX_STRING], *number;

   number = NULL;
   len = strlen(arg);
   strcpy(command, arg);
   ndx = 0;
   
   while (ndx < len) {
      if (isdigit(arg[ndx])) {
         number = arg + ndx;
         command[ndx] = 0;
         len = strlen(number);
         ndx = 0;

         while(number[ndx]) {
            if (!isdigit(number[ndx])) {
               return EXIT_FAILURE;
            }
            ndx++;
         }
         break;
      }
      ndx++;
   }
   
   if (!number || strlen(command) != 1) {
      return EXIT_FAILURE;
   }
   
   value = strtol(number, NULL, DECIMAL);


   switch (*command) {
   case 'S':
      printf("S %d\n", value);
      break;
      
   case 'O':
      printf("O %d\n", value);
      break;

   case 'I':
      printf("I %d\n", value);
      break;

   case 'V':
      printf("V %d\n", value);
      break;

   case 'D':
      printf("D %d\n", value);
      break;

   default :
      return EXIT_FAILURE;
   }
}

void Cycle(int argc, char **argv) {
   int ndx, success;
   Report myself, others;
   ndx = 0;
   success = EXIT_SUCCESS;

   while (++ndx < argc && success != EXIT_FAILURE) {
      success = ParseArgs(argv[ndx], &myself, &others);
   }
}

int main(int argc, char **argv) {
   int exitType = EXIT_SUCCESS;
   
   if (argc != ARGS_ALLOWED) {
      exitType = EXIT_FAILURE;
   }
   else {
      Cycle(argc, argv);
   }

   ExitReturn(exitType);

   return 0;
}