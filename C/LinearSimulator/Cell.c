#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "Report.h"

#define MAX_STRING 100
#define MAX_FD 12
#define DECIMAL 10
#define ARGS_ALLOWED 8
#define MAX_OUT 3
#define MAX_IN 2
#define TRUE 1
#define FALSE 0

void ReportMe(Report *myself, int *fdsOut, int numOut) {
   int ndx;
   
   ndx = 0;
   while (ndx < numOut) {
      write(fdsOut[ndx], myself, sizeof(Report));

      ndx++;
   }
}

double ReadNeighbors(int *fdsIn, int numIn, int currentStep) {
   int found, ndx;
   double sum, average;
   Report left, right;
   
   ndx = 0;

   read(fdsIn[0], &left, sizeof(Report));
   
   if (left.step == currentStep) {
      found = TRUE;
   }  
   else {
      exit(EXIT_FAILURE);
   }
   
   found = FALSE;
   
   read(fdsIn[1], &right, sizeof(Report));
      
   if (right.step == currentStep) {
      found = TRUE;
   }
   else {
      exit(EXIT_FAILURE);
   }
   
   sum = left.value + right.value;
   average = sum / 2;
   
   return average;
}

int ParseArgs(char *arg, Report *myself, 
 int *fdsIn, int *fdsOut, int *numIn, int *numOut) {
   int value, len, failed;
   char command[MAX_STRING], *number;
   double outerValue;
   
   len = strlen(arg);
   strcpy(command, arg);
   number = command + 1;
   outerValue = 0;
   failed = FALSE;

   value = strtol(number, NULL, DECIMAL);

   switch (command[0]) {
   case 'S':
      myself->step = value;
      break;
      
   case 'I':
      if (value > MAX_FD) {
         fprintf(stderr, "Error: in-fd %d is greater allowed maximum %d\n", 
          value, MAX_FD);
         failed = TRUE;
      }
      fdsIn[*numIn] = value;
      (*numIn)++;
      break;
      
   case 'O':
      if (value > MAX_FD) {
         fprintf(stderr, "Error: out-fd %d is greater allowd maximum%d\n", 
          value, MAX_FD);
         failed = TRUE;
      }
      fdsOut[*numOut] = value;
      (*numOut)++;
      break;
      
   case 'V':
      outerValue = strtod(number, NULL);
      myself->value = outerValue;
      break;
      
   case 'D':
      myself->id = value;
      break;
      
   default:
      failed = FALSE;
   }
   
   return failed;
}

void Cycle(int argc, char **argv) {
   int ndx, fdsIn[MAX_IN], fdsOut[MAX_OUT], numIn, numOut, fail;
   short period, maxStep;
   Report myself = {0, 0, 0};

   ndx = 0;
   numIn = 0;
   numOut = 0;
   fail = FALSE;
   
   while (++ndx < argc) {
      fail |= ParseArgs(argv[ndx], &myself, fdsIn, fdsOut, &numIn, &numOut);
   }
   
   if (fail) {
      exit(EXIT_FAILURE);
   }
   
   maxStep = myself.step;
   myself.step = 0;
   ReportMe(&myself, fdsOut, numOut);
   period = 1;
   myself.step = period;
   
   if (numIn == 0) { /*outermost cell */
      while (period <= maxStep) {
         ReportMe(&myself, fdsOut, numOut);
         period++;
         myself.step = period;
      }
   }
   else { /*inner left/right 2out 2in and innermost 3out 2in*/
      while (period <= maxStep) {
         myself.value = ReadNeighbors(fdsIn, numIn, (period - 1));
         ReportMe(&myself, fdsOut, numOut);
         period++;
         myself.step = period;
      }
      ReadNeighbors(fdsIn, numIn, (period - 1));
   }

   ndx = 0;
   
   while (ndx < numIn) {
      close(fdsIn[ndx]);
      ndx++;
   }

   ndx = 0;
   
   while (ndx < numOut) {
      close(fdsOut[ndx]);
      ndx++;
   }
}

int main(int argc, char **argv) { 
   Cycle(argc, argv);
   exit(EXIT_SUCCESS);
   return 0;
}