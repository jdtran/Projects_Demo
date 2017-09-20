#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "Report.h"

#define MAX_STRING 100
#define MAX_CELL 50
#define MAX_OUT 7
#define MAX_IN 9
#define ARGS_ALLOWED 4

#define USAGE -1
#define CHILD_EXIT -2
#define FEW_REPORT -3
#define MAN_REPORT -4

#define TRUE 1
#define FALSE 0
#define DECIMAL 10

void PrintError(int type, int child, int exitStat) {
   switch (type) {
   case USAGE:
      fprintf(stderr, "Usage: LinearSim C S L R (in any order)\n");
      break;
      
   case CHILD_EXIT:
      fprintf(stderr, "Error: Child %d exited with %d\n", child, exitStat);
      break;
      
   case FEW_REPORT:
      fprintf(stderr, "Error: %d cells reported too few reports\n");
      
   case MAN_REPORT:
      fprintf(stderr, "Error: %d cells reported too many reports\n");

      break;
      
   default :
      exit(EXIT_FAILURE);
   }
   
   exit(EXIT_FAILURE);
}

void RightChild(int fdsCommonWrite, int outInter, Report left, Report right) {
   int ndx;
   char *args[MAX_OUT];

   ndx = 0;
   args[0] = "./Cell";
   
   while (++ndx < MAX_OUT) {
      args[ndx] = malloc(MAX_STRING);
   }
   
   sprintf(args[1], "S%d", left.step);
   sprintf(args[2], "D%d", right.id);
   sprintf(args[3], "V%lf", right.value);
      
   if (right.id == 1) { //Only 2 Children Case
      sprintf(args[4], "O%d", fdsCommonWrite);
      args[5] = NULL;
   }
   else {
      sprintf(args[4], "O%d", fdsCommonWrite);
      sprintf(args[5], "O%d", outInter);
      args[6] = NULL;
   }
      
   printf("R working..\n");
   execvp(args[0], args);
   fprintf(stderr, "Error executing process R\n");
}

//Inner children
void InnerChildren (int innerId, int fdsCommonWrite, int inLeft, int inRight, int outLeft, int outRight, Report left, Report right) {
   int ndx;
   short whichOne;
   char *args[MAX_IN];

   ndx = 0;
   whichOne = 0;
   args[0] = "./Cell";
   
   while (++ndx < MAX_IN) {
      args[ndx] = malloc(MAX_STRING);
   }
   
   sprintf(args[1], "S%d", left.step);
   sprintf(args[2], "D%d", innerId);
   sprintf(args[3], "O%d", fdsCommonWrite); // Write To Outpipe
   sprintf(args[4], "I%d", inLeft); // Read From Left In Pipe

   if (right.id == 2) { //3 Children Case
      sprintf(args[5], "I%d", inRight);
      args[6] = NULL;
      whichOne = 0;      
   }
   else if (right.id > 2) { 
      if (innerId == 1 || innerId == (right.id - 1)) { //2 Outs 2 Ins
         sprintf(args[5], "O%d", fdsCommonWrite); // Write To Outpipe
         sprintf(args[6], "I%d", inLeft); // Read From Left In Pipe
         args[7] = NULL;
      }
      else { //3 outs 2 ins
         sprintf(args[5], "O%d", fdsCommonWrite); // Write To Outpipe
         sprintf(args[6], "O%d", fdsCommonWrite); // Write To Outpipe
         sprintf(args[7], "I%d", inLeft); // Read From Left In Pipe
         args[8] = NULL;
      }
   }
   
   /*if (innerId == 1) {      
      sprintf(args[6], "I%d", inRight);
      args[7] = NULL;
      whichOne = 0;
   }*/
   
   printf("I working..\n");
   execvp(args[0], args);
   fprintf(stderr, "Error executing process I\n");
}

void LeftChild(int fdsCommonWrite, int outInter, Report left, Report right) {
   int ndx;
   char *args[MAX_OUT];

   ndx = 0;
   args[0] = "./Cell";
   
   while (++ndx < MAX_OUT) {
      args[ndx] = malloc(MAX_STRING);
   }
   
   sprintf(args[1], "S%d", left.step);
   sprintf(args[2], "D%d", left.id);
   sprintf(args[3], "V%lf", left.value);
      
   if (right.id == 1) { //2 Children Case
      close(outInter);
      sprintf(args[4], "O%d", fdsCommonWrite);
      args[5] = NULL;
   }
   else {
      sprintf(args[4], "O%d", fdsCommonWrite);
      sprintf(args[5], "O%d", outInter);
      args[6] = NULL;
   }

   printf("L working..\n");
   execvp(args[0], args);
   fprintf(stderr, "Error executing process L\n");
}

//set Parent
void ParentProcess(Report left, Report right) {
   int commonPipe[2], pidLeft, pidRight, pidMid;
   int lfReadInterPipe[2], lfWriteInterPipe[2];
   int rtReadInterPipe[2], rtWriteInterPipe[2];
   int innerId;
   char str[100];
   Report readReport;
   
   pipe(commonPipe);
   pipe(rtWriteInterPipe);
   innerId = 1;
   //printf("RT PIPE %d %d\n", rtWriteInterPipe[0], rtWriteInterPipe[1]);

   if ((pidLeft = fork()) == 0) { //Make Left Child
      close(commonPipe[0]);
      close(rtWriteInterPipe[0]);
      LeftChild(commonPipe[1], rtWriteInterPipe[1], left, right);
   }
   else {
      pipe(lfWriteInterPipe);
      close(rtWriteInterPipe[1]);
            
      //printf("LF PIPE %d %d\n", lfWriteInterPipe[0], lfWriteInterPipe[1]);
      
      if ((right.id > 1) && (pidRight = fork()) == 0) {  //Make Inner Child
         close(commonPipe[0]);
         close(rtWriteInterPipe[1]);
         close(lfWriteInterPipe[1]);
         
         /*printf("LeftCell ID %d\n", left.id);
         printf("LeftCell STEP %d\n", left.step);
         printf("LeftCell VALUE %lf\n", left.value);

         printf("RightCell ID %d\n", right.id);
         printf("RightCell STEP %d\n", right.step);
         printf("RightCell VALUE %lf\n", right.value);*/
   
         InnerChildren(1, commonPipe[1], rtWriteInterPipe[0], lfWriteInterPipe[0], left, right);
      }
      else {
         close(rtWriteInterPipe[0]);
         close(lfWriteInterPipe[0]);

         if ((right.id == 1) && !(pidMid = fork())) { //2 Child Case RIGHT CHILD
            close(lfWriteInterPipe[1]);
            close(commonPipe[0]);
            RightChild(commonPipe[1], 0,left, right);
         }
         else if ((right.id > 1) && !(pidMid = fork())) { //RIGHT CHILD
            close(commonPipe[0]);
            RightChild(commonPipe[1], lfWriteInterPipe[1],left, right);
         }
         else {
            close(lfWriteInterPipe[1]);
            wait(NULL);
         }
         
         wait(NULL);
      }
      
      wait(NULL);
      close(commonPipe[1]);
      
      while (0 < read(commonPipe[0], &readReport, sizeof(Report))) {
         printf("Result from %d, step %d: %.3lf\n", readReport.id, readReport.step, readReport.value);
      }

      printf("Done Reading.\n");
      close(commonPipe[0]);
   }
}

short *ParseArgs(char *arg, Report *leftCell, Report *rightCell) {
   int value, len, step, children;
   char command[MAX_STRING], *number;
   double outerValue;
    
   len = strlen(arg);
   strcpy(command, arg);
   number = command + 1;
   outerValue = 0;
   value = strtol(number, NULL, DECIMAL);

   switch (*command) {
   case 'C':
      if (value <= 1 || value > MAX_CELL) {
         PrintError(USAGE, 0, 0);
      }
      children = value;
      leftCell->id = 0;
      rightCell->id = (value - 1);
      break;
      
   case 'S':
      leftCell->step = value;
      rightCell->step = value;
      break;
      
   case 'L':
      outerValue = strtod(number, NULL);
      leftCell->value = outerValue;
      break;
   
   case 'R':
      outerValue = strtod(number, NULL);
      rightCell->value = outerValue;
      break;
   }
   
   /*if (!outerValue) {
      printf("|%c %d|\n", *command, value);
   }
   else {
      printf("|%c %lf|\n", *command, outerValue);
   }*/
}

void ReadArgs(int argc, char **argv) {
   int fixedEnd, ndx;
   Report leftCell, rightCell;
   ndx = 0;
   
   while (++ndx < argc) {
      ParseArgs(argv[ndx], &leftCell, &rightCell);
   }

   ParentProcess(leftCell, rightCell);
}

int main(int argc, char **argv) {
   ReadArgs(argc, argv);
   
   return 0;
}