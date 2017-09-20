#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include "LZWExp.h"
#include "CodeSet.h"
#include "MyLib.h"
#include "SmartAlloc.h"

void LZWPrintDump(void *voidStar, unsigned char *data, int numBytes) {
   while (numBytes--) {
      printf("%c", *data++);
   }
}

void LZWInit(LZWExp *exp, DataSink sink, void *sinkState, int recycleCode) {
   LZWExpInit(exp, sink, sinkState, recycleCode);
}

void LZWSend(LZWExp *exp) {
   UInt intake, success;

   success = 0;

   while (scanf(" %X", &intake) != EOF && !success) {
      success = LZWExpDecode(exp, intake);
   }
   
   if (success == BAD_CODE) {
      printf("Bad code\n");
   }
   else if (LZWExpStop(exp) == MISSING_EOD) {
      printf("Missing EOD\n");
   }

   LZWExpDestruct(exp);
}

void Execute(int argc, char **argv) {
   int recycleCode;
   char *garbo;
   LZWExp *exp;
   DataSink sink;

   if (argc > 2 && !(strcmp(argv[1], "-R"))) {
      recycleCode = strtol(argv[2], &garbo, DECIMAL);
   }
   else {
      recycleCode = DEFAULT_RECYCLE_CODE;
   }

   sink = LZWPrintDump;
   exp = malloc(sizeof(LZWExp));
   LZWInit(exp, sink, NULL, recycleCode);
   LZWSend(exp);
   free(exp);
}

int main(int argc, char **argv) {
   Execute(argc, argv);

   if (report_space()) {
      printf("Allocated space: %lu\n", report_space());
   }

   return 0;
}
