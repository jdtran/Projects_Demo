#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "SmartAlloc.h"
#include "CodeSet.h"

typedef struct CodeEntry {
   unsigned char data;
   unsigned char *dataReference;
   unsigned short outstanding;
   unsigned short size;
   struct CodeEntry *next;
} CodeEntry;

typedef struct CodeSet {
   CodeEntry *codeEntryList;
   int size;
} CodeSet;

/* Allocate, initialize, and return a CodeSet object, via void *
 * The CodeSet will have room for |numCodes| codes, though it will
 * initially be empty. */

void *CreateCodeSet(int numCodes) {
   CodeSet *codeSet = malloc(sizeof(CodeSet));

   codeSet->codeEntryList = malloc(sizeof(CodeEntry) * numCodes);
   codeSet->size = 0;

   return codeSet; 
}

/* Add a new 1-byte code to |codeSet|, returning its index, with
 * the first added code having index 0.  The new code's byte is
 * equal to |val|.  Assume (and assert if needed) that there
 * is room in the |codeSet| for a new code. */
int NewCode(void *codeSet, char val) {
   CodeEntry newEntry;
   CodeSet *cpCS = codeSet;

   newEntry.data = (unsigned) val;
   newEntry.dataReference = NULL;
   newEntry.outstanding = 0;
   newEntry.size = 1;
   newEntry.next = NULL;

   (cpCS->codeEntryList)[cpCS->size] = newEntry; 

   return cpCS->size++;
}

/* Create a new code by copying the existing code at index
 * |oldCode| and extending it by one zero-valued byte.  Any
 * existing code might be extended, not just the most recently
 * added one. Return the new code's index.  Assume |oldCode|
 * is a valid index and that there is enough room for a new code.*/
int ExtendCode(void *codeSet, int oldCode) {
   CodeEntry *copyCE, extendedCE;
   CodeSet *cpCS = codeSet;

   copyCE = cpCS->codeEntryList + oldCode;
   extendedCE.data = 0;
   extendedCE.dataReference = NULL;
   extendedCE.outstanding = 0;
   extendedCE.size =  copyCE->size + 1;
   extendedCE.next = copyCE;
   *(cpCS->codeEntryList + cpCS->size) = extendedCE;

   return cpCS->size++;
}

/* Set the final byte of the code at index |code| to |suffix|.
 * This is used to override the zero-byte added by ExtendCode.
 * If the code in question has been returned by a GetCode call,
 * and not yet freed via FreeCode, then the changed final byte
 * will also show in the Code data that was returned from GetCode.*/
void SetSuffix(void *codeSet, int code, char suffix) {
   CodeEntry *startEntry, *lastEntry, *currentEntry;

   startEntry = ((CodeSet *)codeSet)->codeEntryList + code;
   lastEntry = startEntry;
   currentEntry = startEntry;

   while (currentEntry++->next == lastEntry) {
      lastEntry = currentEntry;
   }

   lastEntry->data = suffix;
   if (startEntry->outstanding) {
      *(startEntry->dataReference + startEntry->size - 1) = suffix; 
   }
}

/* Return the code at index |code| */
Code GetCode(void *codeSet, int code) {
   unsigned char *newData;
   int ndx;
   CodeEntry *head, *accessCode;
   Code retCode;

   accessCode = ((CodeSet *)codeSet)->codeEntryList + code;

   ndx = accessCode->size;

   if (accessCode->outstanding) {
      newData = accessCode->dataReference;
   }
   else {
      newData = calloc(accessCode->size, sizeof(char));
      head = accessCode;

      while (head && !head->outstanding) {
         *(newData + --ndx) = head->data;
         head = head->next;
      }
      if (head) {
         memcpy(newData, head->dataReference, head->size);
      }
      accessCode->dataReference = newData;
   }

   retCode.data = newData;
   retCode.size = accessCode->size;

   accessCode->outstanding++;

   return retCode;
}

/* Mark the code at index |code| as no longer needed, until a new
 * GetCode call is made for that code. */
void FreeCode(void *codeSet, int code) {
   CodeEntry *accessCode;

   accessCode = ((CodeSet *)codeSet)->codeEntryList + code;

   if (accessCode->outstanding == 1) {
      free(accessCode->dataReference);
      accessCode->outstanding = 0;
   }
   else if (accessCode->outstanding > 1) {
      accessCode->outstanding--;
   }
}

/* Free all dynamic storage associated with |codeSet| */
void DestroyCodeSet(void *codeSet) {
   CodeEntry *iPtr;
   CodeSet *cpCS = codeSet;

   iPtr = cpCS->codeEntryList;

   while (iPtr - cpCS->codeEntryList < cpCS->size) {
      if (iPtr->outstanding) {
         free(iPtr->dataReference); 
         iPtr->outstanding = 0;
      }
      iPtr++;
   }
   
   free(cpCS->codeEntryList);
   free(codeSet);
}