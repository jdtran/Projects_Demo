#include <math.h>
#include <limits.h>
#include "LZWExp.h"
#include "MyLib.h"
#include "CodeSet.h"
/* Function pointer to method to call when a block of bytes is
 * expanded and ready for transmission or whatever.  The void * parameter
 * can point to anything, and gives hidden information to the function 
 * so that it knows what file, socket, etc. the code is going to.  "Data"
 * points to the block of expanded bytes, with numBytes bytes in the block.
 * The block is not a C string -- it may have null characters in it, and
 * is not null-terminated.
 * 
 * A call with null "data" pointer marks the end of data, and will be the
 * final call.
*/

 
void LZWExpInit(LZWExp *exp, DataSink sink, void *sinkState, int recCode) {
   UInt bits, dictSize, base;
   int iMaxCode, rec;
   void *dictionary;
   BitUnpacker bitUnpacker = {0, 0, 0, 0};

   dictSize = 0;
   base = INITIAL_DICT + 1;
   bits = 1;
  
   while (base >>= 1) {
      bits++;
   }
   
   iMaxCode = Exp2(1, bits);

   if (recCode != DEFAULT_RECYCLE_CODE) {
      rec = recCode;
   }
   else {
      rec = DEFAULT_RECYCLE_CODE;
   }

   dictionary = CreateCodeSet(rec + 1);

   while (dictSize <= INITIAL_DICT) {
      NewCode(dictionary, dictSize);
      dictSize++;
   }
   
   NewCode(dictionary, 0);
 
   exp->dict = dictionary;
   exp->sink = sink;
   exp->sinkState = sinkState; 
   exp->lastCode = -1;
   exp->numBits = bits;
   exp->maxCode = iMaxCode;
   exp->recycleCode = rec;
   exp->bitUnpacker = bitUnpacker;
   exp->EODSeen = 0;
}

/* Break apart compressed data in "bits" into one or more codes and send 
 * the corresponding symbol sequences to the DataSink.  Save any leftover 
 * compressed bits to combine with the bits from the next call of 
 * LZWExpDecode.  
 *
 * Return 0 on success or BAD_CODE if you receive a code that could not
 * have been sent (e.g. is too high) or if it is a nonzero code following
 * the detection of an EOD code, or on any extra LZWExpDecode calls that
 * come after EOD has already been hit.*/
int LZWExpDecode(LZWExp *exp, UInt bits) {
   BitUnpacker *bup;
   char success;
   UInt ret, lastAdded, base;
   Code received;

   success = 0;

   BuTakeData(&(exp->bitUnpacker), bits);
   if (exp->EODSeen) {
      success = BAD_CODE;
   }

   lastAdded = 0;
  
   while (!success && BuUnpack(&(exp->bitUnpacker), exp->numBits, &ret)) {
      if (ret > exp->lastCode) {
         success = BAD_CODE;
      }
      else if (ret == INITIAL_DICT + 1 && exp->EODSeen) {
         success = BAD_CODE;
      }
      else if (exp->EODSeen && ret != 0) {
         success = BAD_CODE;
      }
      else if (ret == INITIAL_DICT + 1 || (exp->EODSeen && !ret)) {
         exp->EODSeen = 1;
      }
      else {
         received = GetCode(exp->dict, ret);

         if (exp->EODSeen && ret) {
            success = BAD_CODE;
         }
         else {
            if (exp->lastCode != -1) {
               SetSuffix(exp->dict, exp->lastCode, *(received.data));
            }
            exp->sink(exp->sinkState, received.data, received.size);

            if ((exp->lastCode + 1) == exp->recycleCode) {
               DestroyCodeSet(exp->dict);
               bup = &(exp->bitUnpacker);
               LZWExpInit(exp, exp->sink, exp->sinkState, exp->recycleCode);
               exp->bitUnpacker = *bup;
            }
            else {
               lastAdded = ExtendCode(exp->dict, ret);
               exp->lastCode = lastAdded;
               exp->numBits = 1;
               base = lastAdded;
            
               while (base >>= 1) {
                  exp->numBits++;
               }
            
               exp->maxCode = Exp2(1, exp->numBits);
            }
         }

         FreeCode(exp->dict, ret);
      }
   }

   return success;
}

/* Called by main or other driver when no more LZWExpDecode calls are expected
 * e.g. when EOF is reached on the compressed data.  This triggers
 * housekeeping that should be performed at the end of decoding. Returns 0 if
 * all is OK, or NO_EOD if no terminating EOD code has been found. */
int LZWExpStop(LZWExp *exp) {
   char success;
   UInt mask, leftover;

   success = 0;

   if (!exp->EODSeen) {
      success = MISSING_EOD;
   }
   else if (exp->bitUnpacker.bitsLeft) {
      mask = (1 << (exp->bitUnpacker.bitsLeft)) - 1;
      leftover = mask & exp->bitUnpacker.curData;

      if (leftover != 0) {
         printf("Bad code\n");
      }
   }

   return success;
}

/* Free all storage associated with LZWExp (not the sinkState, though,
 * which is "owned" by the caller.  Must be called even if LZWExpInit
 * returned an error.  */
void LZWExpDestruct(LZWExp *exp) {
   DestroyCodeSet(exp->dict);
}