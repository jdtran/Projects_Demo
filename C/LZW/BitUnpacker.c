#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "BitUnpacker.h"

void BuInit(BitUnpacker *bup) {   
   bup->curData = 0;
   bup->nextData = 0;
   bup->bitsLeft = 0;
   bup->validNext = 0;
}

void BuTakeData(BitUnpacker *bup, UInt data) {
   bup->nextData = data;
   bup->validNext = 1;
}

/*BuUnpack
 * return 1 on successful, 0 for needing next block
 */

int BuUnpack(BitUnpacker *bup, int size, UInt *ret) {
   UInt availBits, pulled, nextBits;
   char success, nextSize, leftover, newSize;
 
   success = 1;
   nextSize = 0;
   availBits = -1;

   if (!bup->validNext && (size > bup->bitsLeft)) {
      success = 0;
   }

   else {
      if (!bup->bitsLeft) {
         bup->curData = bup->nextData;
         bup->nextData = 0;
         bup->bitsLeft = UINT_SIZE;
         bup->validNext = 0;

         availBits = bup->curData;
      }
      else {
         availBits = bup->curData & ((1 << bup->bitsLeft) - 1);
      }
 
      leftover = bup->bitsLeft;
      nextBits = size - leftover;
      newSize = UINT_SIZE - nextBits;
 
      if (leftover == size) {
         *ret = availBits;
      }
      else if (size < leftover) {
         *ret = (availBits >> (leftover - size)) & ((1 << size) - 1);
      }
      else {
         pulled = bup->nextData &
          ((1 << nextBits) - 1) << newSize;
         *ret = availBits << nextBits | pulled >> newSize;
         bup->curData = bup->nextData;
         nextSize = 1;
      }
 
      if (nextSize) {
         bup->bitsLeft = newSize;
         bup->validNext = 0;
      }
      else {
         bup->bitsLeft -= size;
      }
   }
 
   return success;
}