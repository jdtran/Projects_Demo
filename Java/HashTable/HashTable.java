/** 
 * Implementation of a hash table.
 * Lab 7 CPE-103, Dr. Chris Buckalew
 * @author John Tran
 * @version 12/3/2013
 */

import java.lang.Math;

public class HashTable{
   LinkedList table[];
   int capacity;
   
   public HashTable(int capacity){
      table = new LinkedList[capacity]; 
      for(int i=0;i<capacity;i++){
         table[i] = new LinkedList();
      } 
      this.capacity = capacity;
   }
   
   public void add(Object element){
      table[makeHash(element)].addLast(element);
   }
   
   public boolean contains(Object element){
      return table[makeHash(element)].contains(element);
   }
   
   public int numElements(){
      int sum = 0;
      for(int i=0; i<capacity; i++)
         sum += table[i].length();
      return sum;
   }
   
   public int capacity(){
      return capacity;
   }
   
   public int maxBucketCount(){
      int max = 0;
      for(int i=0;i<capacity;i++){
         int size = table[i].length();
         if(size > max)
            max = size;
       }
       return max;
   }
   
   public int nonZeroBucketCount(){
      int count = 0;
      for(int i=0;i<capacity;i++){
         int size = table[i].length();
         if(size > 0){
            count ++;
         }
      }
      return count;
   }
   
   public float avgNonZeroBucketCount(){
       int count = 0,sum = 0;
      for(int i=0;i<capacity;i++){
         int size = table[i].length();
         if(size > 0){
            sum += size;
            count ++;
         }
      }
      return ((float)sum) / count; 
   }  
   
   public static class Error extends RuntimeException{
      public Error(String s){
         super(s);
      }
   }
   
   private int makeHash(Object element){
      int hash = element.hashCode();
      hash = hash % capacity;
      return Math.abs(hash);
   }
}