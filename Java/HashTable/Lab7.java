/** 
 * A driver for a simple Hash Table-based Spell Checker program.
 *
 * DO NOT MODIFY THIS CLASS.
 *
 *    Examine this class to see what public methods are necessary in HashTable.
 *    You may write any other private methods in HashTable that you wish to support
 *    your implementation.
 *
 * @author Kurt Mammen, modified by Chris Buckalew
 *
 * @version CPE103 - Lab 7CB
 */
 
import java.util.Scanner;
import java.io.File;

public class Lab7
{
   public static void main(String[] args) throws java.io.FileNotFoundException
   {
      // Prompt for hash table size...
      Scanner scan = new Scanner(System.in);
      System.out.print("Enter hash table size: ");
      int size = scan.nextInt();
      scan.nextLine();
      
      // Create the hash table.
      // FYI: There are 234937 words in the dictionay - not necessarily the
      //      best table size - think about it, try different ideas striving for
      //      low maximum and average bucket counts while not wasting too much
      //      memory - a balancing act!
      HashTable table = new HashTable(size);
      
      // Populate the table with the words form the dictionary...
      Scanner scanner = new Scanner(new File("dictWords.txt"));

      while (scanner.hasNext())
      {
         table.add(scanner.next().trim());
      }
      
      // Report Hash Table statistics
      System.out.println("Number of words: " + table.numElements());
      System.out.println("Table capacity: " + table.capacity());
      System.out.println("Maximum bucket count: " + table.maxBucketCount());
      System.out.println("Number of non-zero buckets: " + table.nonZeroBucketCount());
      System.out.println("Average non-zero bucket size: " + table.avgNonZeroBucketCount());

      // Spell check some words to see if your table is working...      
      while (true)
      {
         System.out.print("\nEnter word to spell check or press <enter> to quit: ");
         
         String word = scan.nextLine();
         
         if (word.equals(""))
         {
            break;
         }
         
         boolean contains = table.contains(word);
         
         if (contains)
         {
            System.out.println(word + " is spelled correctly!");
         }
         else
         {
            System.out.println(word + " is not in the dictionary!");
         }
      }
   }
}
