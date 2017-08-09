/**
 * @author John Tran (jtran57)
 * @version 5 December 2013
 *
 */

import java.util.*;

public class Queue<Comparable> implements Iterable<Comparable> 
{
    private int num;    
    private Node<Comparable> first;
    private Node<Comparable> last;

    public Queue() 
    {
        first = null;
        last  = null;
        num = 0;
    }
    
    public int size() 
    {
        return num;     
    }
    
    public boolean isEmpty() 
    {
       if(first == null)
       {
          return true;
       }
       
       else return false;
    }

    public void enqueue(Comparable item) 
    {
        Node<Comparable> prev = last;
        last = new Node<Comparable>();
        last.item = item;
        last.next = null;
        
        if (isEmpty()) 
        {
           first = last;
        }
        else
        {
           prev.next = last;
        }
        
        num++;
    }

    public Comparable dequeue() 
    {
        if (isEmpty()) 
        {
           throw new NoSuchElementException("Queue underflow");
        }
        
        Comparable item = first.item;
        first = first.next;

        
        if (isEmpty()) 
        {
           last = null;
        }
        
        num--;
        
        return item;
    }
    
    public Iterator<Comparable> iterator()  
    {
        return new ListIterator<Comparable>(first);  
    }

    private class ListIterator<Item> implements Iterator<Item>
    {
        private Node<Item> current;

        public ListIterator(Node<Item> first) 
        {
            current = first;
        }
        public boolean hasNext()  
        {
           return current != null;
        }

        public Item next() 
        {
            if (!hasNext())
            {
               throw new NoSuchElementException();
            }
            else
            {
               Item item = current.item;
               current = current.next; 
               return item;
            }
        }
        
        public void remove()      
        {}
    }
    
    private static class Node<Item> 
    {
        private Item item;
        private Node<Item> next;
    }
}