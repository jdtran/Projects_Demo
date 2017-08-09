/**
 * @author John Tran (jtran57)
 * @version 5 December 2013
 *
 */

import java.io.*;
import java.util.*;
import java.util.PriorityQueue;

public class Graph implements Serializable
{
   public static final int DIRECTED = 0;
   public static final int UNDIRECTED = 1;
   public static final int WEIGHTED = 2;
   public static final int UNWEIGHTED = 3;

   private int[][] adj;
   private int numVertices;
   private boolean directed;
   private boolean weighted;
   private String fileName;

   public Graph(int maze[][])
   {   
      int row = maze.length;
      int col = maze[0].length;
      
      numVertices = row*col;
      adj = new int[numVertices][numVertices];

      for(int i = 0; i < row; i++)
      {
         for(int j = 0; j < col; j++)
         {  

            if(maze[i][j] == 0)
            {
               int vertexFrom = i * col + j;
               int vertexTo;
               
               if((i-1) >= 0 && (i+1) < row)
               {
                  if(maze[i-1][j] == 0)
                  {
                     vertexTo = (i-1) * col + j;
                     addEdge(vertexFrom, vertexTo);
                  }
                  if(maze[i+1][j] == 0)
                  {
                     vertexTo = (i+1) * col + j;
                     addEdge(vertexFrom,vertexTo);
                  }
               }

               if((j-1) > 0 && (j+1) < col)
               {

                  if(maze[i][j-1] == 0)
                  {
                     vertexTo = i * col + (j-1);
                     addEdge(vertexFrom,vertexTo);
                  }

                  if(maze[i][j+1] == 0)
                  {
                     vertexTo = i * col + (j+1);
                     addEdge(vertexFrom,vertexTo);
                  }
               }
            }
         }
      }
   }

   public void print()
   {
      if (fileName != null)
         System.out.println("Constructed from file " + fileName);
      if (directed)
         System.out.print("DIRECTED");
      else
         System.out.print("UNDIRECTED");
      if (weighted)
         System.out.println(" WEIGHTED");
      else
         System.out.println(" UNWEIGHTED");
      System.out.println("Vertices: " + numVertices);
      System.out.println("Edges: " + edges());
      for(int v=0;v<numVertices;++v)
      {
         System.out.print(v + ": ");
         for(int u=0;u<numVertices;++u)
            if (adj[v][u] != 0)
               if (weighted)
                  System.out.print(" " + u + "<" + adj[v][u] + ">");
               else
                  System.out.print(" " + u);
         System.out.println();
      }
   } 

   public int vertices()
   {
      return numVertices;
   }

   public int edges()
   {
      int ctr = 0;
      for(int i=0;i<numVertices;++i)
         for(int j=0;j<numVertices;++j)
            if(adj[i][j] != 0)
               ++ctr;
      if (!directed)
         ctr = ctr/2;
      return ctr;
   }

   public void addEdge(int from,int to)
   {
      if (weighted)
         throw new Graph.Error("Illegal action on weighted graph");
      if (from<0 || from>=numVertices || to<0 || to>=numVertices)
         throw new Graph.Error("Invalid vertex");
      adj[from][to] = 1;
      if (!directed)
         adj[to][from] = 1;
   }

   public void addEdge(int from,int to,int weight)
   {
      if (!weighted)
         throw new Graph.Error("Illegal action on unweighted graph");
      if (from<0 || from>=numVertices || to<0 || to>=numVertices)
         throw new Graph.Error("Invalid vertex");
      if (weight <= 0)
         throw new Graph.Error("Invalid weight");
      adj[from][to] = weight;
      if (!directed)
         adj[to][from] = weight;
   }

   public static class Error extends RuntimeException
   {
      public Error(String message)
      {
         super(message);
      }
   }

   public void serialize(String fileName)
   {
      try
      {
         ObjectOutputStream outStream = 
            new ObjectOutputStream(new FileOutputStream(fileName));
         outStream.writeObject(this);
         outStream.close();
      }
      catch(Exception e)
      { throw new Graph.Error(e.toString()); }
   }

   public static Graph deserialize(String fileName)
   {
      try
      {
         ObjectInputStream inStream = 
            new ObjectInputStream(new FileInputStream(fileName));
         Graph g = (Graph) inStream.readObject();
         inStream.close();
         return g;
      }
      catch(Exception e)
      { throw new Graph.Error(e.toString()); }
   }
   
   public int numFewestEdgePath(int start,int stop)
   {
      int visited[] = new int[numVertices], parent[] = new int[numVertices],current,count=0;

      Arrays.fill(visited,-1);     
      Queue q = new Queue();
      q.enqueue(start);
      visited[start] = 0;
      
      while(!q.isEmpty())
      {
         current = (int)q.dequeue();
         for(int i=0;i<numVertices;i++)
         {
            if(adj[current][i] > 0 && visited[i] < 0)
            {
               q.enqueue(i);
               parent[i] = current;
               visited[i] = visited[parent[i]] +1;
            }
         }
      } 
      
      return visited[stop];
    }
   
   public ArrayList<Integer> fewestEdgePath(int start,int stop)
   {
      int visited[] = new int[numVertices], parent[] = new int[numVertices],current,count=0;
      ArrayList<Integer> listPath = new ArrayList<Integer>();
      
      Arrays.fill(visited,-1);     
      Queue q = new Queue();
      q.enqueue(start);
      visited[start] = 0;
      
      while(!q.isEmpty())
      {
         current = (int)q.dequeue();
         for(int i=0;i<numVertices;i++)
         {
            if(adj[current][i] > 0 && visited[i] < 0)
            {
               q.enqueue(i);
               parent[i] = current;
               visited[i] = visited[parent[i]] +1;
            }
         }
      }
 
      int k = stop;
      while(parent[k] != start)
      {
         listPath.add(parent[k]);
         k = parent[k];
      }

      Collections.reverse(listPath);
      
      return listPath;
    }
  }
