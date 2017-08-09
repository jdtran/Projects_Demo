/**
 * @author John Tran (jtran57)
 * @version 5 December 2013
 *
 */

import java.awt.*;
import java.awt.event.*;
import java.applet.Applet;
import java.util.ArrayList;
import java.util.Iterator;
import javax.swing.*;

import java.util.Scanner;
import java.io.File;

public class Maze extends Applet {
   
   // array that holds the maze info
   private int[][] maze;

   //number of rows and columns in the maze
   private int rows, cols;
   
   boolean startPlaced = false;
   boolean stopPlaced = false;
   private int startVertex, stopVertex;
   private int startVertex_x, startVertex_y;
   
   ArrayList<Integer> blockedVertex = new ArrayList<Integer>();
   
   private boolean firstStep = false;
   private int numMovement = 0;
   ArrayList<Integer> currentPath = new ArrayList<Integer>();
   
   // initial size of maze - if bigger may go off window
   private final int MAXROWS = 20;
   private final int MAXCOLS = 30;
   
   // size of each block in pixels
   private final int blkSize = 20;
   
   //inner class that displays the maze
   private MazeCanvas mazeField;

   // everything is put on this panel   
   private JPanel mazePanel;

   // label, textfield, string, and load button for the file
   private JLabel fileLabel;
   private JTextField fileText;
   String fileName;
   boolean fileLoaded = false;
   private JButton fileButton;
   private JButton goButton;
   
   private Graph mazeGraph;
   //this listener object responds to button events
   private ButtonActionListener buttonListener;
   private MouseEventListener mouseListener;

   // this method sets up the canvas and starts it off
   public void init()
   {
      System.out.println("Maze started"); // goes to console 
      
      
      JTextField firstName = new JTextField();

      final JComponent[] inputs = new JComponent[] {
            new JLabel("Name?"),
            firstName,
      };
      JOptionPane.showMessageDialog(null, inputs, "Maze Game", JOptionPane.PLAIN_MESSAGE);
      
      JOptionPane.showMessageDialog(null, "Welcome to Cal Poly " + firstName.getText() + 
                                          ".\nThis game will simulate your career in our College of Engineering.\n" +
                                          "You are proudly wearing your Cal Poly sweater today, so you will be represented by a green block. (First mouse click) \n" + 
                                          "And the red block is your degree. (Second mouse click-and no your degree isn't covered in blood) " +
                                          "\n Good luck, may the engineering gods be with you."              
                                          , "Maze Game", 
                                          JOptionPane.INFORMATION_MESSAGE);
      
      
      mouseListener = new MouseEventListener();
      buttonListener = new ButtonActionListener();
               
      // the mazePanel is the panel that contains the maze interfaces, including
      // the buttons and output display
      mazePanel = new JPanel();
      // Y_AXIS layout places components from top to bottom, in order of adding
      mazePanel.setLayout(new BoxLayout(mazePanel, BoxLayout.Y_AXIS));
      
      // components for loading the filename
      fileLabel = new JLabel("File name:");
      mazePanel.add(fileLabel);
      fileText = new JTextField("", 20);
      mazePanel.add(fileText);
      fileButton = new JButton("Load File");
      mazePanel.add(fileButton);
      fileButton.addActionListener(buttonListener);
      
      // components for the gogo
      goButton = new JButton("ONWARDS!");
      mazePanel.add(goButton);
      goButton.addActionListener(buttonListener);

      // this is where the maze is drawn
      // if you add more to this layout after the mazeField, 
      //   it may be below the bottom of the window, depending on window size
      mazeField = new MazeCanvas();
      mazePanel.add(mazeField);
      mazeField.addMouseListener(mouseListener);

      // now add the maze panel to the applet
      add(mazePanel);
   }
   
   public ArrayList<Integer> shortestPathDrawMove()
   {  
      Graphics g = mazeField.getGraphics();
      
      ArrayList<Integer> shortPath = new ArrayList<Integer>();
      ArrayList<Integer> updateMovedPath = new ArrayList<Integer>();
      
      int getRow = 0;
      int getCol = 0;
          
      //draw the path
     
      shortPath = mazeGraph.fewestEdgePath(startVertex, stopVertex);
      
      for(int s : shortPath)
      {
         g.setColor(Color.yellow);
         g.fillRect((s%cols) * blkSize, (s/cols) * blkSize, blkSize, blkSize);
      }
      
      //clears the initial start position
      g.setColor(Color.white);
      g.fillRect(startVertex_x, startVertex_y, blkSize, blkSize);
      
      //draw the new position and initialize the new startingVertex
      getRow = shortPath.get(0)%cols;
      getCol = shortPath.get(0)/cols;
            
      g.setColor(Color.green);
      g.fillRect(getRow * blkSize, getCol * blkSize, blkSize, blkSize);
      
      startVertex = shortPath.get(0);
      startVertex_x = startVertex%cols * blkSize;
      startVertex_y = startVertex/cols * blkSize;
      
      for(int i = 1; i < shortPath.size(); i++)
      {
         updateMovedPath.add(shortPath.get(i));
      }
      
      return updateMovedPath;
   }
   
   public void clearSteps(ArrayList<Integer> listPath)
   {
      Graphics g = mazeField.getGraphics();
      
      for(int s : listPath)
      {
         g.setColor(Color.white);
         g.fillRect((s%cols) * blkSize, (s/cols) * blkSize, blkSize, blkSize);
      }
   }
   
   public String generateCourses()
   {
      ArrayList<String> courseNumbers = new ArrayList<String>();
      courseNumbers.add("cpe");
      courseNumbers.add("math");
      courseNumbers.add("phy");
      courseNumbers.add("ee");
      courseNumbers.add("12th \n" +
      		            "rot");
      
      int index = (int)(Math.random() * courseNumbers.size());
      
      return courseNumbers.get(index);
   }
   
   // this object is triggered whenever a button is clicked
   private class ButtonActionListener implements ActionListener {
      public void actionPerformed(ActionEvent event) 
      {
         Graphics g = mazeField.getGraphics();
         
         // find out which button was clicked 
         Object source = event.getSource();
         
         if (source == fileButton)
         {
            fileLoaded = true;
            fileName = fileText.getText();
            
            makeMaze(fileName);
            
            mazeGraph = new Graph(maze);
         }
         if(source == goButton && fileLoaded == false && numMovement > 1)
         {
            JOptionPane.showMessageDialog(null, "McDonalds is hiring.");
         }
         if(source == goButton && fileLoaded == true)
         {  
            if(firstStep == false)
            {
               firstStep = true;
               int numStep = mazeGraph.numFewestEdgePath(startVertex, stopVertex);
               
               if(numStep <= 0)
               {
                  JOptionPane.showMessageDialog(null, "Sorry, Poly kicked you out.");
                  fileLoaded = false;
               }
               else if(numStep <= 8)
               {
                  JOptionPane.showMessageDialog(null, "You lived- I mean...You graduated! Congraulations!");
                  fileLoaded = false;
               }
               
               if(numMovement == 0)
               {
                  currentPath = shortestPathDrawMove();
                  numMovement++;
               }
               else if(numMovement > 0 && numStep > 0)
               {
                  //clears old path, draws new;
                  clearSteps(currentPath);
                  currentPath = shortestPathDrawMove();
                  
                  //print all blocked paths
                  numMovement++;
               }
               
               for(int bv : blockedVertex)
               {
                  g.setColor(Color.black);
                  g.fillRect((bv%cols) * blkSize, (bv/cols) * blkSize, blkSize, blkSize);
               }
            }
            else if(firstStep == true)
            {
               firstStep = false;
               
               int randomBlock = (int)(Math.random()*currentPath.size());
               
               int vertex = currentPath.get(randomBlock);
               
               maze[vertex/cols][vertex%cols] = -1;
               mazeGraph = new Graph(maze);
               
               blockedVertex.add(vertex);
               
               for(int bv : blockedVertex)
               {
                  g.setColor(Color.black);
                  g.fillRect((bv%cols) * blkSize, (bv/cols) * blkSize, blkSize, blkSize);
               }
            }   
            //if first step is false, highlight path and take first step
            //else if first step is true, place block set first step false
         }
      } 
   }
   
   private class MouseEventListener implements MouseListener
   {
      public void mouseClicked(MouseEvent e)
      {
         // location on the mazeCanvas where mouse was clicked
         // upper-left is (0,0)
         
         Graphics g = mazeField.getGraphics();
         
         //three x,y coordinates, startXY given by mouse input. MazeXY "rounds" it. indexXY locate vertices on maze[][]
         int startX = e.getX();
         int startY = e.getY();
   
         int mazeX = startX - (startX % blkSize);
         int mazeY = startY - (startY % blkSize);
         
         int index_x = mazeY/blkSize;
         int index_y = mazeX/blkSize;
           
         //draw and declare start
         if(startY < rows * blkSize && startX < cols * blkSize && startPlaced == false && maze[index_x][index_y] == 0)
         { 
            g.setColor(Color.green);
            g.fillRect(mazeX, mazeY, blkSize, blkSize);
            
            startVertex_x = mazeX;
            startVertex_y = mazeY;
            
            startVertex = index_x * cols + index_y;
            startPlaced = true;
         }
         
         //draw and declare stop
         else if(startY < rows * blkSize && startX < cols * blkSize && startPlaced == true && stopPlaced == false && maze[index_x][index_y] == 0)
         {
            int checkStopVertex = index_x * cols + index_y;
            
            if(checkStopVertex != startVertex)
            {
               g.setColor(Color.red);
               g.fillRect(mazeX, mazeY, blkSize, blkSize);
               
               stopVertex = checkStopVertex;
               
               stopPlaced = true;
            }
         }
     }

      public void mousePressed(MouseEvent e) 
      { }
      public void mouseReleased(MouseEvent e) { }
      public void mouseEntered(MouseEvent e) { }
      public void mouseExited(MouseEvent e) { }
   }

   public boolean makeMaze(String fileName)
   {
      try
      {
         Scanner scanner = new Scanner(new File(fileName));
         rows = scanner.nextInt();
         cols = scanner.nextInt();
         maze = new int[rows][cols];
         //fill out maze matrix
         for(int i=0; i<rows; i++)
         {
            for(int j=0; j<cols; j++)
            {
               maze[i][j] = scanner.nextInt();
            }
         }
   
         // now draw it
         mazeField.paint(mazeField.getGraphics());
         return true;
      }
      catch(Exception e)
      {
         return false;
      }
   }
           
   class MazeCanvas extends Canvas {
      // this class paints the output window 
       
     // the constructor sets it up
      MazeCanvas() {
         rows = MAXROWS;
         cols = MAXCOLS;
         maze = new int[MAXROWS][MAXCOLS];
         setSize(cols*blkSize, rows*blkSize);
         setBackground(Color.white);
      }
 
      public void paint(Graphics g)
      {
         g.setColor(Color.white);
         g.fillRect(0, 0, cols*blkSize, rows*blkSize);
         
         for (int i=0; i<rows; i++)
         { 
            for (int j=0; j<cols; j++)
            {
               if (maze[i][j] == -1)
               {
                  // location is a wall
                  g.setColor(Color.black);
               }
               else
               {
                  // location is clear
                  g.setColor(Color.white);
               }
               // draw the location
               g.fillRect(j*blkSize, i*blkSize, blkSize, blkSize);
            }
         }
      }
   }
}
