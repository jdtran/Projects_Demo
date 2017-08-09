/**
 * -----------------------------------------------------------------------
 * @author  Kevin Nguyen John Tran
 * @version 16 July, 2015
 * @class   Side Project
 * @description Driver for test Score prediction using the 
 *              neural_network class
 * -----------------------------------------------------------------------
 */
import java.util.*;
import java.io.*;
import java.math.*;
import java.util.List;

public class test_prediction {
    //Generated neural network class    
    //Inputs and Outputs
    public static double inputX[][];
    public static double outputY[][];
    public static double out_yHat[][];
    public static neural_network midtermNetwork = new neural_network();
    
//----------------------------------------------------------------------//
//                         Nueral Network                               //
//----------------------------------------------------------------------//
    public static void build_network(){
        //2 inputs         (sleep and study)
        //3 hidden neurons (Randomly Weighted)
        //1 output         (test score)
        midtermNetwork = new neural_network(2, 1, 3);
    }
    
    public String printArray(double[][] m) {
        String result = "";
        for(int i = 0; i < m.length; i++) {
            for(int j = 0; j < m[i].length; j++) {
                result += String.format("%11.8f", m[i][j]);
            }
            result += "\n";
        }
        return result;
    }
    
//----------------------------------------------------------------------//
//                       Training Data Set                              //
//----------------------------------------------------------------------//	    
    public static void initialize_DataEntry(){
        //2D Array Here (Sleep Amt and Study Amt)
        
        inputX = new double[][]{
            {.3,.5},
            {.5, 1},
            {1, .2}
        };

        outputY = new double[][]{
            {.75},
            {.82},
            {.93}
        };
    }
    
    public static double[][] forward_network(){
        out_yHat = midtermNetwork.forward(inputX);
        return out_yHat;
    }
//----------------------------------------------------------------------//
//                      Predictions and Error                           //
//----------------------------------------------------------------------//
    public static String displayOutput(double[][] m, double[][]n) {
        String result = "";
        result += String.format("%9s%9s%9s\n", "Sleep","Study","Score");
        
        for(int i = 0; i < m.length; i++) {
            for(int j = 0; j < m[i].length; j++) {
                result += String.format("%11.2f", m[i][j] * 10);
            }
            for(int j = 0; j < n[i].length; j++) {
                result += String.format("%17.8f", n[i][j]);
            }
            result += "\n";
        }
        return result;
    }
    
//----------------------------------------------------------------------//
//                              MAIN                                    //
//----------------------------------------------------------------------//	 
    public static void main(String[] args){
        System.out.println("Initializing network.");
        initialize_DataEntry();
        build_network();
        System.out.println("Training...");
        this.train_network(); //Part 3 and 4
        System.out.println("Establishing cost...");
        midtermNetwork.costFunctionPrime(inputX, outputY);
        System.out.println("Predicted Scores for:\n" + 
             					displayOutput(inputX, forward_network()));
    }
}