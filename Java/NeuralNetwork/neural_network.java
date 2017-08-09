/**
 * -----------------------------------------------------------------------
 * @author  Kevin Nguyen John Tran
 * @version 16 July, 2015
 * @class   Side Project
 * @description Class for a shallow network made up of variable inputs,
 *              a single output, and a single hidden layer containing
 *              N neurons with 2N connections.
 *
 * -----------------------------------------------------------------------
 */
 
import java.util.*;
import java.io.*;
import java.math.*;

public class neural_network{
    //User defined input
    public int hours_sleep;
    public int hours_studied;
    //Outputs
    public int actual_score = 0;
    public int predicted_score = 0;
    //Define Hyperparameters
    public int inputLayerSize  = 0;
    public int outputLayerSize = 0;
    public int hiddenLayerSize = 0;
    public int num_weights = 0;  
    public double[][] W1;
    public double[][] W2;
    public double[][] z2, a2, z3;
    //backpropagation variables
    public double[][] delta3, dJdW2, delta2, dJdW1;

    public neural_network(){
        inputLayerSize  = 0;
        outputLayerSize = 0;
        hiddenLayerSize = 0;
        num_weights = 0;
        W1 = null;
        W2 = null;
    }
    
    public neural_network(int input_size, int output_size, 
                          int hidden_size){
        inputLayerSize  = input_size;
        outputLayerSize = output_size;
        hiddenLayerSize = hidden_size;
        num_weights = input_size * hidden_size;
        W1 = randWeight(input_size, hidden_size);
        W2 = randWeight(hidden_size, output_size);
    }

//----------------------------------------------------------------------//
//                         Class Functions                              //
//----------------------------------------------------------------------//   
    public double[][] randWeight(int row, int col){
        //Create neurons with random weights
        double newRandomMatrix[][] = new double[row][col];
        
            for(int i = 0; i < row; i++){
                for(int j = 0; j < col; j++){
                    newRandomMatrix[i][j] = Math.random();
                }
            }
        return newRandomMatrix;
    }

//----------------------------Forwarding--------------------------------//
    public double[][] forward(double[][] input_X){
        //Propagate inputs though network
        double[][] y_hat;
        
        this.z2 = this.dot(input_X, W1);
        this.a2 = this.sigmoidMatrix(z2);
        this.z3 = this.dot(a2, W2);
        y_hat = this.sigmoidMatrix(z3); 

        return y_hat;
    }
    
    //Activation Function
    public double sigmoid(double z){
        return 1/(1+Math.exp(-z));
    }
    
    public double[][] sigmoidMatrix(double z[][]){
    	int row = z.length;
    	int col = z[0].length;
        double newSigmoidMatrix[][] = new double[row][col];
        
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                double value = z[i][j];
                newSigmoidMatrix[i][j] = 1/(1+Math.exp(-value));
            }
        }
       
        return newSigmoidMatrix;
    }
//--------------------------Backpropagation-----------------------------//
    public double costFunction(double[][] y, double[][] y_hat){
        double sum = 0;
        int row = y.length;
        int col = y[0].length;
        
        for(int i = 0; i<row; i++){
            for(int j = 0; j<col; j++){
                sum += (Math.pow((y[i][j] - y_hat[i][j]),2));
            }
        }
        sum /= 2;
        
        return sum;
    }
    
    public double[][] sigmoidPrime(double[][] z){
        int row = z.length;
    	int col = z[0].length;
        double newSigmoidMatrix[][] = new double[row][col];
        
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                double value = z[i][j];
                newSigmoidMatrix[i][j] = Math.exp(value)
                                       / Math.pow((1+Math.exp(-value)),2);
            }
        }
    }

    public void costFunctionPrime(double[][] y, double[][] y_hat){

        delta3 = -(y-y_hat) * sigmoidPrime(z3);
        dJdW2 = dot(transpose(a2),delta3);
        
        delta2 = dot(delta3, transpose(W2) * sigmoidPrime(z3));
        dJdW1 = dot(transpose(X), delta3);
    }
}//end of class
