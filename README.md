# NumPy Nets

### The library contains my works on ml/dl with Python3 using pure numpy:
#
#

#### 1. Polynomial Linear Regression:

###### This is a very basic problem of solving the data trend line, however in this case we are using polynomial 
###### approach for developing the understanding of how simple regression can solve multifeature (more realistic) tasks. 
###### Many thanks to blog by Davi Frossard at https://www.cs.toronto.edu/~frossard/topics/multiple-linear-regression/. 
#
###### Version 1.0 :

- Linear data is being generated automatically. If you want to update refer to utils.data_transforming.py
and update lambda functions;
#

    git clone NumpyWorks; python experiments/linear_regression.py
    
    Init arguments:
    
    -c - change configuration file (ex. -c "myconfig.json");
    -p - run with Matplot graphs (ex. -p true)       

#
![alt text](https://github.com/space1panda/NumpyWorks/blob/master/assets/linreg2.png)
![alt text](https://github.com/space1panda/NumpyWorks/blob/master/assets/linregfixed.png)
##### Figure 1: Exemplary Results - overfitting with higher order polynoms issue. 

###### Next version:

- Run training from docker image;
- Datasource object to train solution with real datasets;
- Using feedforward deep Neural Net as an alternative model;
#
#

#### 2. Logistic Regression w/wo deep Neural levels:

###### Solving linear classification problem on generated/real data using single perceptron / deep learning model with sigmoid ###### activation.

