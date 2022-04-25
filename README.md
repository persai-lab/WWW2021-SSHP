Code accomanying the paper: M. Yao, S. Zhao, S. Sahebi, and R. Feyzi Behnagh, “Stimuli-Sensitive Hawkes Processes for Personalized Student Procrastination Modeling,” in The Thirtieth Web Conference (The Web-21), 2021.

# WWW2021-SSHP

## Files:
* inference_real.py: for the predictions of parameters
* gradient_updates.py: for the computations gradients
* trace_projection.py: compute trace projection
* runner.py: for the running of SSHP

## Prerequsite:
* numpy
* pandas
* scipy
* sklearn

## Input format
Input data needs to be saved as a matfile under root as ''data.mat''.
This matfile contians the following subfields:
* X_train: training data that is an 2d array of the size of num_student x num_assignment, within which, each cell is an 1xK array of sorted ascending timestamps (e.g. cell in i-th column and j-th row is array([[2.446, 4.346,...,100.598]]).
* X_test: testing data that has the same structure as to X_train.
* mask: a 2d array of the size of num_student x num_assignment, with 1 and 0 respectively representing the corresponding user,assignment pair is available or not. 

## Run SSHP
run SSHP using the following command:
python3 runner.py.
One can change hyperparamter decay by setting it to different values, e.g. 10.
The estimated parameters will be saved as a matfile under root called "est_decay_{decay value}".


## Collaborators:
* Miley Yao: myao@albany.edu
* Sherry Sahebi: ssahebi@albany.edu
* Reza Feyzi Behnagh: rfeyzibehnagh@albany.edu
## Acknowledgement: 
This material is based upon work supported by the National Science Foundation under Grants Number [1917949](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1917949).
