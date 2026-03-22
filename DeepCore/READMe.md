In this repository, we add the soft label training to the DeepCore repo. 
Example:
Command to run hard label testing for GradMatch:
sh test_hard_label.sh GradMatch 0 cuda:0 

Command to run soft label testing for GradMatch with temperature value of 10:
sh test_soft_label.sh GradMatch 0 cuda:0 \path\to\teacher 10