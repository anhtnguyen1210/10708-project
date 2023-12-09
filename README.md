# 10708 Fall 23 Final Project: Regularized Neural ODEs  
This is the repository for 10708 Fall 23 Final Project: Regularized Neural ODEs by Anh Nguyen (_atnguyen@andrew.cmu.edu_), Sangyun Lee (_sangyunl@andrew.cmu.edu_), and Noah Freedman (_nfreedma@andrew.cmu.edu_).


## How to run
### Create conda environment 
To create a virtual env for reproducing reported results, run the code
```
$conda create --name <env_name> --file requirements.txt
$conda activate <env_name>
```

### Run code

```
$conda activate <env_name>
$python cnf.py --viz --distribution_name <distribution_name>
``` 
where `<distribution_name>` is of `two_circles` | `two_diracs` | `checkerboard` | `multiple_diracs`. Though all results reported are for `two_circles` and `two_diracs`, feel free to play around with other distributions. For each config `weight_c1`, `weight_c2`, and `distribution_name`, there should be a corresponding checkpoints file `<distribution_name>_<weight_c1>_<weight_c2>.pth` and a log file `<distribution_name>_<weight_c1>_<weight_c2>.log` saved in `checkpoints/` and `logs/` folders, respectively, which can be used for evaluating. 

### Visualizing
Visualizing functions can be found at `draw.py`. 




