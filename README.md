# 10708 Fall 23 Final Project: Regularized Neural ODEs  
This is the repository for 10708 Fall 23 Final Project: Regularized Neural ODEs by Anh Nguyen (_atnguyen@andrew.cmu.edu_), Sangyun Lee (_sangyunl@andrew.cmu.edu_), and Noah Freedman (_nfreedma@andrew.cmu.edu_).


## How to run
### Create conda environment 
```
conda create --name <env_name> --file requirements.txt
```

### Run code

```
conda activate <env_name>
python cnf.py --viz --distribution_name <distribution_name>
```
where `<distribution_name>` is of `two_circles` | `two_diracs` | `checkerboard` | `multiple_diracs`.