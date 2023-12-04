c1s='0.0 0.1 0.2 0.3 0.4 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0'
c2s='0.0 0.1 0.2 0.3 0.4 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0'
distribution_names='two_circles checkerboard multiple_diracs'

for c1 in $c1s; do
    for c2 in $c2s; do
        for distribution_name in $distribution_names; do
            python cnf.py --weight_c1 $c1 --weight_c2 $c2 --distribution_name $distribution_name
        done
    done
done
