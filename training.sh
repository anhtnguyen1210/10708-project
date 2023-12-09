c1s='0.0'
c2s='0.0'
distribution_names='multiple_diracs'

for c1 in $c1s; do
    for c2 in $c2s; do
        for distribution_name in $distribution_names; do
            python cnf.py --weight_c1 $c1 --weight_c2 $c2 --distribution_name $distribution_name
        done
    done
done
