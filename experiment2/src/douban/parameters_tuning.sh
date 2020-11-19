for sample_times in 20 40 60 80 100
  do for walk_steps in 40 60 80
    do for alpha in "0.0025" "0.01" "0.025" "0.1"
      do for dimensions in 64 128 256 512
        do for lambda1 in "0.0025" "0.01" "0.025" "0.1" "0.25"
          do for lambda2 in "0.0025" "0.01" "0.025" "0.1" "0.25"
            do for lambda_factor in "0.0025" "0.01" "0.025" "0.1" "0.25"
              do echo "do ./ccse --sample_times $sample_times --walk_steps $walk_steps --alpha $alpha --dimensions $dimensions --lambda1 $lambda1 --lambda2 $lambda2 --lambda_factor $lambda_factor"
              # do ./ccse --sample_times $sample_times --walk_steps $walk_steps --alpha $alpha --dimensions $dimensions --lambda1 $lambda1 --lambda2 $lambda2 --lambda_factor $lambda_factor
            done
          done
        done
      done
    done
  done
done
