for i in {0..9}; do
  for j in 0 1; do # 0: adaboost; 1: logadaboost
    echo "i=$i, j=$j"
    ./src/main ./train.$i ./val.$i 100000 ./model-$j-$i $j > result-$j-$i.log
  done
done