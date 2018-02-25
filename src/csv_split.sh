 cd ~/../ml-20m/


DATA_FILE=ratings.csv
START=$(date +%s)

shuf $DATA_FILE | awk -F ',' '$1{ print >> ("outfile" (NR%2)+1 ".csv") }'

END=$(date +%s)
TIME=$((END-START))

printf "Renaming files to 'train.csv' and 'test.csv'...\n"
mv outfile1.csv train.csv
mv outfile2.csv test.csv

printf "Adding headers to train and test...\n"
sed -i 1i"userId,movieId,rating,timestamp" train.csv
sed -i 1i"userId,movieId,rating,timestamp" test.csv


printf "Checking train test split...\n"
COUNT1=$(wc -l train_data.csv | awk '{ print $1 }')
COUNT2=$(wc -l test_data.csv | awk '{ print $1 }')
printf "train_data obs %d\n" "$COUNT1"
printf "test_data obs %d\n" "$COUNT2"

printf "Ensuring no data overlap...\n"
awk 'FNR==NR{ a[$1];next }($1 in a){ print }' train_data.csv test_data.csv

printf "Execution time: %d\n" "$COUNT2"


