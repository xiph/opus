mkdir datasets
cd datasets
for i in `grep https ../../../datasets.txt`
do
	wget $i
done
