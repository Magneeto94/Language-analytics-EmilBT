 This containes homework for assignment 4
 I have tried to make a bash script, but I am pretty unsurtain about it and if it works.
 If that does not work the way it is suppose to please tell me if you know what I det wrong.
 
 Otherwise you will have to download and run set up the venv from the command line by using these instructions:
 
 1. Download repo to your worker02 account by using: git clone https://github.com/Magneeto94/Language-analytics-EmilBT.git

2. then navigate to the folder: cd Assignment4

3. activate the venv: python -m venv networkanalysis
 
4. install the requirements: pip install -r requirements.txt

5. run the script (there is 2 arguments):
  -f the path to the file you would like to create a network from -f "data/fake_or_real_news-Copy1.csv"
  and the minimum number of edges the nodes shoud have in the network -e number.
 
The 2 files can be found in the folder called outpath. a csv-file and a png-file