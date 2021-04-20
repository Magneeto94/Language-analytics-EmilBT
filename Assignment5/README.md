# Assignment 5 - (Un)supervised machine learning

<br>
DESCRIPTION
__Applying (un)supervised machine learning to text data__
<br>

Train an __LDA__ model on your data to extract structured information that can provide insight into your data. For example, maybe you are interested in seeing how different authors cluster together or how concepts change over time in this dataset.
<br>
You should formulate a short research statement explaining why you have chosen this dataset and what you hope to investigate. This only needs to be a paragraph or two long and should be included as a README file along with the code. E.g.: I chose this dataset because I am interested in... I wanted to see if it was possible to predict X for this corpus.
<br>
In this case, your peer reviewer will not just be looking to the quality of your code. Instead, they'll also consider the whole project including choice of data, methods, and output. Think about how you want your output to look. Should there be visualizations? CSVs?
<br>
You should also include a couple of paragraphs in the README on the results, so that a reader can make sense of it all. E.g.: I wanted to study if it was possible to predict X. The most successful model I trained had a weighted accuracy of 0.6, implying that it is not possible to predict X from the text content alone. And so on.
<br>


__Tips__
<br>
* Think carefully about the kind of preprocessing steps your text data may require - and document these decisions!
* Your choice of data will (or should) dictate the task you choose - that is to say, some data are clearly more suited to supervised than unsupervised learning and vice versa. Make sure you use an appropriate method for the data and for the question you want to answer
* Your peer reviewer needs to see how you came to your results - they don't strictly speaking need lots of fancy command line arguments set up using argparse(). You should still try to have well-structured code, of course, but you can focus less on having a fully-featured command line tool
<br>

__Bonus challenges__
<br>
Do both tasks - either with the same or different datasets

<br>
__General instructions__
<br>
* You should upload standalone .py script(s) which can be executed from the command line
* You must include a requirements.txt file and a bash script to set up a virtual environment for the project You can use those on worker02 as a template
* You can either upload the scripts here or push to GitHub and include a link - or both!
* Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line

<br>
__Purpose__
<br>
This assignment is designed to test that you have an understanding of:

* how to formulate research projects with computational elements;
* how to perform (un)supervised machine learning on text data;
* how to present results in an accessible manner.


____________________________________________________________________________________________________________________________________________
<br>
__Research:__
In this assignment I am investigating which topics there where disscussed on Reddit in relation to the Game Stop and their stocks. In this Game Stop Thread there where a lot of words like "Robinhood", "going to the moon" ect. that is usually not associated with stock exchange, there fore it could be interesting to look at the thread.
To make things a little bit easier for my self and for the computer I am only looking at the 200000 first posts in the data set.
The data set i am using was found on kaggle via this link: https://www.kaggle.com/unanimad/reddit-rwallstreetbets
<br>

__Results__
When trying different topics I found that using 4 topics gave the most satisfactory result, based on the fact that at least 2 topics where clustered with more topics than 4. 
By looking at the topics it looks like the 3. topic mostely conserns it self with waluable things like: "silver", "dimond" and money. While the first topic's top terms are the abbreviations: "gne" and "amc". But all in all it is hard to distingguish the kategories just by looking at them.
<br>

## To run script

Download the repo to your worker02 and run the bash script: "create_venv_ass5" to create the venv. Then activate it and run the python script "LDA.py".

I was not able to upload the data file, and I am  not sure why. The script will not run with out. You can download the dataset from this website: https://www.kaggle.com/unanimad/reddit-rwallstreetbets

But you are still able to look at the graphs in the output folder.


