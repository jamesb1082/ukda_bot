
## Getting Started ## 
This repository contains different attempts to try and create a chatbot, from frameworks, 
distance functions, and ultimately a siamese neural network (SNN). Quick steps:
* Set up a Python virtual environment. 
* Run the bash script setup.sh 
* Populate the data directory. 
* Make minor modifications to the wmd script paths. 
* Run a relevant python file to test that chatbot method. 

## Prerequisites ## 
* Python 2.7
* It is recommended that a new python virtual environment is set up for this project. 
* Please ensure that both setup.sh and misc/download_word2vec.sh have the relevant permissions.


## Installing ## 
* By running Setup.sh you download and install all relevant pretrained vectors, and python 
packages to get this project working "out of the box". Please note that these downloads are 
large and so will take a while.

## Deployment ## 
This project is more a proof of concept than a deployable product, and as such there are no deployment notes. 


## Explanation of Underlying Data System ## 

For this project, a relatively simple interface has been created between the files stored and 
the various different approaches to creating a chatbot. One can find this interface as a Python object in dm/managers.py. 

The Datamanager object looks for questions in data/questions directory as default, however, this can be manually changed when the object is instantiated. In those directories, it is looking for txt files where each file will have:

* A unique file name
* Contain only one question or answer 

Now, the correct qa pairs need to be specified, one needs to edit data/qa.csv to contain rows of the format: 

```
question1,answer1
```

Now you should have a working data directory. It is worth noting that some tests might need fortuher files, but that is done on a case by case basis.

## Running Different Chatbots ## 

### Approach 1: Baseline Bot ### 
This uses the chatterbot python package and loads your own corpus of correct qa pairs into it to see if it will act correctly. To do this run baseline_bot/simple_bot.py from inside the basline_bot directory.

### Approach 2: Stats Analysis ### 
It is worth mentioning that the stats_analysis chatbot does not have an interactive chat ability. This approach attempts to use cosine similarity and wmd as distant functions to select answers which are closest according to the particular distance function. To run this, run evaluation.py from inside the directory stats_analysis.

### Approach 3: SNN Chatbot ### 
This approach uses a SNN to try and learn a distance function and was the best performing technique in our dataset. This requires creating a specalist dataset to learn the initial dataset.

To set up this approach: 
* Open a python interactive shell in the snn_chatbot directory. 
* type in "import preprocess as p"
* Then type "p.create_dataset(N)" where N is the first N questions in your qa.csv files. 

The chatbot first needs training, which is done by running train_bot.py inside the snn_chatbot directory.  
#### 3.1 Terminal Application ####
The chatbot itself can be run inside terminal which is done by running the chatbot.py file inside the snn_chatbot directory. 

#### 3.2 Simple webpage using Flask #### 
If you want to show your chatbot to users who do not really know how to use terminal, there is a simple flask site, provided which will load a pretrained model and can be interacted with, via a HTML
page on local host. 
