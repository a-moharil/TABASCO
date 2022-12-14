# User Guide : v1.1
This document consists of the guidelines indicating the usage of TABASCO : A Transformer Based Contextuatlization Toolkit.

To use TABASCO:
  1) Clone the repository in any of your local folders.
  2) Install all the dependancies :`
  ```pip install -r /path/to/requirements.txt```
  3) In the repo directory use the following command to start the application :
  ```python app.py ``` or ```python3 app.py ```
  4) In your browswer open ```http://localhost:5000```, (once the cursor in the terminal starts blinking) to get to the home page of TABASCO 

NOTE :- 
1) To use the GPU kindly follow the pytorch installation depending on your OS and CUDA version : https://pytorch.org/.
2) Before installing CUDA, make sure you have the appropriate nvidia-drivers installed on your machine :
   
   a) Nvidia-Driver installation guide for Ubuntu 20.04 : https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux
   
   b) Nvidia-Driver ```.exe```  download link for Windows :- https://www.nvidia.com/download/index.aspx
   
4) Make sure that you have CUDA installed on your device. 
   
   a) Installation guide for Windows :- https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
   
   b) Installation guide for Linux :- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

**we highly recommend using the GPU to accelerate the computation and reduce the processing time.** 

**POTENTIAL ERROR FIXES :**

After running the command ``` python app.py``` if you get an error :
```<root dir>\<User>/.cache\\huggingface\\hub\\models--bert-base-uncased\\snapshots\\5546055f03398095e385d7dc625e636cc8910bf2\\pytorch_model.bin' not found```
or 
```<root dir>\<User>/.cache\\huggingface\\hub\\models--bert-base-uncased\\snapshots\\5546055f03398095e385d7dc625e636cc8910bf2\\config.json' not found```

Then go to the directory ```/.cache\\huggingface\\hub\\models--bert-base-uncased\\snapshots\\5546055f03398095e385d7dc625e636cc8910bf2\\```, open the google drive link in ``` ./bert-dependencies``` folder and unzip all the contents in the said directory.

**TO NOTE : If you are using Windows :**
  1) Kindly install python : https://www.python.org/downloads/.
  2) Make sure you tick the check box on : ```ADD TO PATH``` while installing.

## Home
Once the user runs the app.py script on the terminal, a web-page will be hosted on the user local port (default: 5000). The home page of the toolkit looks as follows :
![home](https://user-images.githubusercontent.com/93342024/194392711-96758f1d-f97c-4b07-b5ca-6d4a36c71d03.png)
When directed on the home page, the user is requested to upload a document via the dropzone as follows:

![upload_final](https://user-images.githubusercontent.com/93342024/194393556-04245dd0-0b8c-4a4f-828e-72a1d729f1bb.png)
It is to be noted that the maximum upload file size is 500 mb and the allowed format is strictly restrited to '.pdf' and '.txt'.

Once the document has been uploaded, TABASCO will ask the user to select the number N of most frequently occurring top nouns from the document. This number ranges from 25 to 200, with a step size of 25.
![topn](https://user-images.githubusercontent.com/93342024/194394989-1721598c-3d7f-4f9e-891a-5e2492482716.png)

After selecting N, the user can press the submit button to get the top N nouns for disambiguation.

## Frequency List
On the '\list' web-page, the user can see the N most frequently used nouns and their corresponding freqency of occurrence.
![lcrop](https://user-images.githubusercontent.com/93342024/194396088-e0469e6e-057d-4ad6-b909-edb6969edc1a.png)

The frequency plot on clicking will expand in another tab and the user can have a detailed look about the frequency distribution of the N words in the document.
![frq_plot](https://user-images.githubusercontent.com/93342024/194396590-4c5e648e-41bb-4347-b5f3-aa6e8cd83541.png)

From this list of top N nouns, the user is supposed to select a term T that needs to be contextualized or is ambigous from the assessment and pre-judgement of the user. 
The another parameter that the user needs to select is the maximum frequency F of occurrence of all the sentences consisting of the target term T. If F is lower than the actual frequency in which T occurs then TABASCO will consider only F sentences, if F is greater than the acutal frequency then the computation is done over the actual number of sentences in which T occcurs. F ranges from 0 to 3000 with a step size of 50.
![targterm](https://user-images.githubusercontent.com/93342024/194397703-06b4207e-9f1b-4f02-931c-69b483682b98.png)

The user can click submit once the target term has been entered and F has been selected.

## Target Matrix
The '\targetmat' web-page displays the predicted contexts for the word T alog with the elbow plot of the clustering algorithm.
![pred](https://user-images.githubusercontent.com/93342024/194398729-7f58d817-ba67-402e-a556-c3e4c5e0e468.png)
After, obtaining this prediction regarding the number of context clusters, the user is directed to select the threshold ??. The value of ?? ranges from 0-1 with a step size of 0.1. We recommend a default thereshold 0.48 but this can vary with the datasets under investigation.

![thresss](https://user-images.githubusercontent.com/93342024/194400474-a78f0a49-bae0-47e1-82da-e7c6eba93b08.png)

After selecting the threshold, the user will be redicrected to the '\context' page where the user can view the corresponding thereshold plots for the word labels and can read the detailed and the summary reports.
![context1](https://user-images.githubusercontent.com/93342024/194400807-c5333930-341c-47bf-abbe-1bd97187c4d5.png)

Below the threshold plots, the user will find the buttons for the detailed (blue) and the summary (yellow) reports.
![context2](https://user-images.githubusercontent.com/93342024/194400965-b01fc992-21f0-4821-a31e-ced17d3bbc3a.png)

## Results

### Detailed Report
As discussed in the paper, the detailed report for all corresponding predicted clusters consists of the occurrences of T in that respective cluster, with the corresponding sentance instance and the predicted set of context words that indicate the context implication of T in that particular sentence.

![detailed](https://user-images.githubusercontent.com/93342024/194401388-b741eb6f-3665-4983-bc50-172ac8398774.png)

### Summary Report
The summary report for every corresponding cluster consists of the top 50 (or less, depending on ??) context words from that respective cluster along with 50 (or less) random isntances of T which imply the usage of T in context of that respective cluster. 
![summary](https://user-images.githubusercontent.com/93342024/194401840-5620eddd-8ca5-4b88-835c-4afcf57f4377.png).

After obtaining these results, the user can play with the values ?? (by going back to the '\targetmat' page) based on the threshold plots and keep obtaining finer results in every successive iteration. 











