# User Guide
This document consists of the guidelines indicating the usage of TABASCO : A Transformer Based Contextuatlization Toolkit

## Home
Once the user runs the app.py script on the terminal, a web-page will be hosted on the user local port (default: 5000). The home page of the toolkit looks as follows :
![home](https://user-images.githubusercontent.com/93342024/194392711-96758f1d-f97c-4b07-b5ca-6d4a36c71d03.png)
When directed on the home page, the user is requested to upload a document via the dropzone as follows:

![upload_final](https://user-images.githubusercontent.com/93342024/194393556-04245dd0-0b8c-4a4f-828e-72a1d729f1bb.png)
It is to be noted that the maximum upload file size is 500 mb and the allowed format is strictly restrited to '.pdf' and '.txt'.
Once the document has been uploaded, TABASCO will ask the user to select the number N of most frequently occurring top nouns from the document. This number ranges from 25 to 200, with a step size of 25.
![topn](https://user-images.githubusercontent.com/93342024/194394989-1721598c-3d7f-4f9e-891a-5e2492482716.png)
Once the document has been uploaded and N has been selected, the user can press the submit button to get the top N nouns for disambiguation.

# Frequency List
On the '\list' web-page, the user can see the N most frequently used nouns and their corresponding freqency of occurrence.
![lcrop](https://user-images.githubusercontent.com/93342024/194396088-e0469e6e-057d-4ad6-b909-edb6969edc1a.png)

The frequency plot on clicking will expand in another tab and the user can have a detailed look about the frequency distribution of the N words in the document.
![frq_plot](https://user-images.githubusercontent.com/93342024/194396590-4c5e648e-41bb-4347-b5f3-aa6e8cd83541.png)

From this list of top N nouns, the user is supposed to select a term T that needs to be contextualized or is ambigous from the assessment and pre-judgement of the user. 
The another parameter that the user needs to select is the maximum frequency F of occurrence of all the sentences consisting of the target term T. If F is lower than the actual frequency in which T occurs then TABASCO will consider only F sentences, if F is greater than the acutal frequency then the computation is done over the actual number of sentences in which T occcurs.
![targterm](https://user-images.githubusercontent.com/93342024/194397703-06b4207e-9f1b-4f02-931c-69b483682b98.png)

The user can click submit once the target term has been entered and F has been selected.


