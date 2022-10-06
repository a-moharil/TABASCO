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
