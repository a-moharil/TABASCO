# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import requests
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from scipy.stats import norm
from sklearn.mixture import GaussianMixture #For GMM clustering
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
import torch
from transformers import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
import shutil
import PyPDF2
from PyPDF2 import PdfReader
import seaborn as sns
from scipy.spatial.distance import cosine
import pickle
from sklearn_extra.cluster import KMedoids
import os
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import math
from collections import Counter
from nltk.collocations import *
from nltk import ngrams, FreqDist
import operator
from nltk.corpus import stopwords
from itertools import islice
import matplotlib.animation as animation
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import InterclusterDistance
from sklearn.manifold import MDS
from yellowbrick.cluster import KElbowVisualizer
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import matplotlib.pyplot as plt
from waitress import serve
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
#FLASK IMPORTS#

from flask import Flask, render_template, url_for, redirect, request
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
from flask import send_from_directory

dir_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DROPZONE_MAX_FILE_SIZE'] = 100
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0



ALLOWED_EXTENSIONS = set([ 'pdf', 'txt'])
dropzone = Dropzone(app)

global device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
#torch.cuda.max_memory_allocated(device=device)
print(torch.device(dev))
print(f'\n *** The selected device being used is {str(device).title()} ***\n')
## Toolkit Helper Functions ##

class bold_color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def standardize_text(text_field):
    text_field = text_field.replace(r"http\S+", " ")
    text_field = text_field.replace(r"http", " ")
    text_field = text_field.replace(r"(\d)", " ")
    text_field = text_field.replace(r"@\S+", " ")
    text_field = text_field.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n,\\,/,.,:,;,'""']", " ")
    text_field = text_field.replace(r"\\", " ")
    text_field = text_field.replace(r".", " ")
    text_field = text_field.replace(r"!", " ")
    text_field = text_field.replace(r";", " ")
    text_field = text_field.replace(r",", " ")
    text_field = text_field.replace(r":", " ")
    text_field = text_field.replace(r"←", " ")
    text_field = text_field.replace(r"≠", " ")
    text_field = text_field.replace(r"'", " ")
    text_field = text_field.replace(r"(", " ")
    text_field = text_field.replace(r")", " ")
    text_field = text_field.replace(r"[", " ")
    text_field = text_field.replace(r"]", " ")
    text_field = text_field.replace(r"[]", " ")
    text_field = text_field.replace(r"?", " ")
    text_field = text_field.replace(r"()", " ")
    text_field = text_field.replace(r'"', " ")
    text_field = text_field.replace(r"-", " ")
    text_field = text_field.replace(r"{", " ")
    text_field = text_field.replace(r"}", " ")
    text_field = text_field.replace(r"*", " ")
    text_field = text_field.replace(r"~,!", " ")
    text_field = text_field.replace(r"@", " ")
    text_field = re.sub("[?]", " ", text_field)
    #text_field = text_field.replace(r"#", " ")
    text_field = text_field.replace(r"$", " ")
    text_field = text_field.replace(r"%", " ")
    text_field = text_field.replace(r"^", " ")
    text_field = text_field.replace(r"&", " ")
    text_field = text_field.replace(r"=", " ")
    text_field = text_field.replace(r"+", " ")
    text_field = text_field.replace(r"`", " ")
    text_field = text_field.replace(r"<", " ")
    text_field = text_field.replace(r">", " ")
    text_field = text_field.replace(r"·", " ")
    text_field = re.sub("[”“]", " ", text_field)
    text_field = text_field.replace(r"//", " ")
    text_field = text_field.replace(r"|", " ")
    text_field = text_field.replace(r"|", " ")
    text_field = text_field.replace(r"&[A-Z][a-z][0-9]", " ")
    text_field = text_field.replace(r"[0-9]+", " ")
    text_field = text_field.replace(r"[a-z]+", " ")
    text_field = text_field.replace(r"[a-zA-z]", " ")
    text_field = text_field.replace(r"\[0-9a-zA-Z]", " ")
    text_field = re.sub("[–]", " ", text_field)
    text_field = text_field.replace(r"λ", " ")
    text_field = text_field.replace(r"@", "at")
    text_field = text_field.lower()
    text_field = re.sub("\s[0-9]+", " ", text_field)
    text_field = re.sub("\b[a-z]\b", " ", text_field)
    text_field = re.sub("—", " ", text_field)
    text_field = re.sub("_", " ", text_field)
    text_field = re.sub("™", " ", text_field)
    text_field = re.sub("/", " ", text_field)
    text_field = re.sub("[0-9]", " ", text_field)
    text_field = re.sub("[½¼¢~]", " ", text_field)
    text_field = text_field.replace('\\n', " ")
    text_field = text_field.replace("(", " ")
    text_field = text_field.replace(")", " ")
    #text_field = text_field.replace("#", " ")
    text_field = text_field.replace("&", " ")
    text_field = text_field.replace("\\", " ")
    text_field = text_field.replace('\\n', "")
    text_field = text_field.replace("(", "")
    text_field = text_field.replace(")", "")
    text_field = ' '.join(i for i in text_field.split() if not (i.isalpha() and len(i) == 1))
    return text_field

def convert_size(size_bytes):
    if size_bytes == 0:
       return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def pdf2text(input_text):
    from PyPDF2 import PdfReader
    text_list = []
    #size = os.path.getsize(input_text)
    #size = convert_size(size)
    #if size < restricted_size:  # restricting the file size to 100MB pdf.
    reader = PdfReader(input_text)
    number_of_pages = len(reader.pages)
    for page in range(number_of_pages):
        curr_page = reader.pages[page]
        curr_text = curr_page.extract_text()
        text_list.append(curr_text)
    str_text = " ".join(text_list)
    return str_text
    # else:
    #     print(
    #         f'The uploaded file is bigger than {restricted_size}, kindly upload a file less than the {restricted_size}')
    #
def get_sent(text, input_word, target_list):
    global inp_str
    global wrt
    sent_word = []
    inp_str_idx = target_list.index(input_word)
    inp_str = target_list[inp_str_idx]
    wrt = str(inp_str + 's')
    sentences = sent_tokenize(text)
    #print(sentences) #remove this tag
    #global sent_word
    #sent_word = [" ".join([sentences[i-1], j, sentences[i+1]]) for i,j in enumerate(sentences) if inp_str in word_tokenize(j)]
    for i,j in enumerate(sentences):
        try:
            words_of_sent = word_tokenize(j)
            if inp_str in words_of_sent:
                sent_word.append(j)
                # sent_word.append(" ".join([sentences[i-1], j, sentences[i+1]]))
        except IndexError:
            #sent_word.append('None')
            continue
    return sent_word

#Create a list output for cleaned fetched sentences of the target term
def filter_sent(list_input):
    global filt_sent_
    filt_sent_= []
    for cln in list_input:
        cleaned = standardize_text(cln)
        filt_sent_.append(cleaned)
        #return filt_sent_


def get_target_matrix(text_file, model, Frequency, target_list, wrd):
    global inter_sent, vector_mat_list, label_list
    vector_mat_list, label_list = [], []
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    sent_word = get_sent(text_file, wrd, target_list)  # Fetch the sentences cprresponding to the selected word
    print(len(sent_word))
    vector_bucket_return, word_bucket_return = [], []  # Vocab of each sentence is stored in vector bucket and word bucket, to avoid overwritting we need another set of list to store the vectors and resp vocabulary.
    if len(sent_word) <= Frequency:
        print(f"Less than {Frequency} occurences of the word {wrd} found")
        print("\n Processing \n")
        vector_bucket, inter_sent, sent_word_, word_bucket = [], [], [], []


        dir = dir_path + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        for gh in range(len(sent_word)):
            sent_word_.append(sent_word[gh])
        filter_sent(sent_word_)



        # Looping issue is occurring (list overwrting most probably hence the target matrix is not being generated.)
        for sent_par in range(len(filt_sent_)):
            try:
                vector, vector_bucket, word_bucket = model.forward(filt_sent_[sent_par])
                inter_sent.append(sent_word_[sent_par])
                vector_mat_list.append(vector)
                label = inp_str + " " + str(sent_par)
                label_list.append(label)

                vector_bucket_return = [vocab_vect for vocab_vect in vector_bucket]
                word_bucket_return = [word_vect for word_vect in word_bucket]


            except (IndexError, RuntimeError) as e:
                # del locals()["sent_word_" + str(len(sent_word))][sent_par]
                residual_list = []
                final_sent_list = []
                residual_list.append(sent_word_[sent_par])
                continue
    else:
        print(f"word frequency exeeded than {Frequency}, limiting to {Frequency}")
        vector_bucket, inter_sent, sent_word_, word_bucket = [], [], [], []

        dir = dir_path + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        for gh in range(Frequency):
            sent_word_.append(sent_word[gh]) #an additional list sente_word_ has been created to limit on the frequency
        filter_sent(sent_word_)

        model = DisambModel(bert_model, bert_tokenizer, device=device)

        for sent_par in range(len(filt_sent_)):
            try:
                vector, vector_bucket, word_bucket = model.forward(filt_sent_[sent_par])
                inter_sent.append(sent_word_[sent_par])
                vector_mat_list.append(vector)
                label = inp_str + " " + str(sent_par)
                label_list.append(label)
            except (IndexError, RuntimeError) as e:
                # del locals()["sent_word_" + str(len(sent_word))][sent_par]
                residual_list = []
                final_sent_list = []
                residual_list.append(sent_word_[sent_par])
                continue

    # 3 lists have been created for the input string
    # sent_word ---- containing the sentence in which the word has been used
    # vector_mat_list ---- containing the embedded vector wrt to the sentence
    # label_list ----- labelled input string

    ## NxN Matrix Creation
    input_shape = len(vector_mat_list)
    target_matrix = torch.zeros(size=(input_shape, input_shape))

    # if torch.cuda.is_available():
    #     dev = "cuda:0"
    # else:
    #     dev = "cpu"
    # device = torch.device(dev)
    #
    # if device == "cuda:0":
    #     vector_mat_list = vector_mat_list.cpu()
    #     print(vector_mat_list[0])

    matrix_list = []
    for i_ in range(input_shape):
        for j_ in range(input_shape):
            # target_matrix[i_][j_] = 1 - cosine(vector_mat_list[i_].cpu(), vector_mat_list[j_].cpu())
            # matrix_list.append([i_, target_matrix[i_][j_]])
            target_matrix[i_][j_] = cos(vector_mat_list[i_], vector_mat_list[j_])

            # print(str(i_), str(j_))

    mat_dir = dir_path + "/matrices/"
    if os.path.exists(mat_dir):
        shutil.rmtree(mat_dir)
    os.makedirs(mat_dir)

    with open(mat_dir + inp_str + "_matrix.dat", "wb") as tm:
        pickle.dump(target_matrix, tm)
    print(f'The similarity matrix is of dimension {target_matrix.shape}')
    return target_matrix, vector_bucket_return, word_bucket_return

def get_optimal_k(target_matrix, flag=0):
    # if device == "cuda:0":
    #     X = target_matrix.numpy()
    # else:
    #     X = target_matrix
    X = target_matrix.numpy()
    if flag ==0:
        pca = PCA(n_components=5)
    if flag==1:
        pca = PCA(n_components=1, svd_solver='full')

    X_ = pca.fit_transform(X)

    clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
    clf.fit(X)

    K = range(1, 11)
    wcss = []
    distortions = []
    for i in K:
        kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    kn = KneeLocator(list(K), distortions, S=1.0, curve='convex', direction='decreasing')
    figure = Figure()
    figure = plt.plot(range(1, 11), wcss)
    print(f"the knee point is {kn.knee}")
    plt.title(" The Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.ylabel("WCSS")
    plt.savefig(
        dir_path + "/static/cluster_plot/" + "elbow_plot.png")
    plt.clf()
    plt.close()

    per_dec_list = []
    k_op = 1
    K_clust = 1
    K_clustr = 1
    for wc in range(0, len(wcss) - 1):
        perc_dec = (1 - (wcss[wc + 1] / wcss[wc])) * 100
        per_dec_list.append(perc_dec)
    print('\n')

    for per in range(len(per_dec_list)):
        if per_dec_list[per] > 23:
            k_op = wcss.index(wcss[per + 1])
            K_clust = k_op + 1
        else:
            K_clustr = 1

    if K_clustr == K_clust:
        optimal_k = 1
    else:
        optimal_k = K_clust

    if optimal_k > 5:
        optimal_k = 1

    return optimal_k


def get_clusters(optimal_k, X):
    word_of_cluster_0, sent_of_cluster_0, vocab_of_cluster_0 = [], [], []
    word_of_cluster_1, sent_of_cluster_1, vocab_of_cluster_1 = [], [], []
    word_of_cluster_2, sent_of_cluster_2, vocab_of_cluster_2 = [], [], []
    word_of_cluster_3, sent_of_cluster_3, vocab_of_cluster_3 = [], [], []
    word_of_cluster_4, sent_of_cluster_4, vocab_of_cluster_4 = [], [], []
    word_of_cluster_5, sent_of_cluster_5, vocab_of_cluster_5 = [], [], []
    print(len(word_of_cluster_1))
    if optimal_k == 1:
        kmeans = KMeans(n_clusters=optimal_k, init="k-means++",
                        max_iter=300, n_init=10, random_state=42)

        kmeans.fit(X)

        y_kmeans = kmeans.predict(X)

        for km in range(len(y_kmeans)):
            if y_kmeans[km] == 0:
                word_of_cluster_0.append(label_list[km])
                sent_of_cluster_0.append(inter_sent[km])
                # vocab_of_cluster_0.append(word_bucket[km])
        return word_of_cluster_0, sent_of_cluster_0

    if optimal_k == 2:
        kmeans = KMeans(n_clusters=optimal_k, init="k-means++", max_iter=300, n_init=10, random_state=42)

        kmeans.fit(X)

        y_kmeans = kmeans.predict(X)

        for km in range(len(y_kmeans)):
            if y_kmeans[km] == 0:
                word_of_cluster_0.append(label_list[km])
                sent_of_cluster_0.append(inter_sent[km])
                # vocab_of_cluster_0.append(word_bucket[km])
            if y_kmeans[km] == 1:
                word_of_cluster_1.append(label_list[km])
                sent_of_cluster_1.append(inter_sent[km])
                # vocab_of_cluster_1.append(word_bucket[km])

        return word_of_cluster_0, sent_of_cluster_0, word_of_cluster_1, sent_of_cluster_1

    if optimal_k == 3:
        kmeans = KMeans(n_clusters=optimal_k, init="k-means++", max_iter=300, n_init=10, random_state=42)

        kmeans.fit(X)

        y_kmeans = kmeans.predict(X)

        for km in range(len(y_kmeans)):
            if y_kmeans[km] == 0:
                word_of_cluster_0.append(label_list[km])
                sent_of_cluster_0.append(inter_sent[km])
                # vocab_of_cluster_0.append(word_bucket[km])
            if y_kmeans[km] == 1:
                word_of_cluster_1.append(label_list[km])
                sent_of_cluster_1.append(inter_sent[km])
                # vocab_of_cluster_1.append(word_bucket[km])
            if y_kmeans[km] == 2:
                word_of_cluster_2.append(label_list[km])
                sent_of_cluster_2.append(inter_sent[km])
                # vocab_of_cluster_2.append(word_bucket[km])

        return word_of_cluster_0, sent_of_cluster_0, word_of_cluster_1, sent_of_cluster_1, word_of_cluster_2, sent_of_cluster_2

    if optimal_k == 4:

        kmeans = KMeans(n_clusters=optimal_k, init="k-means++", max_iter=300, n_init=10, random_state=42)

        kmeans.fit(X)

        y_kmeans = kmeans.predict(X)

        for km in range(len(y_kmeans)):
            if y_kmeans[km] == 0:
                word_of_cluster_0.append(label_list[km])
                sent_of_cluster_0.append(inter_sent[km])
                # vocab_of_cluster_0.append(word_bucket[km])
            if y_kmeans[km] == 1:
                word_of_cluster_1.append(label_list[km])
                sent_of_cluster_1.append(inter_sent[km])
                # vocab_of_cluster_1.append(word_bucket[km])
            if y_kmeans[km] == 2:
                word_of_cluster_2.append(label_list[km])
                sent_of_cluster_2.append(inter_sent[km])
                # vocab_of_cluster_2.append(word_bucket[km])
            if y_kmeans[km] == 3:
                word_of_cluster_3.append(label_list[km])
                sent_of_cluster_3.append(inter_sent[km])
                # vocab_of_cluster_3.append(word_bucket[km])
        return word_of_cluster_0, sent_of_cluster_0, word_of_cluster_1, sent_of_cluster_1, word_of_cluster_2,\
               sent_of_cluster_2, word_of_cluster_3, sent_of_cluster_3

    if optimal_k == 5:
        kmeans = KMeans(n_clusters=optimal_k, init="k-means++", max_iter=300, n_init=10, random_state=42)

        kmeans.fit(X)

        y_kmeans = kmeans.predict(X)

        for km in range(len(y_kmeans)):
            if y_kmeans[km] == 0:
                word_of_cluster_0.append(label_list[km])
                sent_of_cluster_0.append(inter_sent[km])
                # vocab_of_cluster_0.append(word_bucket[km])
            if y_kmeans[km] == 1:
                word_of_cluster_1.append(label_list[km])
                sent_of_cluster_1.append(inter_sent[km])
                # vocab_of_cluster_1.append(word_bucket[km])
            if y_kmeans[km] == 2:
                word_of_cluster_2.append(label_list[km])
                sent_of_cluster_2.append(inter_sent[km])
                # vocab_of_cluster_2.append(word_bucket[km])
            if y_kmeans[km] == 3:
                word_of_cluster_3.append(label_list[km])
                sent_of_cluster_3.append(inter_sent[km])
                # vocab_of_cluster_3.append(word_bucket[km])
            if y_kmeans[km] == 4:
                word_of_cluster_4.append(label_list[km])
                sent_of_cluster_4.append(inter_sent[km])
                # vocab_of_cluster_4.append(word_bucket[km])

        return word_of_cluster_0, sent_of_cluster_0, word_of_cluster_1, sent_of_cluster_1, word_of_cluster_2, sent_of_cluster_2, word_of_cluster_3, sent_of_cluster_3, word_of_cluster_4, sent_of_cluster_4
def Sort(sub_li):

    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    sub_li.sort(key = lambda x: x[1], reverse = True)
    return sub_li

# We compute the similarity and obtain the context word for each sentence using this function.

def get_word_vector(word_no, word_list ,static_list ,dynamic_list, threshold, freq):  #outputs a sorted vector list for the word vectors of the target word.
    global rel_list_
    global rel_list
    global cos_dist_
    global frame
    global sort_list
    sort_list = []
    cos_dist_ = []
    new_word_list = []
    global ref
    ref = {}
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    badCHAR = ["##", "sep", "cls", " "]
    for word in word_list:
        wrx = re.sub(r"[\n\t\s]*", "", standardize_text(word))
        new_word_list.append(wrx)
    try:
        for dynm in range(len(dynamic_list)):
            cos_dist = cos(static_list[word_no], dynamic_list[dynm])
            cos_dist_.append(cos_dist.cpu())

        ref = {"Words":new_word_list, "Distance": cos_dist_}
        frame = pd.DataFrame(ref, columns = ["Words", "Distance"])
        frame = frame.sort_values(by = ["Distance"] ,ascending = False)
        frame = frame[frame["Words"].str.contains("|".join(badCHAR)) == False]
        rel_list = frame[frame["Distance"] >= threshold].values.tolist()
        print(rel_list)
        return rel_list

    except IndexError as e:
        pass


#We are writing the output in txt files here.
def write_contextual_output(cluster_number_str, sorted_list, dir_path ,clustering_type,directory_input, inp_str , word_of_cluster_list, sent_of_cluster, n_el_clus_list):
    with open(dir_path + "/static/text/text_" + str(cluster_number_str) + ".txt", "a") as cw:
        cw.write("\n")
        #cw.write("\nThe Target Word is " + "\u0332".join(inp_str.title()) + "\n")
        #cw.write("The word belongs to cluster {}".format(str(int(cluster_number_str) + 1)) + "\n")
        for word in str(word_of_cluster_list[n_el_clus_list]).split():
            if word.isdigit() == True:
                word_label = word
        #cw.write("Label :- " + " " + str(word_of_cluster_list[n_el_clus_list]).title())
        cw.write(f"{inp_str.title()} {word_label}" + "\n")
        cw.write('Instance {} of the word {} in the document belongs to cluster {}'.format("\u0332".join(str(word_label)), "\u0332".join(inp_str.title()), str(int(cluster_number_str) + 1) ))
        cw.write( "\n" + "~~~~~"+"\n")
        cw.write("\n" + "\u0332".join("Corresponding Sentence :- ") +"\n")
        cw.write(str(sent_of_cluster[n_el_clus_list]))
        cw.write("\n")
        cw.write("\n" + "~~~~~" + "\n")
        cw.write("\u0332".join("Context Words [Most similar to the target term] :-") + "\n")
        try:
            for th in sorted_list:
                if th == sorted_list[-1] and th == sorted_list[0]:
                    cw.write(str(th[0]))
                else:
                    cw.write(" , " + str(th[0]))
        except TypeError:
            print('end of sort list')
        cw.write("\n")
        cw.write("############################################################## \n \n" + "\n"*10)
    cw.close()

# TO DO// Sort this mess a bit and make sure you are not writting a lot of if statements.

def get_context(cluster_no, sent_of_cluster, n_el_clus_list, threshold, frequency, word_of_cluster, word_list_refined ,vector_mat_list, vector_list_refined): #This function will run in a loop.
    empty_sent = []
    global summary_list
    summary_list = []
    for word in str(word_of_cluster[n_el_clus_list]).split():
        if word.isdigit() == True:
            label_number = int(word)
    word_label = label_number
    sort_list = get_word_vector(word_label, word_list_refined ,vector_mat_list, vector_list_refined, threshold, frequency)
    # badCHAR = '#'
    # sort_list = [word for word in sort_list if badCHAR not in word]
    write_contextual_output(cluster_no,  sort_list, dir_path ,clustering_type,directory_input, inp_str , word_of_cluster, sent_of_cluster, n_el_clus_list)
    return sort_list


#Get the non target terms and embeddings for the respective cluster.
def get_cluster_vocab_embeddings(list_of_cluster_sentences, n_th_cluster, tokenizer, model, device=device):
    global word_list_
    global word_list_refined
    global vector_list_
    global vector_list_refined

    word_list_, word_list_refined, vector_list_, vector_list_refined = [], [], [], []
    for emd in range(len(list_of_cluster_sentences)):
        try:
            marked_text = "[CLS]" + " " + list_of_cluster_sentences[emd] + " " + "[SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segment_ids = [1]*len(tokenized_text)
            tokens_tensors = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segment_ids])
            with torch.no_grad():
                output = model(tokens_tensors, segments_tensors)
                hidden_states = output[2]
            token_embeddings_ = torch.stack(hidden_states, dim=0).to(device)
            token_embeddings_ = torch.squeeze(token_embeddings_, dim=1).to(device)
            token_embeddings_ = token_embeddings_.permute(1,0,2).to(device)
            token_vecs_sum_ = []
            for token in token_embeddings_:
                sum_vec_ = torch.sum(token[-4:], dim=0).to(device)
                token_vecs_sum_.append(sum_vec_)
            token_vecs = hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)
            for tok in range(len(tokenized_text)):
                if tokenized_text[tok] != inp_str and tokenized_text[tok] != wrt:
                    vector_list_.append(token_vecs_sum_[tok])
                    word_list_.append(tokenized_text[tok])
        except (IndexError, RuntimeError) as e:
            continue

     # word_list_refined is used to avoid repetiotions. Could have worked if i had created a set instead.
    for refi in range(len(word_list_)):
        if word_list_[refi] not in word_list_refined:
            word_list_refined.append(word_list_[refi])
            vector_list_refined.append(vector_list_[refi])
    #print(word_list_refined)
    return  word_list_refined, vector_list_refined #word_list_refined and vector_list_refined are passed to the get_context func above


def tplot(clust, label_int, vector_mat_list,word_list_refined, vector_list_refined):
    global label_
    label_int_list = []
    #locals()["word_list_refined" + str(clust)]
    for j in range(len(word_list_refined)):
        label_int_list.append(j)

    cos_d_ = []
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    label_ = int(label_int)
    print("\n********************************************\n")
    #locals()["vector_list_refined" + str(clust)]
    for dynm in range(len(vector_list_refined)):
        cos_d = cos(vector_mat_list[label_],vector_list_refined[dynm])
        cos_d_.append(cos_d.cpu())

    fer0 = {}
    fer0 = {"Words" : word_list_refined, "Word Int": label_int_list, "Distance": cos_d_}

    frame0 = pd.DataFrame(fer0, columns = ["Words", "Word Int", "Distance"])

    frame0.plot.scatter(x="Word Int", y = "Distance", c='DarkBlue')
    plt.title(f"Threshold Plot for the word {inp_str} and corresponding label {0}")
    plt.savefig(dir_path + "/static/t_plot/tp_plot_" + str(clust) + "_scatter.png")
    plt.close()


    print("The Mean Distance is {} \n********************************************\n".format(frame0["Distance"].mean()))
    print("The Mode Distance is {} \n********************************************\n".format(frame0["Distance"].mode()))
    print("The Median Distance is {} \n********************************************\n".format(frame0["Distance"].median()))
    print("The Max Distance is {} \n********************************************\n".format(frame0["Distance"].max()))
    print("\n********************************************\n")
    print("Kurtosis Information is {} \n********************************************\n".format(frame0.kurtosis(axis=0)))



#Main Model#############################################################################################################

# We call the model for one input sentence hence listing the output will create problems.
#reminder: -ADD DEVICE HERE FOR GPU ACCL.
class DisambModel():
    def __init__(self, bert_model, bert_tokenizer, device):
        super(DisambModel, self).__init__()

        self.model = bert_model
        self.tokenizer = bert_tokenizer
        self.device = device

    def forward(self, input_sentence):
        token_vecs_sum = []
        i_list = []
        # global vector_bucket, word_bucket
        vector_bucket = []
        word_bucket = []

        marked_text = "[CLS]" + " " + input_sentence + " " + "[SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)  # obtain the tokens
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [1] * len(tokenized_text)  # One Hot Encoding with respect to |Vocab|
        tokens_tensors = torch.tensor([indexed_tokens])  # create tensors for each word
        segments_tensor = torch.tensor([segment_ids])  # Sentence Embeddings
        with torch.no_grad():
            output = self.model(tokens_tensors, segments_tensor)  # obtain embeddings from the model
            hidden_states = output[2]
        token_embeddings = torch.stack(hidden_states, dim=0).to(device)  # stack embedings along dim 0
        token_embeddings = torch.squeeze(token_embeddings, dim=1).to(device)  # remove the dim 1 and sequeeze the tensor
        token_embeddings = token_embeddings.permute(1, 0, 2).to(device)  # permute the view along dim 1,0,2

        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0).to(device)
            token_vecs_sum.append(sum_vec)
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs).to(device)

        for i, token_str in enumerate(tokenized_text):
            if token_str == inp_str:
                # print(i, token_str)
                i_list.append(i)

        vector = token_vecs_sum[i_list[0]]
        #

        # create a word list for all the words that don't match the input list
        for tok in range(len(tokenized_text)):
            if tokenized_text != inp_str and tokenized_text[tok] != wrt:
                vector_bucket.append(token_vecs_sum[tok])
                word_bucket.append(tokenized_text[tok])

        return vector, vector_bucket, word_bucket


#Choose N words and Sentences
#Sort the summary list
#write the output
#This function will summarize the results for one cluster
def get_summary_results(cluster_word_list, cluster_sentence_list, sum_dict, cluster_num):
    word_of_cluster_truncated = []
    sent_of_cluster_truncated = []
    sorted_summary_context_list = []

    if len(cluster_sentence_list) > 50:
        random_list = random.sample(range(0, 50), 49) #Selecting random 50 words
        for idx in random_list:
            word_of_cluster_truncated.append(cluster_word_list[idx])
            sent_of_cluster_truncated.append(cluster_sentence_list[idx])
        sorted_summary_dict = dict(sorted(sum_dict.items(), key=operator.itemgetter(1), reverse=True))
        for key, value in sorted_summary_dict.items():
            sorted_summary_context_list.append(key)
    else:
        random_list = random.sample(range(0, len(cluster_sentence_list)), len(cluster_sentence_list))
        for idx in random_list:
            word_of_cluster_truncated.append(cluster_word_list[idx])
            sent_of_cluster_truncated.append(cluster_sentence_list[idx])
        sorted_summary_dict = dict(sorted(sum_dict.items(), key=operator.itemgetter(1), reverse=True))
        for key, value in sorted_summary_dict.items():
            sorted_summary_context_list.append(key)


    with open(dir_path + "/static/summary/summary_text_" + str(cluster_num) + ".txt", "a") as cx:
        cx.write("Top-most context words from the cluster :-")
        cx.write("\n" + "***********************************************" + "\n")
        if len(sorted_summary_context_list) >= 50:
            for th in range(0,50):
                if th == 49:
                    cx.write(sorted_summary_context_list[th])
                else:
                    cx.write(sorted_summary_context_list[th] + " , ")
            cx.write("\n" + "***********************************************" + "\n")
        else:
            for th in range(len(sorted_summary_context_list)):
                if th == 49:
                    cx.write(sorted_summary_context_list[th])
                else:
                    cx.write(sorted_summary_context_list[th] + " , ")
            cx.write("\n" + "***********************************************" + "\n")

    cx.close()

    for word_idx in range(len(word_of_cluster_truncated)):
        with open(dir_path + "/static/summary/summary_text_" + str(cluster_num) + ".txt", "a") as cw:
                # cw.write("The Target Word is " + inp_str.title() + "\n")
                # cw.write("The word {} belongs to cluster {}".format(str(int(cluster_num) + 1)) + "\n")
                for lab in str(word_of_cluster_truncated[word_idx]).split():
                    if lab.isdigit() == True:
                        summary_label = lab
                cw.write(f"{inp_str.title()} {summary_label}" + "\n")
                cw.write(f"Instance {summary_label} of {inp_str.title()} belongs to {str(int(cluster_num) + 1)}" + "\n")
                # cw.write(" :- " + " " + str(word_of_cluster_truncated[word_idx]).title())
                cw.write("\n" + "~~~~~~")
                cw.write("\n" + "Corresponding Sentence :- " + "\n")
                cw.write(str(sent_of_cluster_truncated[word_idx]))
                cw.write("\n")
                cw.write("##############################################################" + "\n")
        cw.close()

###Non-Toolkit related Helpers#########################################################
def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_figure(xs, ys, n):
    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)
    # axis.plot(xs, ys)
    if len(xs) >= 50:
        plt.title(f'Frequency Distribution of Top {n} Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha="right")  # Rotates X-Axis Ticks by 45-degrees
        plt.tick_params(axis='x', which='major', labelsize=4)
        plt.tight_layout()
        plt.plot(xs,ys)
        plt.savefig('static/frq_plot/frq_plot.png', dpi = 800)
        plt.close()
    elif len(xs) == 200:
        plt.title(f'Frequency Distribution of Top {n} Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha="right")  # Rotates X-Axis Ticks by 45-degrees
        plt.tick_params(axis='x', which='major', labelsize=3)
        plt.tight_layout()
        plt.plot(xs, ys)
        plt.savefig('static/frq_plot/frq_plot.png', dpi = 800)
        plt.close()
    else:
        plt.title(f'Frequency Distribution of Top {n} Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha="right")  # Rotates X-Axis Ticks by 45-degrees
        plt.tick_params(axis='x', which='major', labelsize=7)
        plt.tight_layout()
        plt.plot(xs,ys)
        plt.savefig('static/frq_plot/frq_plot.png', dpi = 800)
        plt.close()

######## ROUTING FUNCTIONS###########################################################################

@app.route('/', methods=['POST', 'GET'])
def upload():
    global filename, N
    # We remove the previous results on a new run.
    myfile1 = dir_path + '/static/text/text_0.txt'
    myfile2 = dir_path + '/static/text/text_1.txt'
    myfile3 = dir_path + '/static/text/text_2.txt'
    myfile4 = dir_path + '/static/text/text_3.txt'
    myfile5 = dir_path + '/static/text/text_4.txt'

    #remove the summary results.
    Myfile1 = dir_path + '/static/summary/summary_text_0.txt'
    Myfile2 = dir_path + '/static/summary/summary_text_1.txt'
    Myfile3 = dir_path + '/static/summary/summary_text_2.txt'
    Myfile4 = dir_path + '/static/summary/summary_text_3.txt'
    Myfile5 = dir_path + '/static/summary/summary_text_4.txt'

    #remove plot from memory
    Myplot = dir_path + '/static/frq_plot/frq_plot.png'

    #removing text detaied results
    if os.path.isfile(myfile1):
        os.remove(myfile1)
    if os.path.isfile(myfile2):
        os.remove(myfile2)
    if os.path.isfile(myfile3):
        os.remove(myfile3)
    if os.path.isfile(myfile4):
        os.remove(myfile4)
    if os.path.isfile(myfile5):
        os.remove(myfile5)
    #removing summary files
    if os.path.isfile(Myfile1):
        os.remove(Myfile1)
    if os.path.isfile(Myfile2):
        os.remove(Myfile2)
    if os.path.isfile(Myfile3):
        os.remove(Myfile3)
    if os.path.isfile(Myfile4):
        os.remove(Myfile4)
    if os.path.isfile(Myfile5):
        os.remove(Myfile5)
    #remove plot
    if os.path.isfile(Myplot):
        os.remove(Myplot)
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('index.html')

@app.route('/list', methods=['POST', 'GET'])
def get_list():
    global target_list, frequency_list1, frequency_list2, frequency_list3, pdf_text, wrd
    target_list, token_list, frequency_list1, frequency_list2, frequency_list3 = [], [], [], [], []
    if request.method == 'POST':
        text_list = []
        N = request.form['ip_range']
        str_ = f"{dir_path}/uploads/{filename}"
        if filename.rsplit('.', 1)[1].lower() == 'txt':
            with open(str_, 'r') as f:
                lines = f.readlines()
                for word in lines:
                    text_list.append(word)
            pdf_text = " ".join(text_list)
        else:
            pdf_text = pdf2text(str(str_))
        token_text = word_tokenize(pdf_text)
        filtered_words = [word for word in token_text if word not in stopwords.words('english')]
        unigrams = ngrams(filtered_words, 1)
        counts = dict()
        counts = FreqDist(unigrams)
        sorted_counts = dict(sorted(counts.items(), key=operator.itemgetter(1), reverse=True))
        #print(sorted_counts.items())
        for k in sorted_counts.items():
            token_list.append(standardize_text(k[0][0]))
            frequency_list1.append(k[1])
        tag_list = nltk.pos_tag(token_list)
        #We filter out the nouns here and take the top 50 Nouns.
        t_list = []
        for word_idx in range(len(tag_list)):
            if tag_list[word_idx][1] == "NN":
                t_list.append(tag_list[word_idx][0])
                frequency_list2.append(frequency_list1[word_idx])
        # try:
        for j in range(int(N)-1):
            if t_list[j] not in target_list:
                target_list.append(t_list[j])
                frequency_list3.append(frequency_list2[j])
        print(f'len of target_list is {len(target_list)}')
        print(f'len of t_list is {len(t_list)}')
        create_figure(target_list, frequency_list3,N)


    return render_template('list.html', text = target_list, freq = frequency_list3, zipped = zip(target_list, frequency_list3), url = dir_path + '/static/frq_plot/frq_plot.png', ip_for_range = N)


@app.route('/targetmat', methods = ["GET", "POST"])
def target_mat():
    global target_matrix, vector_bucket, word_bucket, opt_k
    # We remove the previous results on a new run.
    myfile1 = dir_path + '/static/text/text_0.txt'
    myfile2 = dir_path + '/static/text/text_1.txt'
    myfile3 = dir_path + '/static/text/text_2.txt'
    myfile4 = dir_path + '/static/text/text_3.txt'
    myfile5 = dir_path + '/static/text/text_4.txt'

    #remove the summary results.
    Myfile1 = dir_path + '/static/summary/summary_text_0.txt'
    Myfile2 = dir_path + '/static/summary/summary_text_1.txt'
    Myfile3 = dir_path + '/static/summary/summary_text_2.txt'
    Myfile4 = dir_path + '/static/summary/summary_text_3.txt'
    Myfile5 = dir_path + '/static/summary/summary_text_4.txt'

    #remove plot from memory
    Myplot = dir_path + '/static/frq_plot/frq_plot.png'


    #removing text detaied results
    if os.path.isfile(myfile1):
        os.remove(myfile1)
    if os.path.isfile(myfile2):
        os.remove(myfile2)
    if os.path.isfile(myfile3):
        os.remove(myfile3)
    if os.path.isfile(myfile4):
        os.remove(myfile4)
    if os.path.isfile(myfile5):
        os.remove(myfile5)
    #removing summary files
    if os.path.isfile(Myfile1):
        os.remove(Myfile1)
    if os.path.isfile(Myfile2):
        os.remove(Myfile2)
    if os.path.isfile(Myfile3):
        os.remove(Myfile3)
    if os.path.isfile(Myfile4):
        os.remove(Myfile4)
    if os.path.isfile(Myfile5):
        os.remove(Myfile5)
    if request.method == "POST":
        wrd = request.form["target_word"]
        wrd = wrd.lower()
        Frequency = request.form["ip_freq"]
        target_matrix, vector_bucket, word_bucket = get_target_matrix(pdf_text, model, int(Frequency), target_list, wrd)
        try:
            opt_k = get_optimal_k(target_matrix, flag=0)
        except ValueError:
            target_matrix_T = target_matrix.reshape(-1,1)
            opt_k = get_optimal_k(target_matrix_T, flag=1)
    return render_template('mat.html', word = wrd, k = opt_k, url_cluster =  "/static/cluster_plot/elbow_plot.png")



@app.route('/context', methods = ["GET", "POST"])
def context():
    global summary_dict_0, summary_dict_1, summary_dict_2, summary_dict_3, summary_dict_4, summary_dict_5
    global key_list_0, key_list_1, key_list_2, key_list_3, key_list_4, key_list_5
    global value_list_0, value_list_1, value_list_2, value_list_3, value_list_4, value_list_5
    plt.close()
    key_list_0, key_list_1, key_list_2, key_list_3, key_list_4, key_list_5 = [], [], [], [], [], []
    value_list_0, value_list_1, value_list_2, value_list_3, value_list_4, value_list_5 = [], [], [], [], [], []
    summary_dict_0, summary_dict_1, summary_dict_2, summary_dict_3, summary_dict_4, summary_dict_5 = dict(), dict(), dict(),dict(), dict(),dict()

    if request.method == "POST":
        inp_threshold = request.form["thresh"]
        threshold = float(inp_threshold)
        if opt_k == 1 or opt_k > 5:
            word_of_cluster_0, sent_of_cluster_0 = get_clusters(opt_k, target_matrix)
            word_list_refined_0, vector_list_refined_0 = get_cluster_vocab_embeddings(sent_of_cluster_0, 0, bert_tokenizer,
                                                                                      bert_model)
            # Plotting the threshold plot for respective clusters.
            tplot(0, 0, vector_mat_list, word_list_refined_0, vector_list_refined_0)
            for word0 in range(len(word_of_cluster_0)):
                try:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, no_similar_words,
                                                          word_of_cluster_0,
                                                          word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])

                except IndexError as e:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, len(rel_list),
                                                          word_of_cluster_0,
                                                          word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])

            summary_dict_0 = {key: value for key, value in zip(key_list_0, value_list_0)}
            get_summary_results(word_of_cluster_0, sent_of_cluster_0, summary_dict_0, 0)

        if opt_k == 2:
            word_of_cluster_0, sent_of_cluster_0, word_of_cluster_1, sent_of_cluster_1 = get_clusters(opt_k, target_matrix)
            word_list_refined_0, vector_list_refined_0 = get_cluster_vocab_embeddings(sent_of_cluster_0, 0, bert_tokenizer,
                                                                                      bert_model)
            word_list_refined_1, vector_list_refined_1 = get_cluster_vocab_embeddings(sent_of_cluster_1, 1, bert_tokenizer,
                                                                                      bert_model)
            tplot(0, 0, vector_mat_list, word_list_refined_0, vector_list_refined_0)
            tplot(1, 0, vector_mat_list, word_list_refined_1, vector_list_refined_1)

            for word0 in range(len(word_of_cluster_0)):
                try:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, no_similar_words, word_of_cluster_0,
                                word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])

                except IndexError as e:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, len(rel_list
                                                                                                      ), word_of_cluster_0,
                                word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])
                except TypeError:
                    print('')

            summary_dict_0 = {key: value for key, value in zip(key_list_0, value_list_0)}
            get_summary_results(word_of_cluster_0, sent_of_cluster_0, summary_dict_0, 0)

            for word1 in range(len(word_of_cluster_1)):
                try:
                    list_of_context_words_1 = get_context(1, sent_of_cluster_1, word1, threshold, no_similar_words, word_of_cluster_1,
                                word_list_refined_1, vector_mat_list, vector_list_refined_1)
                    for key_ in list_of_context_words_1:
                        key_list_1.append(key_[0])
                        value_list_1.append(key_[1])


                except IndexError as e:
                    list_of_context_words_1 = get_context(1, sent_of_cluster_1, word1, threshold, len(rel_list), word_of_cluster_1,
                                word_list_refined_1, vector_mat_list, vector_list_refined_1)
                    for key_ in list_of_context_words_1:
                        key_list_1.append(key_[0])
                        value_list_1.append(key_[1])
                except TypeError:
                    print('')

            summary_dict_1 = {key: value for key, value in zip(key_list_1, value_list_1)}
            get_summary_results(word_of_cluster_1, sent_of_cluster_1, summary_dict_1, 1)

        if opt_k == 3:
            word_of_cluster_0, sent_of_cluster_0, word_of_cluster_1, sent_of_cluster_1, word_of_cluster_2, \
            sent_of_cluster_2 = get_clusters(opt_k, target_matrix)
            word_list_refined_0, vector_list_refined_0 = get_cluster_vocab_embeddings(sent_of_cluster_0, 0,
                                                                                      bert_tokenizer, bert_model)
            word_list_refined_1, vector_list_refined_1 = get_cluster_vocab_embeddings(sent_of_cluster_1, 1,
                                                                                      bert_tokenizer, bert_model)
            word_list_refined_2, vector_list_refined_2 = get_cluster_vocab_embeddings(sent_of_cluster_2, 2,
                                                                                      bert_tokenizer, bert_model)
            tplot(0, 0, vector_mat_list, word_list_refined_0, vector_list_refined_0)
            tplot(1, 0, vector_mat_list, word_list_refined_1, vector_list_refined_1)
            tplot(2, 0, vector_mat_list, word_list_refined_2, vector_list_refined_2)
            for word0 in range(len(word_of_cluster_0)):
                try:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, no_similar_words,
                                                          word_of_cluster_0,
                                                          word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])

                except IndexError as e:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, len(rel_list),
                                                          word_of_cluster_0,
                                                          word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])
                except TypeError:
                    print('')

            summary_dict_0 = {key: value for key, value in zip(key_list_0, value_list_0)}
            get_summary_results(word_of_cluster_0, sent_of_cluster_0, summary_dict_0, 0)

            for word1 in range(len(word_of_cluster_1)):
                try:
                    list_of_context_words_1 = get_context(1, sent_of_cluster_1, word1, threshold, no_similar_words,
                                                          word_of_cluster_1,
                                                          word_list_refined_1, vector_mat_list, vector_list_refined_1)
                    for key_ in list_of_context_words_1:
                        key_list_1.append(key_[0])
                        value_list_1.append(key_[1])
                except IndexError as e:
                    list_of_context_words_1 = get_context(1, sent_of_cluster_1, word1, threshold, no_similar_words,
                                                          word_of_cluster_1,
                                                          word_list_refined_1, vector_mat_list, vector_list_refined_1)
                    for key_ in list_of_context_words_1:
                        key_list_1.append(key_[0])
                        value_list_1.append(key_[1])
                except TypeError:
                    print('')
            summary_dict_1 = {key: value for key, value in zip(key_list_1, value_list_1)}
            get_summary_results(word_of_cluster_1, sent_of_cluster_1, summary_dict_1, 1)


            for word2 in range(len(word_of_cluster_2)):
                try:
                    list_of_context_words_2 = get_context(2, sent_of_cluster_2, word2, threshold, no_similar_words, word_of_cluster_2,
                                word_list_refined_2, vector_mat_list, vector_list_refined_2)
                    for key_ in list_of_context_words_2:
                        key_list_2.append(key_[0])
                        value_list_2.append(key_[1])

                except IndexError as e:
                    list_of_context_words_2 = get_context(2, sent_of_cluster_2, word2, threshold, no_similar_words, word_of_cluster_2,
                                word_list_refined_2, vector_mat_list, vector_list_refined_2)
                    for key_ in list_of_context_words_2:
                        key_list_2.append(key_[0])
                        value_list_2.append(key_[1])
                except TypeError:
                    print('')
            summary_dict_2 = {key: value for key, value in zip(key_list_2, value_list_2)}
            get_summary_results(word_of_cluster_2, sent_of_cluster_2, summary_dict_2, 2)



        if opt_k == 4:
            # sent_of_cluster_0, word_of_cluster_1, sent_of_cluster_1, word_of_cluster_2, sent_of_cluster_2, \
            # word_of_cluster_3, sent_of_cluster_3 = get_clusters(opt_k, target_matrix)

            word_of_cluster_0, sent_of_cluster_0, word_of_cluster_1, sent_of_cluster_1, word_of_cluster_2, \
            sent_of_cluster_2, word_of_cluster_3, sent_of_cluster_3 = get_clusters(opt_k, target_matrix)

            word_list_refined_0, vector_list_refined_0 = get_cluster_vocab_embeddings(sent_of_cluster_0, 0,
                                                                                      bert_tokenizer, bert_model)
            word_list_refined_1, vector_list_refined_1 = get_cluster_vocab_embeddings(sent_of_cluster_1, 1,
                                                                                      bert_tokenizer, bert_model)
            word_list_refined_2, vector_list_refined_2 = get_cluster_vocab_embeddings(sent_of_cluster_2, 2,
                                                                                      bert_tokenizer, bert_model)
            word_list_refined_3, vector_list_refined_3 = get_cluster_vocab_embeddings(sent_of_cluster_3, 3,
                                                                                      bert_tokenizer, bert_model)

            tplot(0, 0, vector_mat_list, word_list_refined_0, vector_list_refined_0)
            tplot(1, 0, vector_mat_list, word_list_refined_1, vector_list_refined_1)
            tplot(2, 0, vector_mat_list, word_list_refined_2, vector_list_refined_2)
            tplot(3, 0, vector_mat_list, word_list_refined_3, vector_list_refined_3)
            for word0 in range(len(word_of_cluster_0)):
                try:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, no_similar_words,
                                                          word_of_cluster_0,
                                                          word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])

                except IndexError as e:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, len(rel_list),
                                                          word_of_cluster_0,
                                                          word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])
                except TypeError:
                    print('')

            summary_dict_0 = {key: value for key, value in zip(key_list_0, value_list_0)}
            get_summary_results(word_of_cluster_0, sent_of_cluster_0, summary_dict_0, 0)


            for word1 in range(len(word_of_cluster_1)):
                try:
                    list_of_context_words_1 = get_context(1, sent_of_cluster_1, word1, threshold, no_similar_words,
                                                          word_of_cluster_1,
                                                          word_list_refined_1, vector_mat_list, vector_list_refined_1)
                    for key_ in list_of_context_words_1:
                        key_list_1.append(key_[0])
                        value_list_1.append(key_[1])
                except IndexError as e:
                    list_of_context_words_1 = get_context(1, sent_of_cluster_1, word1, threshold, no_similar_words,
                                                          word_of_cluster_1,
                                                          word_list_refined_1, vector_mat_list, vector_list_refined_1)
                    for key_ in list_of_context_words_1:
                        key_list_1.append(key_[0])
                        value_list_1.append(key_[1])
                except TypeError:
                    print('')
            summary_dict_1 = {key: value for key, value in zip(key_list_1, value_list_1)}
            get_summary_results(word_of_cluster_1, sent_of_cluster_1, summary_dict_1, 1)

            for word2 in range(len(word_of_cluster_2)):
                try:
                    list_of_context_words_2 = get_context(2, sent_of_cluster_2, word2, threshold, no_similar_words, word_of_cluster_2,
                                word_list_refined_2, vector_mat_list, vector_list_refined_2)
                    for key_ in list_of_context_words_2:
                        key_list_2.append(key_[0])
                        value_list_2.append(key_[1])

                except IndexError as e:
                    list_of_context_words_2 = get_context(2, sent_of_cluster_2, word2, threshold, no_similar_words, word_of_cluster_2,
                                word_list_refined_2, vector_mat_list, vector_list_refined_2)
                    for key_ in list_of_context_words_2:
                        key_list_2.append(key_[0])
                        value_list_2.append(key_[1])
                except TypeError:
                    print('')
            summary_dict_2 = {key: value for key, value in zip(key_list_2, value_list_2)}
            get_summary_results(word_of_cluster_2, sent_of_cluster_2, summary_dict_2, 2)


            for word3 in range(len(word_of_cluster_3)):
                try:
                    list_of_context_words_3 = get_context(3, sent_of_cluster_3, word3, threshold, no_similar_words, word_of_cluster_3,
                                word_list_refined_3, vector_mat_list, vector_list_refined_3)
                    for key_ in list_of_context_words_3:
                        key_list_3.append(key_[0])
                        value_list_3.append(key_[1])
                except IndexError as e:
                    list_of_context_words_3 = get_context(3, sent_of_cluster_3, word3, threshold, no_similar_words, word_of_cluster_3,
                                word_list_refined_3, vector_mat_list, vector_list_refined_3)
                    for key_ in list_of_context_words_3:
                        key_list_3.append(key_[0])
                        value_list_3.append(key_[1])
                except TypeError:
                    print('')
            summary_dict_3 = {key: value for key, value in zip(key_list_3, value_list_3)}
            get_summary_results(word_of_cluster_3, sent_of_cluster_3, summary_dict_3, 3)


        if opt_k == 5:
            sent_of_cluster_0, word_of_cluster_1, sent_of_cluster_1, word_of_cluster_2, sent_of_cluster_2, \
            word_of_cluster_3, sent_of_cluster_3, word_of_cluster_4, sent_of_cluster_4 = get_clusters(opt_k, target_matrix)
            word_list_refined_0, vector_list_refined_0 = get_cluster_vocab_embeddings(sent_of_cluster_0, 0, bert_tokenizer,
                                                                                      bert_model)
            word_list_refined_1, vector_list_refined_1 = get_cluster_vocab_embeddings(sent_of_cluster_1, 1, bert_tokenizer,
                                                                                      bert_model)
            word_list_refined_2, vector_list_refined_2 = get_cluster_vocab_embeddings(sent_of_cluster_2, 2, bert_tokenizer,
                                                                                      bert_model)
            word_list_refined_3, vector_list_refined_3 = get_cluster_vocab_embeddings(sent_of_cluster_3, 3, bert_tokenizer,
                                                                                      bert_model)
            word_list_refined_4, vector_list_refined_4 = get_cluster_vocab_embeddings(sent_of_cluster_4, 4, bert_tokenizer,
                                                                                      bert_model)

            tplot(0, 0, vector_mat_list, word_list_refined_0, vector_list_refined_0)
            tplot(1, 0, vector_mat_list, word_list_refined_1, vector_list_refined_1)
            tplot(2, 0, vector_mat_list, word_list_refined_2, vector_list_refined_2)
            tplot(3, 0, vector_mat_list, word_list_refined_3, vector_list_refined_3)
            tplot(4, 0, vector_mat_list, word_list_refined_4, vector_list_refined_4)

            for word0 in range(len(word_of_cluster_0)):
                try:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, no_similar_words,
                                                          word_of_cluster_0,
                                                          word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])

                except IndexError as e:
                    list_of_context_words_0 = get_context(0, sent_of_cluster_0, word0, threshold, len(rel_list),
                                                          word_of_cluster_0,
                                                          word_list_refined_0, vector_mat_list, vector_list_refined_0)
                    for key_ in list_of_context_words_0:
                        key_list_0.append(key_[0])
                        value_list_0.append(key_[1])

            summary_dict_0 = {key: value for key, value in zip(key_list_0, value_list_0)}
            get_summary_results(word_of_cluster_0, sent_of_cluster_0, summary_dict_0, 0)


            for word1 in range(len(word_of_cluster_1)):
                try:
                    list_of_context_words_1 = get_context(1, sent_of_cluster_1, word1, threshold, no_similar_words,
                                                          word_of_cluster_1,
                                                          word_list_refined_1, vector_mat_list, vector_list_refined_1)
                    for key_ in list_of_context_words_1:
                        key_list_1.append(key_[0])
                        value_list_1.append(key_[1])
                except IndexError as e:
                    list_of_context_words_1 = get_context(1, sent_of_cluster_1, word1, threshold, no_similar_words,
                                                          word_of_cluster_1,
                                                          word_list_refined_1, vector_mat_list, vector_list_refined_1)
                    for key_ in list_of_context_words_1:
                        key_list_1.append(key_[0])
                        value_list_1.append(key_[1])
            summary_dict_1 = {key: value for key, value in zip(key_list_1, value_list_1)}
            get_summary_results(word_of_cluster_1, sent_of_cluster_1, summary_dict_1,1)

            for word2 in range(len(word_of_cluster_2)):
                try:
                    list_of_context_words_2 = get_context(2, sent_of_cluster_2, word2, threshold, no_similar_words,
                                                          word_of_cluster_2,
                                                          word_list_refined_2, vector_mat_list, vector_list_refined_2)
                    for key_ in list_of_context_words_2:
                        key_list_2.append(key_[0])
                        value_list_2.append(key_[1])

                except IndexError as e:
                    list_of_context_words_2 = get_context(2, sent_of_cluster_2, word2, threshold, no_similar_words,
                                                          word_of_cluster_2,
                                                          word_list_refined_2, vector_mat_list, vector_list_refined_2)
                    for key_ in list_of_context_words_2:
                        key_list_2.append(key_[0])
                        value_list_2.append(key_[1])
            summary_dict_2 = {key: value for key, value in zip(key_list_2, value_list_2)}
            get_summary_results(word_of_cluster_2, sent_of_cluster_2, summary_dict_2, 2)


            for word3 in range(len(word_of_cluster_3)):
                try:
                    list_of_context_words_3 = get_context(3, sent_of_cluster_3, word3, threshold, no_similar_words,
                                                          word_of_cluster_3,
                                                          word_list_refined_3, vector_mat_list, vector_list_refined_3)
                    for key_ in list_of_context_words_3:
                        key_list_3.append(key_[0])
                        value_list_3.append(key_[1])
                except IndexError as e:
                    list_of_context_words_3 = get_context(3, sent_of_cluster_3, word3, threshold, no_similar_words,
                                                          word_of_cluster_3,
                                                          word_list_refined_3, vector_mat_list, vector_list_refined_3)
                    for key_ in list_of_context_words_3:
                        key_list_3.append(key_[0])
                        value_list_3.append(key_[1])
            summary_dict_3 = {key: value for key, value in zip(key_list_3, value_list_3)}
            get_summary_results(word_of_cluster_3, sent_of_cluster_3, summary_dict_3, 3)


            for word4 in range(len(word_of_cluster_4)):
                try:
                    list_of_context_words_4 = get_context(4, sent_of_cluster_4, word4, threshold, no_similar_words, word_of_cluster_4,
                                word_list_refined_4, vector_mat_list, vector_list_refined_4)

                    for key_ in list_of_context_words_4:
                        key_list_4.append(key_[0])
                        value_list_4.append(key_[1])
                except IndexError as e:
                    list_of_context_words_4 = get_context(4, sent_of_cluster_4, word4, threshold, no_similar_words,
                                                          word_of_cluster_4,
                                                          word_list_refined_4, vector_mat_list, vector_list_refined_4)

                    for key_ in list_of_context_words_4:
                        key_list_4.append(key_[0])
                        value_list_4.append(key_[1])
            summary_dict_4 = {key: value for key, value in zip(key_list_4, value_list_4)}
            get_summary_results(word_of_cluster_4, sent_of_cluster_4, summary_dict_4, 4)

    return render_template('context.html',k = opt_k, url_1 ='static/t_plot/tp_plot_0_scatter.png', url_2 ='static/t_plot/tp_plot_1_scatter.png', url_3 ='/static/t_plot/tp_plot_2_scatter.png',
                           url_4 ='/static/t_plot/tp_plot_3_scatter.png',url_5 ='/static/t_plot/tp_plot_4_scatter.png', th = threshold)


@app.route('/result1', methods = ["POST", "GET"])
def result_1():
    with open("static/text/text_0.txt") as f:
        content = f.readlines()
    return render_template('result1.html', content = content)

@app.route('/result2', methods = ["POST", "GET"])
def result_2():
    with open("static/text/text_1.txt") as f:
        content = f.readlines()
    return render_template('result2.html', content = content)


@app.route('/result3', methods = ["POST", "GET"])
def result_3():
    with open("static/text/text_2.txt") as f:
        content = f.readlines()
    return render_template('result3.html', content = content)


@app.route('/result4', methods = ["POST", "GET"])
def result_4():
    with open("static/text/text_3.txt") as f:
        content = f.readlines()
    return render_template('result4.html', content = content)


@app.route('/result5', methods = ["POST", "GET"])
def result_5():
    with open("static/text/text_4.txt") as f:
        content = f.readlines()
    return render_template('result5.html', content = content)

#Summary Results :-
@app.route('/summary_result1', methods = ["POST", "GET"])
def summary_result_1():
    with open("static/summary/summary_text_0.txt") as f:
        content = f.readlines()
    return render_template('summary_result1.html', content = content)

@app.route('/summary_result2', methods = ["POST", "GET"])
def summary_result_2():
    with open("static/summary/summary_text_1.txt") as f:
        content = f.readlines()
    return render_template('summary_result2.html', content = content)

@app.route('/summary_result3', methods = ["POST", "GET"])
def summary_result_3():
    with open("static/summary/summary_text_2.txt") as f:
        content = f.readlines()
    return render_template('summary_result3.html', content = content)

@app.route('/summary_result4', methods = ["POST", "GET"])
def summary_result_4():
    with open("static/summary/summary_text_3.txt") as f:
        content = f.readlines()
    return render_template('summary_result4.html', content = content)

@app.route('/summary_result5', methods = ["POST", "GET"])
def summary_result_5():
    with open("static/summary/summary_text_4.txt") as f:
        content = f.readlines()
    return render_template('summary_result5.html', content = content)

if __name__ == '__main__':
    global N
    dir_path = os.path.dirname(os.path.realpath(__file__))
    directory_input = "elbow"
    clustering_type = "kmeans"
    restricted_size = '500 MB'

    #threshold = 0.48
    no_similar_words = 50
    #Frequency = 300
    bert_model = BertModel.from_pretrained('bert-base-uncased',
                                           output_hidden_states=True,  # Whether the model returns all hidden-states.
                                           )
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = DisambModel(bert_model, bert_tokenizer, device)
    serve(app, host="0.0.0.0", port=5000)
