# Dimensionality Reduction and Optimization of the Glove Words Database using Principal Component Analysis and Birch Clustering

## Navigation

Optimization2.ipynb contains the most recent code, all that is necessary for simulating the process on one's machine. 
The commented out lines of code depict the code that must only be run once, and several different methods of clustering and dimensionality reduction are included for the comparative analysis stage, though these are not necessary. 
The code also produces the results. 

The 10k and 20k word clusters and database were too large to be uploaded to github, so the same process was repeated with 1k words and enclosed in WordFiles. For finding the clusters with 10k words, attach a pull request or something and I can try and send these results! 
The clustering biases are still maintained, however, as shown by gloveWordsBirchClusters.txt

The clustering biases identified in Word2Vec are not displayed on Github, but still apply. This project was primarily run on GloVe, but the dimensionality reduction has also been proved to work with Word2Vec, and I am working on expanding it to latest lstm architecture. 

## Background

From the video game Semantle to ChatGPT, Natural Language Processing (NLP) systems are used in effective communication between humans and computers. They are necessary for sorting documents, creating a thesaurus, machine translation, and computer interpretations of words. Using an NLP, AIs are then able to continue to predict future words by determining the words with the closest semantic similarity to make a reasonable sounding sentence. GloVe (Global Vectors for Word Representation) is such a system that improves on the commonly used Word2Vec (Mikolov et al.), and in this project I will refine GloVe. 

Optimizing the bag-of-words model, GloVe uses probability distributions and a co-occurrence matrix to find how relevant words are to each other (Pennington et al). GloVe can also associate words like ‘rapid’ and ‘rapidly’ and ‘ice’ and ‘water.’ Using cosine-distance based vectors, GloVe can create true mathematical equations, such as the following example with defining King, Man, Women, and Queen as three-dimensional vectors.


> King - Man + Woman = Queen

> (Man, Royal, Dragon): (1, 1, -1) - (1, -1, -1) + (1, -1, -1) = (-1, 1, -1)

## Goals 

My project refined GloVe, so that it uses less dimensions. English word representations use 300 dimensions (Mikolov et al.) regardless of the method, as do other languages like Mandarin (Gong 2017). This number seems to be the sweet spot between enough information and speed of training the AI. My project optimized this number to reduce both runtime and memory requirements. 

In the process, my work offered an algorithm for optimization that AIs can incorporate as they learn from a word database. By reducing dimensions, more memory space clears up to store additional word complexities and meanings, and creates a potential for even more, specialized words to be incorporated. In addition, with a smaller dataset size, functions on these words have less to iterate through, which means less runtime. This would be like if ChatGPT had no lag and was able to communicate instantly. 

Due to the large database size of over 10,000 words, I utilized streamlined algorithms for a reasonable termination time. As a result, I evaluated several methods of reduction and analysis and select the fastest solution. I also had to keep in mind additional constraints such as quality of word reductions and more. 
My work has determined that 149 dimensions is enough to store all the data in the 300 dimensions, and that 33 dimensions store the majority of data. 
 
## Dimensionality Reduction 
  
For the dimensionality reduction process, I used Principal Component Analysis (PCA) to move high dimensional data to low dimensions while still preserving relative distances. For example, the highlighted cluster in the associated graph contains the days of the week, revealing how PCA kept their similarities in even two dimensions. It removed dimensions that were less relevant through projections onto lower planes and helped reduce dataset size. The example of ‘king, man, woman, queen,’ includes a vector with three dimensions, man, dragon, fruit. Fruit was an unnecessary dimension that did not fit any of the words, and thus could be removed with PCA. 

PCA had the advantages of a low runtime as well as being a linear model. After an O(n) time for computation of eigenvectors and values for future use, it needed simply O(1) for retrieval, because selecting the first k eigenvectors determined by PCA is all that is necessary for the new matrix. PCA is a linear method, as well, which was desirable because I wanted to project the words onto a lower and lower number of planes after identifying these planes, thus allowing me to get a reduction in dimensionality sizes.

Other alternatives included:

t-SNE: Unfeasible due to extreme runtime. It took over 5 minutes for just 100 words, and had a significantly high complexity
NMF: Did not support negative values. Normalization procedures, coupled with removal of negatives would cause data loss
Forward Selection/Backwards elimination: Want to store all information, not just ‘most important’ features
 

## Clusters

Coming up with the idea to use clusters for defining a word was a significant idea. Essentially, how does one check if a word has the same definition in a computer that doesn't store "word definitions"? Well, this can be determined with clustering algorithms. Other methods like triangulations, calculating distances, or word subtractions do not work, are more difficult, or functionally do the same thing as the clustering algorithms. 
 
I confirmed that words had the same meaning regardless of the dimensionality by using clustering algorithms. If words are clustered in 300 dimensions, such as ‘love’ and ‘story,’ they should still be clustered in lower dimensions to prove that no data is lost because the words mean the same thing, even if the words are more compressed together. In this context, a cluster is defined as a set of words such that the definition of one word can be inferred based on the cluster, such as the days of the week. Clustering algorithms needed to support cosine distances, based on GloVe, and need to have even sizes and be efficient.

I comparatively analyzed clustering methods such as Birch Clustering, Agglomerative, DBScan, k-means, and a BFS to see which ones would be optimal. Agglomerative and DBScan clustering failed by identifying merely high-density regions of the grid and clustering all those words, leaving the rest of the words alone and without a relative meaning. K-means was not fast enough and could not be used. I also implemented a cosine distance feature to Birch in the sklearn library that I was then able to use, thus addressing Birch’s weakness, so Birch was fast, efficient, and had relatively even cluster sizes. The breadth-first search was slower because it had to repeatedly find the closest word, with O(n^2 ), which on a large dataset took too much time. 

However, clustering algorithms tend to preserve one definition or one cluster per word – if a word has two clusters, both clusters would merge, or the word would be reduced to just one meaning. To fix this, I used BFS. The intuition behind a breadth-first search is to find words that have multiple meanings and thus correlations; so that two clusters can still be connected by the same root node. Thus, coupling the BFS with Birch allowed me to both preserve the relationships between words like ‘car’, ‘trunk’, and ‘elephant’, and to efficiently determine clusters that contain related words, enabling the preservation of the complexities of words.

To confirm that clusters would be preserved in lower dimensions, I used a HashMap that associated keys of relative “centers” of clusters, and then checked to see if the keys were still associated with either the closest or second closest of their initial cluster. I first tried with just the closest, but realized that this may not always be optimal, so I extended it to second-closest as well.

If this remained true for almost every cluster, then the dimension would satisfy the condition. After that, I used a binary search that could use these checks to find the lowest possible dimension that would still preserve the meanings of words. The binary search would iterate through all possible dimensions in an efficient manner. Then, the data would be simplified using PCA to a lower dimension, and Birch would create clusters. These clusters would then be checked for associations with the Hashmaps. All of this code is open source on my Github, here! 

# Results

Through my research, I found that all data stored in 300 dimensions can be conserved in just 247. Removing the 30 outliers that were significantly higher or lower, I was also able to store all this data in half the number of dimensions- 149. Using the median, thus storing most of the data, 33 dimensions, just 10% of the total size of data was all that was necessary.

In addition, as demonstrated by the graph, the location at which the most data is stored per dimension occurs at 87 dimensions. This can be found by locating the point at which the derivative is smallest, or thus the location where the second derivative is zero. At this location more than the majority of data is stored and the minimum number of dimensions is necessary. 

This also had huge runtime ramifications; leading to a 4.19 second loss going from 5.49 seconds to 1.3, thus alleviating significant amounts of lag in generating sentences. Running the ‘find_most_similar’ function iteratively with the updated and the previous dataset (something that simply runs through all words), gave this significant runtime reduction, and simulating an O(n^3 ) function gave 1 minute instead of 10. 

Thus, my work has resulted in significant lowering of runtime and memory that vastly improves the database and all future implementations – from ChatGPT to Semantle. 

## Clustering biases

There were also, unfortunately, biases in how words were associated with each other. Examples of these are below:
Pakistan Iran Iraqi Danger
Senate Congress Building White Republican
Obama Democrat
Bush Clinton president America
killed soldiers dead arrested
church government

When an entire race is sorted with danger, or when an African American president is excluded from the rest, or when the word ‘white’ is sorted with Congress, bias is consistently ingrained in AIs due to the training dataset that is used and might be flawed. Regardless of the reason for this bias, when users interact with an AI and see these associations, these biases might spread or connote things to the user that are unfriendly and exclusive.

These biases were not seen in the 300-dimension clustering, but they are visible now because other clusters have moved away. As words vary minorly in their position, newer associations that were less visible arise.

To fix this, I believe that using a variety of datasets from multicultural literature or children’s picture books from a variety of sources would help. might help reverse some of these biases. I am currently working on this and have reached out to the creators of GloVe to fix these biases.

## Future improvements 

In the future, I would like to expand this research into more specific corpuses, especially through the lens of ChatGPT. Right now, ChatGPT can say something about everything, but it is flawed in more specialized applications. Ask it to solve a complex math problem, and it stops at solving some trivial algebra, for example. With a specialized corpus of language, however, ChatGPT could become more refined. Specifically, I am working on a chemistry-themed database, using elements and different names of compounds as a major part of the training dataset. This allows AIs and chatbots to complete more meaningful discussions and approach the truth, rather than sentences that seem like the truth. My paper also lays out a process for identifying the right number of dimensions for these smaller themed corpuses to retain information.

English is also not the only language, and I would like to see if more ameliorations can be identified in other languages with less words, such as French. 
I would also like to address the issue of bias. Currently, I am re-training the dataset using a picture-book based vocabulary, and then predicting more advanced word locations. Then, given this location, I plan to check it with the actual location of the word in the larger dataset and then continue adding in words slowly with consistent human supervision to avoid this.  


