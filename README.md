This repo is based on Andrea Gigli's suggestion of using bipartite complex networks to build a recommendation engine (https://www.slideshare.net/andrgig/recommendation-systems-in-banking-and-financial-services). 
The data comes from the Online Retail set of UCI (http://archive.ics.uci.edu/ml/datasets/online+retail).

I tried my best in commenting the code, however I strongly encourage you to go through his slides before diving into the code. 
After building the network according to his methodology I run community detection algorithms to find products that are often purchased together. 
These clusters of products can then be used to make suitable suggestions to a customer after he purchases (or puts on his online basket) his very first product. 
Although I have used only a very small subset of the relative UCI repository, the clusters tend to contain similar products either in category (e.g. same products in different colors) or area of use (e.g. kitchen-related stuff). 

