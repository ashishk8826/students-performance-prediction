# students-performance-prediction
This project ' Student’s Performance Prediction Using Nature Inspired Algorithms’ is a Research cum 
development project. This project provides several techniques which tries to improve K-means clustering, 
some techniques are like “Salp swarm Optimization”, "Chaotic Salp swarm", “Grey Wolf Optimization”, Etc.
There are certain drawbacks in K-Means. In genetic algorithm, the term of Premature convergence means that 
a population for an optimization problem converged too early, resulting in being less effective, the intuition 
is like looking at stars - the points should have consistent spacing between them, Initial no of clusters need to 
be defined, there is only local search on K-Means and not global search.
In this term, specifically certain types of techniques are used i.e., Salp swarm, Chaotic Salp Swarm, Grey Wolf 
Optimization, etc. 
The main objective is to improve the optimization of Student’s Performance Prediction and that is done
through global search i.e., update position and velocity of cluster particles till accurate centroid positions are 
known. 
Updating personal best of each particle and global best of system. Here global best means best set of centroids 
and each particle of swarm represents different set of centroids.
SSE is used as a fitness function in improving K-Means algo which is compared with personal best of a particle 
to find out minimum score.



Introduction
1.1 General Introduction
Clustering or cluster analysis is grouping of many objects in a way by which we can see that 
objects in the same group are called a cluster and they are also we can visualize that more 
similar in some sense to each other than to those in other groups clusters. There are mainly 
4 types of clustering – • Centroid Clustering:- This type of clustering is one of the most common 
methodologies or type of clustering used in cluster analysis. In the centroid cluster 
analysis, you can choose the different numbers of clusters that you want to classify 
or You can make group of clusters and you can choose different clusters for 
classifying. For example, if you’re an animal store owner and you may want to 
choose to Categorise your customer list by people who bought lion and/or mouse 
products.
• Density Clustering:- Density clustering is the type of clustering which groups the 
data points by the category how densely populated they are. To group data which are 
closely related data points so in this type of clustering this density clustering 
algorithm is used to maximum advantage to understand that the murkier the data 
points...the more related they are. To determine this, the density clustering will select 
a random point and then using density clustering it start measuring the distance. • Distribution Clustering:- Distribution Clustering is the type of clustering which 
helps in identifying the probability for the clustering and by this it shows that point 
belongs to a cluster. Around each possible centroid distribution clustering algorithm 
defines the density distributions for each of the cluster, quantitating the probability 
of possession based on those distributions the algorithm to increase efficiency of the 
typical of the distributions to best represent the data.
• Connectivity Clustering:- Unlike the other clustering techniques of clustering 
analysis as reviewed above, connectivity clustering at first determines each data point 
as its retaining cluster. The key assumption of this technique is that points nearer to 
each one is more related.



1.2 Problem Statement
Predicting Student’s Academic Performance Using Salp Swarm and Chaotic Salp Swarm 
Algorithms and Grey Wolf Optimization.
1.3 Significance
Although K-means bunch algorithmic rule is straightforward and well-liked, it's an
elementary disadvantage of falling into native optima that rely on the indiscriminately 
generated initial centre of mass values. optimisation algorithms square measure 
documented for his or her ability to guide repetitive computation in sorting out 
international optima. They conjointly speed up the bunch method by achieving early 
convergence. up to date optimisation algorithms impressed by biology, together with the 
Wolf, Firefly, Cuckoo, Bat and emmet algorithms, simulate swarm behaviour during 
which peers square measure attracted whereas steering towards a worldwide objective. 
it's found that these bio-inspired algorithms have their own virtues and will be logically 
integrated into K-means bunch to avoid native optima throughout iteration to 
convergence. during this paper, the constructs of the combination of bio-inspired 
optimisation strategies into K-means bunch square measure given. The extended 
versions of bunch algorithms integrated with bio-inspired optimisation strategies 
manufacture improved results. Experiments square measure conducted to validate the 
advantages of the projected approach.


1.4 Empirical Study
In this project, primarily K-Means cluster used that may be a variety of reiterative cluster. 
it's a technique of vector division, originally from signal process, that aims to partition n 
observations into k clusters during which every observation belongs to the cluster with 
the closest mean (cluster centres or cluster centroid), serving as an image of the cluster. 
This leads to a partitioning of the information house into Voronoi cells. it's standard for 
cluster analysis in data processing. k-means cluster minimizes within-cluster variances 
(squared geometrician distances), however not regular geometrician distances, which 
might be the harder. Nature galvanized rule may be a terribly active space of analysis is 
that the style of nature-inspired metaheuristics. several recent metaheuristics, 
particularly biological process computation-based algorithms, area unit galvanized by 
natural systems. Nature acts as a supply of ideas, mechanisms and principles for planning 
of artificial computing systems to touch upon complicated process issues.


1.5 Brief Description Solution Approach
In this project, primarily K-Means clustering used which is a type of iterative clustering.
It is a method of vector quantization, originally from signal processing, that aims 
to partition n observations into k clusters in which each observation belongs to 
the cluster with the nearest mean (cluster centres or cluster centroid), serving as a 
prototype of the cluster. This results in a partitioning of the data space into Voronoi cells. 
It is popular for cluster analysis in data mining. k-means clustering minimizes withincluster variances (squared Euclidean distances), but not regular Euclidean distances, 
which would be the more difficult.
Nature Inspired Algorithm is a very active area of research is the design of natureinspired metaheuristics. Many recent metaheuristics, especially evolutionary 
computation-based algorithms, are inspired by natural systems. Nature acts as a source 
of concepts, mechanisms and principles for designing of artificial computing systems to 
deal with complex computational problems. Such metaheuristics include simulated 
annealing, evolutionary algorithms, ant colony optimization and particle swarm 
optimization. A large number of more recent metaphor-inspired metaheuristics have 
started to attract criticism in the research community for hiding their lack of novelty 
behind an elaborate metaphor.
1.6 Comparison of existing approaches to the problem framed
The integration of the k-Means algorithm with each one of the most recent natureinspired algorithms to overcome the drawback of the K-means algorithm which is falling 
in the local optima and to maximize clusters integrity. The various researches done in 
the areas of K-mean algorithm and it is concluded that when K-mean is used in 
combination with the other optimization technique like Salp swarm algorithm, Grey wolf 
optimization algorithm, Chaotic Salp optimization algorithm etc, All the papers have 
their own advantages and drawbacks. We have tried to overcome some of their gaps in 
our work. We were successful in implementing several optimizing techniques on our 
dataset. We were able to deal with the highly imbalanced dataset by applying optimizing 
techniques like k-means using Chaotic Particle swarm optimization algorithm, k-means 
using Chaotic Salp swarm optimization algorithm in a single report. The results prove 
that optimization techniques when used with appropriate algorithm provide better 
accuracy than individual optimization techniques and it is highly successful in giving 
much better results as compared to the results shown in the papers presented.



DATA SET Preprocessing
Data Quality Assessment
Because information is commonly taken from multiple sources that area unit unremarkably not too 
reliable which too in several formats, principally time is consumed in addressing information 
quality problems once engaged on a machine learning downside. it's merely chimerical to expect 
that the info is going to be excellent. There are also some issues because of human error, limitations 
of mensuration devices, or flaws within the information assortment method. Let’s reconsider many 
of them and ways to wear down them :
Missing values
There is no missing price within the dataset that is taken from UCI Repository.
Inconsistent values
There is no inconsistent price within the dataset that is taken from UCI Repository.
Duplicate values
There is no Duplicate price within the dataset that is taken from UCI Repository.
Feature Aggregation
Feature Aggregations area unit performed thus on take the collective values so as to place the info 
in higher perspective. This led to reduction of memory consumption and time interval. Aggregation 
offer U.S. with a high-level read of {the information the info the information} because the 
behaviour of teams or aggregates is a lot of stable than individual data objects.
Feature Sampling
In this dataset there's no demand od feature sampling. as a result of Sampling may be a quite 
common technique for choosing a set of the dataset that we tend to area unit analysing. In most 
cases, operating with the entire dataset will end up to be too big-ticket considering the memory and 
time constraints. employing a sampling rule will facilitate U.S. cut back the scale of the dataset to 
a degree wherever we are able to use an improved, however dearer, machine learning rule.
spatiality Reduction
Dimensionality reduction aims to cut back the number of options - however not just by choosing a 
sample of options from the feature-set, that are a few things else — Feature set choice or just 
Feature choice. Conceptually, dimension refers to the number of geometric planes the dataset lies 
29 | P a g e
in, that may well be high most so it can't be pictured with pen and paper. a lot of the amount of 
such planes, a lot of is that the quality of the dataset. In the dataset that have taken we've to perform 
the spatiality reduction as a result of in our dataset there are a unit thirty-three attributes and lots 
of them is of no use thus, we've to get rid of it.
Feature cryptograph
As mentioned before, the full purpose of knowledge pre-processing is to cypher the info so as to 
bring it to such a state that the machine currently understands it. This dataset is structural thus 
there's no downside of understanding by machine.
Non-Functional Requirements: 
The non-functional requirements represent requirements that should work to assist the project 
to accomplish its goal.
The non-functional requirements for the current system are:
Interface:- The project constructed is console-based. The output is displayed on a laptop 
screen display the clusters of the input data set. 
Performance :The developed system must be able to group the given data into clusters based 
on the K means algorithm. 
Scalability: The system must provide as many options like changing headlines in the XML 
file and then the changes occur in the clusters as well.
Software Requirements
• Library and Software used –
1. Pyswarm library
2. K-Mean’s library
3. Matplotlib
4. Pandas
5. Numpy
6. Sklearn
7. Jupyter Notebook and Spyder
Hardware Requirements:
Computer: - a pair of gigacycle minimum, multi-core processor
Memory (RAM): - a minimum of 2GB, ideally higher
Hard disk space: - a minimum of 128 GB
Processor: - 64-bit
3.5 Solution Approach
Improved results of Student Performance Prediction and optimized exploitation three ways: -
• Salp Swarm optimization: - SSA is one among the random population-based algorithms 
instructed by Mir Jalili et al. (2017) in 2017. independent agency simulates the swarming 
mechanism of salps once search in oceans. In serious oceans, salps typically form a swarm 
referred to as urochordate chain. In independent agency algorithmic program, the leader is that 
the urochordate at the front of chain and no matter remains of salps square measure known as 
followers. prefer to different swarm-based techniques, the position of salps is outlined in 
Associate in Nursing s-dimensional search house, wherever s is that the variety of variables of 
a given drawback. Therefore, the position of all salps is hold on in a very two-dimensional 
matrix known as z. it's additionally assumed that there's a food supply known as P within the 
search house because the swarm’s target. 
Exploration and Exploitation: - Exploitation is outlined because the ability of the optimisation 
algorithmic program to enhance the most effective resolution it's found to date by looking a
little space of the answer. whereas in PSO, it implies that all the particles meet to constant peak 
of the target perform and don't leave it. in contrast to the exploration characteristics of 
Associate in Nursing algorithmic program describes the power of the algorithmic program to 
depart this peak and look for higher resolution. 
● Chaotic Salp swarm :- The planned Chaotic urochordate Swarm algorithmic program 
(CSSA) is applied on fourteen unimodal and multimodal benchmarks by optimisation issues 
and twenty benchmark datasets. 10 totally different chaotic maps square measure utilized to 
reinforce the convergence rate and ensuing preciseness. Simulation results showed that the 
planned CSSA may be a promising algorithmic program. Also, the results reveal the aptitude 
of CSSA find Associate in Nursing optimum feature set, that maximizes the classification 
accuracy, whereas minimizing the number of elect options. Population primarily based metaheuristic algorithms share varies benefits embrace quantifiability, simplicity and process time 
reduction. However, these algorithms have 2 main disadvantages:- recession in native optima 
and low convergence rate. a technique to beat these issues and enhance the performance of 
meta-heuristic algorithms is to deploy the chaos theory. Chaotic maps square measure used 
rather than random numbers in PSO primarily based algorithms to reinforce the convergence. 
during this manner, authors introduce a chaotic primarily based independent agency (CSSA), 
that replaces random variables with chaotic ones. CSSA uses chaotic maps to regulate the worth 
of the second constant.
• Grey Wolf Optimization: - The GWO algorithmic program that relies on social behaviour
of grey wolves is planned by Mirjalili in 2014. The searching behaviour of wolves and therefore 
the social hierarchy between wolf’s square measure sculpturesque mathematically to style the 
GWO algorithmic program. The modelling of the algorithmic program primarily consists of 4 
steps: social hierarchy, skirting prey, searching and offensive prey. Social Hierarchy: There 
square measure four kinds of wolves like alpha (α), beta (β), delta (δ) and omega (ω) in social 
hierarchy of mathematically model of algorithmic program. per the algorithmic program, the 
most effective 3 positions of the population square measure diagrammatic by alpha, beta and 
delta wolves severally. the remainder of the wolf’s square measure accepted to be omega. The 
searching organization is radio-controlled by alpha, beta and delta. and therefore, the wolves 
that assumed as omega follow these 3 leader wolves. Encircling Prey: The grey wolves 
surround the victim throughout the searching. It shows the updated position of every member 
in population throughout skirting.
