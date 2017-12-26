# README #

## Requirements ##
- Python 2.7 or above
- numpy
- scipy
- Perl 5 or above (for SemEval evaluation)
- sklearn (for short-text classification evaluation)

## Execution ##

### Evaluating using Semantic Similarity Benchmarks ###

To evaluate on semantic similarity benchmarks, go to the src directory and execute
```
python eval.py -m lex -d noOfDimensions -i wordRepsFile -o result.csv
```

* -m option specifies the mode of operation.
    'lex' to evaluate on semantic similarity benchmarks.
    'ana' to evaluate on word analogy benchmarks.
    'rel' to evaluate on relation classification benchmarks.
    'txt' to evaluate on short text classification benchmarks.
    You can combine multiple evaluations using a comma. For example, -m=lex,ana,rel,txt will perform all evaluations in one go.

* -d option is used to specify the dimensionality of the word representations.

* -i specifies the input file from which we will read word representations. The format of this file is as follows.
Each line represents the word vector for a particular word. First element in each line is the word and subsequent elements
(in total the number of columns corresponding to the dimensionality specified using the -d option) contains the value of
each dimension of the representation.

* -o is the name of the output file into which we will write the Pearson correlation coefficients and their significance values.
This is a csv file.

* There are several ways to compute the relational similarity between two pairs of words such as CosAdd, CosMult, PairDiff, and CosSub. This tool uses CosAdd as the default method. You can try different methods, which are also implemented in the tool. See source code for more details. 

#### The following semantic similarity benchmarks are available in this suite. ####

| Dataset   | word pairs | Publication/distribution |
| --------  | ---------- | ------------------------ |
| Word Similarity 353 (WS) | 353 | [Link](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) |
| Miller-Charles (MC) | 28 | MILLER, G. A. et CHARLES, W. G. (1991). Contextual correlates of semantic similarity. Language and Cognitive Processes, 6(1):1-28. |
| Rubenstein-Goodenough (RG) | 65 | RUBENSTEIN, H. et GOODENOUGH, J. B. (1965). Contextual correlates of synonymy. Communications of the ACM, 8(10):627-633.|
| MEN | 3000 | [Link](http://clic.cimec.unitn.it/~elia.bruni/MEN) |
| Stanford Contextual Word Similarity (SCWC) | 2003 | [Link](http://nlp.stanford.edu/pubs/HuangACL12.pdf) |
|Rare Words (RW) | 2034 | [Link](http://nlp.stanford.edu/~lmthang/data/papers/conll13_morpho.pdf) |
| SimLex | 999 | [Link](http://www.cl.cam.ac.uk/~fh295/simlex.html) |

#### The following word analogy benchmarks are available in this suite. ####
| Dataset   | instances | Publication/distribution |
| --------  | ---------- | ------------------------ |
| SAT | 374 questions | [Link](https://aclweb.org/aclwiki/index.php?title=Similarity_(State_of_the_art)) |
| SemEval 2012 Task 2 | 79 paradigms | [Link](https://sites.google.com/site/semeval2012task2/)|
| Google dataset | 19558 questions (syntactic + semantic analogies)| [Link](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)|
| MSR dataset | 7999 syntactic questions | [Link](http://www.marekrei.com/blog/linguistic-regularities-word-representations/)|

* There are several ways to compute the relational similarity between two pairs of words such as CosAdd, CosMult, PairDiff, and CosSub. This tool uses CosAdd as the default method. You can try different methods, which are also implemented in the tool. See source code for more details. 

#### The following relation classification benchmarks are available in this suite. ####
| Dataset   | word pairs | Publication/distribution |
| --------  | ---------- | ------------------------ |
| DiffVec | 12473 pairs | [Link](http://www.aclweb.org/anthology/P16-1158)|

#### The following short-text classification benchmarks are available in this suite. ####
| Dataset   | word pairs | Publication/distribution |
| --------  | ---------- | ------------------------ |
| TR (Stanford Sentiment Treebank) | train = 6001, test = 1821 | [Link](http://nlp.stanford.edu/sentiment/treebank.html)|
| MR (Movie Review Dataset) | train =, 8530 test = 2132 | [Link](https://www.cs.cornell.edu/people/pabo/movie-review-data/)|
| CR (Customer Review Dataset) | train = 1196, test = 298| [Link](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) |
| SUBJ (Subjectivity Dataset) | train = 8000, test = 2000| [Link](https://www.cs.cornell.edu/people/pabo/movie-review-data/)|

### Psycholinguistic score prediction. ###
We use the input word embeddings in a neural network (containing a single hidden layer of 100 neurons and relu activation) to learn a regression model (no activation in the output layer). We use randomly selected 80% of words from MRC database and ANEW dataset to train a regression model for valence, arousal, dominance, concreteness and imageability.  We then measure the Pearson correlation between the predicted ratings and human ratings and report the corresponding correlation coefficients.
See Section 4.2 of [this](https://arxiv.org/abs/1709.01186#) paper for further details regarding this setting.


