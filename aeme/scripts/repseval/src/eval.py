#!/usr/bin/env python
"""
Perform evaluations of the word representations using three analogy datasets:
Mikolov (Google + MSRA), SAT, and SemEval.
and various semantic similarity datasets such as WS353, RG, MC, SCWC, RW, MEN.
"""

import numpy
import sys
import collections
import argparse
import os
import scipy.stats


from wordreps import WordReps, get_embedding, cosine, normalize

pkg_dir = os.path.dirname(os.path.abspath(__file__))
VERBOSE = False

__author__ = "Danushka Bollegala"
__licence__ = "BSD"
__version__ = "2.0"


def eval_SemEval(WR, method):
    """
    Answer SemEval questions. 
    """
    from semeval import SemEval
    S = SemEval(os.path.join(pkg_dir, "../benchmarks/semeval"))
    total_accuracy = 0
    print "Total no. of instances in SemEval =", len(S.data)
    for Q in S.data:
        scores = []
        for (first, second) in Q["wpairs"]:
            val = 0
            for (p_first, p_second) in Q["paradigms"]:
                va = get_embedding(first, WR)
                vb = get_embedding(second, WR)
                vc = get_embedding(p_first, WR)
                vd = get_embedding(p_second, WR)
                val += scoring_formula(va, vb, vc, vd, method)
            val /= float(len(Q["paradigms"]))
            scores.append(((first, second), val))

        # sort the scores and write to a file. 
        scores.sort(lambda x, y: -1 if x[1] > y[1] else 1)
        score_fname = os.path.join(pkg_dir, "../work/semeval/%s.txt" % Q["filename"])
        score_file = open(score_fname, 'w')
        for ((first, second), score) in scores:
            score_file.write('%f "%s:%s"\n' % (score, first, second))
        score_file.close()
        total_accuracy += S.get_accuracy(score_fname, Q["filename"])
    acc = total_accuracy / float(len(S.data))
    print "SemEval Accuracy =", acc
    return acc


def eval_SAT_Analogies(WR, method):
    """
    Solve SAT word analogy questions using the vectors. 
    """
    from sat import SAT
    S = SAT()
    questions = S.getQuestions()
    corrects = total = skipped = 0
    for Q in questions:
        total += 1
        (q_first, q_second) = Q['QUESTION']
        if q_first['word'] in WR.vects and q_second['word'] in WR.vects:
            va = get_embedding(q_first['word'], WR)
            vb = get_embedding(q_second['word'], WR)
            max_sim = -100
            max_cand = -100
            for (i, (c_first, c_second)) in enumerate(Q["CHOICES"]):
                sim = 0
                if c_first['word'] in WR.vects and c_second['word'] in WR.vects:
                    vc = get_embedding(c_first['word'], WR)
                    vd = get_embedding(c_second['word'], WR)
                    sim = scoring_formula(va, vb, vc, vd, method)
                    if max_sim < sim:
                        max_sim = sim 
                        max_cand = i
            if max_cand == Q['ANS']:
                corrects += 1
        else:
            skipped += 1
    acc = float(100 * corrects) / float(total)
    coverage = 100.0 - (float(100 * skipped) / float(total))
    print "SAT Accuracy = %f (%d / %d)" % (acc, corrects, total)
    print "Qustion coverage = %f (skipped = %d)" % (coverage, skipped) 
    return {"acc":acc, "coverage":coverage}


def eval_diff_vect(WR):
    """
    Uses the DiffVect dataset for performing 1-NN relation classification.
    We will use PairDiff to create a vector for a word-pair and then measure the similarity
    between the target pair and the reamining word-pairs in the dataset.
    If the 1-NN has the same relation label as the target pair, then we consider it to
    be a correct match. We compute accuracy = correct_matches / total_instances.
    """
    analogy_file = open(os.path.join(pkg_dir, "../benchmarks/diff-vec"))
    relation = {}
    pairs = []
    label = ""
    while 1:
        line = analogy_file.readline()
        if len(line) == 0:
            break
        if line.startswith(':'):  # This is a label 
            label = line.split(':')[1].strip()
        else:
            p = line.strip().split()
            (a, b) = p
            pairs.append((a, b))
            relation[(a, b)] = label
    analogy_file.close()
    n = len(pairs)
    M = numpy.zeros((n, WR.dim), dtype=numpy.float64)
    for (i, (a, b)) in enumerate(pairs):
        M[i, :] = normalize(get_embedding(a, WR) - get_embedding(b, WR))
    S = numpy.dot(M, M.T)
    preds = (-S).argsort()[:,1]
    corrects = sum([relation[pairs[i]] == relation[pairs[preds[i]]] for i in range(n)])
    accuracy = float(100 * corrects) / float(n)
    print "DiffVec Accuracy =", accuracy
    return accuracy


def eval_Google_Analogies(WR, M, cands):
    """
    Evaluate the accuracy of the learnt vectors on the analogy task. 
    """
    analogy_file = open(os.path.join(pkg_dir, "../benchmarks/google-analogies.txt"))
    questions = collections.OrderedDict()
    total_questions = {}
    corrects = {}
    while 1:
        line = analogy_file.readline().lower()
        if len(line) == 0:
            break
        if line.startswith(':'):  # This is a label 
            label = line.split(':')[1].strip()
            questions[label] = []
            total_questions[label] = 0
            corrects[label] = 0
        else:
            p = line.strip().split()
            if all(word in WR.vects for word in p):
                total_questions[label] += 1
                questions[label].append((p[0], p[1], p[2], p[3]))
    analogy_file.close()

    valid_questions = sum([len(questions[label]) for label in questions])

    print "== Google Analogy Dataset =="
    print "Total no. of question types =", len(questions) 
    print "Total no. of candidates =", len(cands)
    print "Total no. of valid questions =", valid_questions
    
    # predict the fourth word for each question.
    count = 1
    correct_count = 0
    total = 0
    for label in questions:
        for (a,b,c,d) in questions[label]:
            total += 1
            acc = float(correct_count * 100) / float(total)
            print "%d%% (%d / %d) acc = %2.2f%%" % ((100 * count) / float(valid_questions), count, valid_questions, acc), "\r", 
            count += 1
            va = get_embedding(a, WR)
            vb = get_embedding(b, WR)
            vc = get_embedding(c, WR)
            x = normalize(vb - va + vc)
            s = numpy.dot(M, x)
            nns = [cands[i] for i in numpy.argsort(-s)]
            nns = filter(lambda y: y not in [a, b, c], nns)
            #print "Question: ", a, b, c, d, numpy.sum(x)
            if d == nns[0]:
                corrects[label] += 1
                correct_count += 1
            
    
    # Compute accuracy
    n = semantic_total = syntactic_total = semantic_corrects = syntactic_corrects = 0
    for label in total_questions:
        n += total_questions[label]
        if label.startswith("gram"):
            syntactic_total += total_questions[label]
            syntactic_corrects += corrects[label]
        else:
            semantic_total += total_questions[label]
            semantic_corrects += corrects[label]
    print "Percentage of questions attempted = %f (%d / %d)" % ((100 * valid_questions) /float(n), valid_questions, n)
    for label in questions:
        if total_questions[label] != 0:
            acc = float(100 * corrects[label]) / float(total_questions[label])
        else:
            acc = 0

        print "%s = %f (correct = %d, attempted = %d, total = %d)" % (
            label, acc, corrects[label], len(questions[label]), total_questions[label])
    semantic_accuracy = float(100 * semantic_corrects) / float(semantic_total)
    syntactic_accuracy = float(100 * syntactic_corrects) / float(syntactic_total)
    total_corrects = semantic_corrects + syntactic_corrects
    accuracy = float(100 * total_corrects) / float(n)
    print "Semantic Accuracy =", semantic_accuracy 
    print "Syntactic Accuracy =", syntactic_accuracy
    print "Total accuracy =", accuracy
    return {"semantic": semantic_accuracy, "syntactic":syntactic_accuracy, "total":accuracy}


def eval_MSR_Analogies(WR, M, cands):
    """
    Evaluate the accuracy of the learnt vectors on the analogy task using MSR dataset. 
    We consider the set of fourth words in the test dataset as the
    candidate space for the correct answer.
    """
    analogy_file = open(os.path.join(pkg_dir, "../benchmarks/msr-analogies.txt"))
    questions = []
    total_questions = 0
    corrects = 0
    while 1:
        line = analogy_file.readline()
        if len(line) == 0:
            break
        p = line.strip().split()
        if all(word in WR.vects for word in p):
            total_questions += 1
            questions.append((p[0], p[1], p[2], p[3]))
    analogy_file.close()

    print "== MSR Analogy Dataset =="
    print "Total no. of valid questions =", len(questions)
    print "Total no. of candidates =", len(cands)
    
    # predict the fourth word for each question.
    count = 1
    for (a,b,c,d) in questions:
        print "%d / %d" % (count, len(questions)), "\r", 
        count += 1
        # set of candidates for the current question are the fourth
        # words in all questions, except the three words for the current question.
        va = get_embedding(a, WR)
        vb = get_embedding(b, WR)
        vc = get_embedding(c, WR)
        x = normalize(vb - va + vc)
        s = numpy.dot(M, x)
        nns = [cands[i] for i in (-s).argsort()]
        nns = filter(lambda y: y not in [a, b, c], nns)
        if d == nns[0]:
            corrects += 1
    accuracy = float(100 * corrects) / float(len(questions))
    print "MSR accuracy =", accuracy
    return accuracy


def eval_short_text_classification(bench_path, WR):
    """
    Evaluate the word embeddings by measuring their accuracy on short text classification tasks.
    Each instance is represented as a BOW, and we compute the centroid of all word embeddings
    for the words in an instance. Next, we train a binary logistic regression classifier.
    We represent test instances in the same manner and report test accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    #print "Short text classification for: ", bench_path
    train_X = []
    train_y = []
    with open("%s/train" % bench_path) as train_file:
        for line in train_file:
            p = line.strip().split()
            train_y.append(int(p[0]))
            train_X.append(get_text_instance(p[1:], WR))

    test_X = []
    test_y = []        
    with open("%s/test" % bench_path) as test_file:
        for line in test_file:
            p = line.strip().split()
            test_y.append(int(p[0]))
            test_X.append(get_text_instance(p[1:], WR))

    train_X = numpy.array(train_X)
    train_y = numpy.array(train_y)
    test_X = numpy.array(test_X)
    test_y = numpy.array(test_y)

    #print "Total no. of train instances", train_X.shape[0]
    #print "Total no. of test instances ", test_X.shape[0]

    LR = LogisticRegression(penalty='l2', C=1.0)
    LR.fit(train_X, train_y)
    #print numpy.mean([t == test_y[i] for (i,t) in enumerate(LR.predict(test_X))])
    acc = 100 * LR.score(test_X, test_y)
    print "Accuracy for %s = %f" % (bench_path.split('/')[-1], acc)
    return acc


def get_text_instance(txt, WR):
    """
    Add all the word embeddings in the txt and return the centroid.
    """
    x = numpy.zeros(WR.dim, dtype=numpy.float)
    for token in txt:
        p = token.split(':')
        if len(p) != 2:
            continue
        feat = p[0].strip()
        feat_val = float(p[1])
        if feat in WR.vects:
            x += feat_val * WR.vects[feat]
    return x


############### SCORING FORMULAS ###################################################
def scoring_formula(va, vb, vc, vd, method):
    """
    Call different scoring formulas. 
    """
    t = numpy.copy(vb)
    vb = vc
    vc = t

    if method == "CosSub":
        return subt_cos(va, vb, vc, vd)
    elif method == "PairDiff":
        return PairDiff(va, vb, vc, vd)
    elif method == "CosMult":
        return mult_cos(va, vb, vc, vd)
    elif method == "CosAdd":
        return add_cos(va, vb, vc, vd)
    elif method == "DomFunc":
        return domain_funct(va, vb, vc, vd)
    elif method == "EleMult":
        return elementwise_multiplication(va, vb, vc, vd)
    else:
        raise ValueError


def mult_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    log(cos(vb, vd)) + log(cos(vc,vd)) - log(cos(va,vd))
    """
    first = (1.0 + cosine(vb, vd)) / 2.0
    second = (1.0 + cosine(vc, vd)) / 2.0
    third = (1.0 + cosine(va, vd)) / 2.0
    score = numpy.log(first) + numpy.log(second) - numpy.log(third)
    return score


def add_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(vb - va + vc, vd)
    """
    x = normalize(vb - va + vc)
    return cosine(x, vd)


def domain_funct(va, vb, vc, vd):
    """
    Uses the Formula proposed by Turney in Domain and Function paper.
    """
    return numpy.sqrt((1.0 + cosine(va, vc))/2.0 * (1.0 + cosine(vb, vd))/2.0)


def elementwise_multiplication(va, vb, vc, vd):
    """
    Represent the first word-pair by the elementwise multiplication of va and vb.
    Do the same for vc and vd. Finally measure the cosine similarity between the
    two resultant vectors.
    """
    return cosine(va * vb, vc * vd)


def subt_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(va - vc, vb - vd)
    """
    return cosine(normalize(va - vc), normalize(vb - vd))


def PairDiff(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(vd - vc, vb - va)
    """
    return cosine(normalize(vd - vc), normalize(vb - va))
####################################################################################


def get_words_in_benchmarks():
    """
    Get the set of words in benchmarks.
    """
    print "Collecting words from all benchmarks..."
    words = set()
    benchmarks = ["ws", "rg", "mc", "rw", "scws", "men", "simlex", "behavior"]
    for bench in benchmarks:
        with open("../benchmarks/%s_pairs.txt" % bench) as F:
            for line in F:
                p = line.strip().split()
                words.add(p[0])
                words.add(p[1])

    # Get words in Google analogies.
    analogy_file = open("../benchmarks/google-analogies.txt")
    while 1:
        line = analogy_file.readline()
        if len(line) == 0:
            break
        if line.startswith(':'):  # This is a label 
            label = line.split(':')[1].strip()
        else:
            p = line.strip().split()
            words.add(p[0])
            words.add(p[1])
            words.add(p[2])
            words.add(p[3])
    analogy_file.close()

    # Get words in MSR analogies.
    analogy_file = open("../benchmarks/msr-analogies.txt")
    while 1:
        line = analogy_file.readline()
        p = line.strip().split()
        if len(p) == 0:
            break
        words.add(p[0])
        words.add(p[1])
        words.add(p[2])
        words.add(p[3])
    analogy_file.close()

    # Get words in DiffVect dataset.
    diff_vect_file = open("../benchmarks/diff-vec")
    for line in diff_vect_file:
        if not line.startswith(':'):
            p = line.strip().split()
            words.add(p[0])
            words.add(p[1])
    diff_vect_file.close()

    # Get SAT words.
    from sat import SAT
    S = SAT()
    questions = S.getQuestions()
    for Q in questions:
        (q_first, q_second) = Q['QUESTION']
        words.add(q_first['word'])
        words.add(q_second['word'])
        for (i, (c_first, c_second)) in enumerate(Q["CHOICES"]):
            words.add(c_first['word'])
            words.add(c_second['word'])

    # Get SemEval words.
    from semeval import SemEval
    S = SemEval("../benchmarks/semeval")
    for Q in S.data:
        for (first, second) in Q["wpairs"]:
            words.add(first)
            words.add(second)
            for (p_first, p_second) in Q["paradigms"]:
                words.add(p_first)
                words.add(p_second)

    # Get text classification datasets.
    for dataset in ["CR", "MR", "SUBJ", "TR"]:
        words = words.union(get_words_short_text("../benchmarks/%s/train" % dataset))
        words = words.union(get_words_short_text("../benchmarks/%s/test" % dataset))

    # Get Psycholinguistic ratings datasets.
    with open("../benchmarks/psycho.csv") as data_file:
        for line in data_file:
            p = line.strip().split(',')
            word = p[0].strip()
            words.add(word)

    with open("../benchmarks/all_words.txt", 'w') as G:
        for word in words:
            G.write("%s\n" % word)
    pass

def get_words_short_text(fname):
    """
    Return the set of words in train/test files in short-text classification datasets
    """
    feats = set()
    with open(fname) as F:
        for line in F:
            p = line.strip().split()
            for ent in p[1:]:
                feats.add(ent.split(':')[0])
    return feats


def get_correlation(dataset_fname, vects, corr_measure):
    """
    Measure the cosine similarities for words in the dataset using their representations 
    given in vects. Next, compute the correlation coefficient. Specify method form
    spearman and pearson.
    """
    ignore_missing = True
    global VERBOSE
    if VERBOSE:
        if ignore_missing:
            sys.stderr.write("Ignoring missing pairs\n")
        else:
            sys.stderr.write("Not ignoring missing pairs\n")
    mcFile = open(dataset_fname)
    mcPairs = {}
    mcWords = set()
    for line in mcFile:
        p = line.strip().split()
        mcPairs[(p[0], p[1])] = float(p[2])
        mcWords.add(p[0])
        mcWords.add(p[1])
    mcFile.close()
    #print "Total no. of unique words in the dataset =", len(mcWords)
    found = mcWords.intersection(set(vects.keys()))
    #print "Total no. of words found =", len(found)
    missing = []
    for x in mcWords:
        if x not in vects:
            missing.append(x)
    human = []
    computed = []
    found_pairs = False
    missing_count = 0
    for wp in mcPairs:
        (x, y) = wp
        if (x in missing or y in missing):
            missing_count += 1
            if ignore_missing:
                continue
            else:
                comp = 0
        else:
            found_pairs = True
            comp = cosine(vects[x], vects[y])
        rating = mcPairs[wp]
        human.append(rating) 
        computed.append(comp)       
        #print "%s, %s, %f, %f" % (x, y, rating, comp)
    if VERBOSE:    
        sys.stderr.write("Missing pairs = %d (out of %d)\n" % (missing_count, len(mcPairs)))

    if found_pairs is False:
        #print "No pairs were scored!"
        return (0, 0)
    if corr_measure == "pearson":
        return scipy.stats.pearsonr(computed, human)
    elif corr_measure == "spearman":
        return scipy.stats.spearmanr(computed, human)
    else:
        raise ValueError
    pass


def evaluate_embeddings(embed_fname, dim, res_fname, mode):
    """
    This function can be used to evaluate an embedding.
    """
    res = []
    WR = WordReps()
    # We will load vectors only for the words in the benchmarks.
    words = set()
    with open(os.path.join(pkg_dir, "../benchmarks/all_words.txt")) as F:
        for line in F:
            words.add(line.strip())
    WR.read_model(embed_fname, dim, words, case_sensitive=True)

    if "lex" in mode or "all" in mode:
        # semantic similarity benchmarks.
        benchmarks = ["ws", "rg", "mc", "rw", "scws", "men", "simlex", "behavior"]  
        for bench in benchmarks:
            (corr, sig) = get_correlation(os.path.join(pkg_dir, "../benchmarks/%s_pairs.txt" % bench), WR.vects, "spearman")
            print "%s = %f" % (bench, corr)
            res.append((bench, corr))

    cands = list(words)
    M = numpy.zeros((len(cands), WR.dim), dtype=numpy.float64)
    for (i,w) in enumerate(cands):
        M[i,:] = normalize(get_embedding(w, WR))    

    if "ana" in mode or "all" in mode:    
        # word analogy benchmarks.
        google = eval_Google_Analogies(WR, M, cands)
        res.append(("Google-semantic", google["semantic"]))
        res.append(("Google-syntactic", google["syntactic"]))
        res.append(("Google-total", google["total"]))
        res.append(("MSR", eval_MSR_Analogies(WR, M, cands)))
        res.append(("SemEval", eval_SemEval(WR, "CosAdd")))
        res.append(("SAT", eval_SAT_Analogies(WR, "CosAdd")["acc"]))

    if "rel" in mode or "all" in mode:
        res.append(("DiffVec", eval_diff_vect(WR)))

    if "txt" in mode or "all" in mode:    
        # short text classification benchmarks.
        res.append(("TR", eval_short_text_classification("../benchmarks/TR", WR)))
        res.append(("MR", eval_short_text_classification("../benchmarks/MR", WR)))
        res.append(("CR", eval_short_text_classification("../benchmarks/CR", WR)))
        res.append(("SUBJ", eval_short_text_classification("../benchmarks/SUBJ", WR)))

    if "psy" in mode or "all" in mode:
        psy_corr = get_psycho(WR)
        for rating_type in psy_corr:
            res.append((rating_type, psy_corr[rating_type]))

    res_file = open(res_fname, 'w')
    res_file.write("# %s\n" % ", ".join([ent[0] for ent in res]))
    res_file.write("%s\n" % ", ".join([str(ent[1]) for ent in res]))
    res_file.close()
    return res

def get_psycho(WR):
    """
    Predict psycholinguistic ratings using the word embeddings.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score
    ratings = {"Valence":[], "Arousal":[], "Dominance":[], "Concreteness":[], "Imageability":[]}
    words = []
    X = []
    res = {}
    with open("../benchmarks/psycho.csv") as data_file:
        data_file.readline()
        for line in data_file:
            p = line.strip().split(',')
            word = p[0].strip()
            words.append(word)
            ratings["Valence"].append(float(p[3]))
            ratings["Arousal"].append(float(p[4]))
            ratings["Dominance"].append(float(p[5]))
            ratings["Concreteness"].append(float(p[6]))
            ratings["Imageability"].append(float(p[7]))
            X.append(WR.vects.get(word, numpy.zeros(WR.dim)))

    n = len(words)
    test_end = n / 5
    for rating_type in ratings:
        M = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)
        M.fit(X[test_end:], ratings[rating_type][test_end:])
        res[rating_type] = scipy.stats.pearsonr(ratings[rating_type][0:test_end], M.predict(X[0:test_end]))[0]
    return res


def show_neighbors(fname, dim, nns):
    from sklearn.neighbors import NearestNeighbors
    WR = WordReps()
    sys.stdout.write("Loading word embeddings from %s\n" % fname)
    sys.stdout.flush()
    WR.read_model(fname, dim)
    M = numpy.zeros((len(WR.vects), dim), dtype=numpy.float)
    wids = {}
    for (i,w) in enumerate(WR.vocab):
        M[i,:] = WR.vects[w]
        wids[w] = i
    sys.stdout.write("Computing nearest neighbours... ")
    sys.stdout.flush()
    nbrs = NearestNeighbors(n_neighbors=nns, algorithm='ball_tree').fit(M)
    distances, indices = nbrs.kneighbors(M)
    sys.stdout.write("Done\n")
    sys.stdout.flush()
    while 1:
        sys.stdout.write("\nEnter query:")
        query = sys.stdin.readline().strip()
        sys.stdout.write("Showing nearest neighbours for = %s\n" % query)
        if query in wids:
            for nn in indices[wids[query], 1:]:
                print WR.vocab[nn]
    pass
    


def conf_interval(r, num):
    stderr = 1.0 / numpy.sqrt(num - 3)
    delta = 1.96 * stderr
    lower = numpy.tanh(numpy.arctanh(r) - delta)
    upper = numpy.tanh(numpy.arctanh(r) + delta)
    print "lower %.6f upper %.6f" % (lower, upper)


def random_shuffle(input_fname, output_fname):
    """
    Randomly suffle lines in a file except for the header.
    """
    import random
    F = open(input_fname)
    head = F.readline()
    L = [line for line in F]
    random.shuffle(L)
    F.close()
    G = open(output_fname, 'w')
    G.write(head)
    for line in L:
        G.write(line)
    G.close()
    pass

def main():
    """
    Catch the arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate pre-trained word representation on semantic similarity, word analogy, relation classification, short-text classification and psycholinguistic score prediction tasks.")
    parser.add_argument("-dim", type=int, help="specify the dimensionality of the word representations as an integer.")
    parser.add_argument("-input", type=str, help="specify the input file from which to read word representations.")
    parser.add_argument("-output", type=str, help="specify the csv formatted output file to which the evaluation result to be written.")
    parser.add_argument("-mode", type=str, 
        help="mode of operation. lex for semantic similarity, ana for analogy, rel for relation classification, txt for text classification and psy for psycholingustic score prediction.\
              Use a comma to concatenate multiple options. nns for nearest neighbours in interactive mode, and all for all tasks.")
    args = parser.parse_args()

    if args.mode:
        mode = args.mode.split(',')   
        print "Modes of operations =", mode
        if args.input and args.dim and mode[0] == "nns":
            show_neighbors(args.input, args.dim, 11)
        elif args.input and args.dim and args.output:
            evaluate_embeddings(args.input, args.dim, args.output, mode)
    else:
        sys.stderr.write(parser.print_help())
    pass


if __name__ == "__main__":
    #get_words_in_benchmarks()
    #random_shuffle("../benchmarks/psycho.csv", "../benchmarks/random_psycho.csv")
    main()
   
    
    
    
   
