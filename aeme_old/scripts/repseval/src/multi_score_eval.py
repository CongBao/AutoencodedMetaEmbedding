#! /usr/bin/python -u
"""
Perform evaluations of the word representations using three analogy datasets:
Mikolov (Google + MSRA), SAT, and SemEval.
and various semantic similarity datasets such as WS353, RG, MC, SCWC, RW, MEN.
"""

import numpy
import scipy.stats
import sys
import collections
import argparse
import os
from wordreps import WordReps, get_embedding, cosine, normalize

pkg_dir = os.path.dirname(os.path.abspath(__file__))

__author__ = "Danushka Bollegala"
__licence__ = "BSD"
__version__ = "1.0"


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
    return {"acc": acc}


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
    We consider the set of fourth words in the test dataset as the
    candidate space for the correct answer.
    """
    analogy_file = open(os.path.join(pkg_dir, "../benchmarks/google-analogies.txt"))
    questions = collections.OrderedDict()
    total_questions = {}
    corrects = {}
    while 1:
        line = analogy_file.readline()
        if len(line) == 0:
            break
        if line.startswith(':'):  # This is a label 
            label = line.split(':')[1].strip()
            questions[label] = []
            total_questions[label] = 0
            corrects[label] = 0
        else:
            p = line.strip().split()
            total_questions[label] += 1
            questions[label].append((p[0], p[1], p[2], p[3]))
    analogy_file.close()

    print "== Google Analogy Dataset =="
    print "Total no. of question types =", len(questions) 
    print "Total no. of candidates =", len(cands)
    
    # predict the fourth word for each question.
    count = 1
    for label in questions:
        for (a,b,c,d) in questions[label]:
            print "%d%% (%d / %d)" % ((100 * count) / float(valid_questions), count, valid_questions), "\r", 
            count += 1
            va = get_embedding(a, WR)
            vb = get_embedding(b, WR)
            vc = get_embedding(c, WR)
            x = normalize(vb - va + vc)
            s = numpy.dot(M, x)
            nns = [cands[i] for i in (-s).argsort()[:4]]
            nns = filter(lambda y: y not in [a, b, c], nns)
            if nns[0] == d:
                corrects[label] += 1
    
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
        acc = float(100 * corrects[label]) / float(total_questions[label])
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
        total_questions += 1
        questions.append((p[0], p[1], p[2], p[3]))
    analogy_file.close()

    print "== MSR Analogy Dataset =="
    print "Total no. of questions =", len(questions)
    print "Total no. of candidates =", len(cands)
    
    # predict the fourth word for each question.
    count = 1
    for (a,b,c,d) in questions:
        print "%d / %d" % (count, len(questions)), "\r", 
        count += 1
        # set of candidates for the current question are the fourth
        # words in all questions, except the three words for the current question.
        scores = []
        va = get_embedding(a, WR)
        vb = get_embedding(b, WR)
        vc = get_embedding(c, WR)
        x = normalize(vb - va + vc)
        s = numpy.dot(M, x)
        nns = [cands[i] for i in (-s).argsort()[:4]]
        nns = filter(lambda y: y not in [a, b, c], nns)
        if nns[0] == d:
            corrects += 1
    accuracy = float(corrects) / float(len(questions))
    print "MSR accuracy =", accuracy
    return {"accuracy": accuracy}


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


# def read_candidates(fname):
#     cands = []
#     with open(fname) as F:
#         for line in F:
#             word = line.strip()
#             if word not in cands:
#                 cands.append(word)
#     return cands

# def convert_benchmark_to_lowercase(input_fname, output_fname):
#     F = open(input_fname)
#     G = open(output_fname, 'w')
#     for line in F:
#         G.write("%s" % line.lower())
#     F.close()
#     G.close()
#     pass



def convert_embedding_to_lowercase(input_fname, output_fname, dim):
    """
    Convert all word embeddings to lowercase. We will add the lowercase
    and uppercase versions for a particular word and create a new embedding
    for that word.
    """
    print "Converting...", input_fname
    WR = WordReps()
    WR.read_model(input_fname, dim)
    vects = {}
    z = numpy.zeros(dim, dtype=float)
    for word in WR.vects:
        vects[word.lower()] = WR.vects.get(word, z) + WR.vects.get(word.lower(), z)
    with open(output_fname, 'w') as F:
        for word in vects:
            F.write("%s %s\n" % (word, " ".join([str(x) for x in vects[word]])))
    pass


def get_words_in_benchmarks():
    """
    Get the set of words in benchmarks.
    """
    print "Collecting words from all benchmarks..."
    words = set()
    benchmarks = ["ws", "rg", "mc", "rw", "scws", "men", "simlex"]
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


    with open("../benchmarks/all_words.txt", 'w') as G:
        for word in words:
            G.write("%s\n" % word)
    pass


def get_correlation(dataset_fname, vects, corr_measure):
    """
    Measure the cosine similarities for words in the dataset using their representations 
    given in vects. Next, compute the correlation coefficient. Specify method form
    spearman and pearson.
    """
    ignore_missing = False
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


def evaluate_embeddings(embed_fname, dim, res_fname):
    """
    This function can be used to evaluate an embedding.
    """
    res = {}
    WR = WordReps()
    # We will load vectors only for the words in the benchmarks.
    words = set()
    with open(os.path.join(pkg_dir, "../benchmarks/all_words.txt")) as F:
        for line in F:
            words.add(line.strip())
    WR.read_model(embed_fname, dim, words)

    # semantic similarity benchmarks.
    benchmarks = ["ws", "rg", "mc", "rw", "scws", "men", "simlex"]  
    for bench in benchmarks:
        (corr, sig) = get_correlation(os.path.join(pkg_dir, "../benchmarks/%s_pairs.txt" % bench), WR.vects, "spearman")
        print "%s = %f" % (bench, corr)
        res[bench] = corr

    cands = list(words)
    M = numpy.zeros((len(cands), WR.dim), dtype=numpy.float64)
    for (i,w) in enumerate(cands):
        M[i,:] = normalize(get_embedding(w, WR))


    # word analogy benchmarks.
    res["Google_res"] = eval_Google_Analogies(WR, M, cands)
    res["MSR_res"] = eval_MSR_Analogies(WR, M, cands)
    res["SemEval_res"] = eval_SemEval(WR, "CosAdd")
    res["DiffVec_acc"] = eval_diff_vect(WR)
    #res["SAT_res"] = eval_SAT_Analogies(WR, scoring_method)

    res_file = open(res_fname, 'w')
    res_file.write("#RG, MC, WS, RW, SCWS, MEN, SimLex, sem, syn, total, SemEval, MSR, DiffVec\n")
    res_file.write("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (res["rg"], res["mc"], res["ws"], res["rw"], res["scws"], 
            res["men"], res["simlex"], res["Google_res"]["semantic"], res["Google_res"]["syntactic"], 
            res["Google_res"]["total"], res["SemEval_res"]["acc"], res["MSR_res"]["accuracy"], res["DiffVec_acc"]))
    res_file.close()
    return res


def batch_eval():
    nns = [10, 50, 100, 200, 300, 600]
    comps = [10, 50, 100, 200, 300, 600]
    for nn in nns:
        for comp in comps:
            embed_fname = "../../../work/glove+sg/n=%d+k=%d" % (nn, comp)
            if os.path.exists(embed_fname):
                res_fname = "../work/n=%d+k=%d.csv" % (nn, comp)
                print "Evaluating nns = %d, comps = %d" % (nn, comp)
                evaluate_embeddings(embed_fname, comp, res_fname)
    pass


def main():
    """
    Catch the arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate pre-trained word representation on semantic similarity and word analogy tasks.")
    parser.add_argument("-dim", type=int, help="specify the dimensionality of the word representations as an integer.")
    parser.add_argument("-input", type=str, help="specify the input file from which to read word representations.")
    parser.add_argument("-output", type=str, help="specify the csv formatted output file to which the evaluation result to be written.")
    args = parser.parse_args()
    
    if args.input and args.dim and args.output:
        evaluate_embeddings(args.input, args.dim, args.output)
    else:
        print parser.print_help()
        sys.stderr.write("Invalid option for mode. It must be either lex or ana\n")
    pass


if __name__ == "__main__":
    main()
    #batch_eval()
    #get_words_in_benchmarks()
    #convert_embedding_to_lowercase("../../../../embeddings/glove.42B.300d.txt", "../../../../embeddings/lowercase/glove.42B.300d.lower.txt", 300)
    #convert_embedding_to_lowercase("../../../../embeddings/HLBL+100", "../../../../embeddings/lowercase/HLBL+100.lower.txt", 100)
    #convert_embedding_to_lowercase("../../../../embeddings/Huang+50", "../../../../embeddings/lowercase/Huang+50.lower.txt", 50)
    #convert_embedding_to_lowercase("../../../../embeddings/CW+200", "../../../../embeddings/lowercase/CW+200.lower.txt", 200)
    #convert_benchmark_to_lowercase("../benchmarks/google-analogies.txt", "../benchmarks/lower")

    
    
   
