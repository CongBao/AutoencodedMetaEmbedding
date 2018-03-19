# coidng: utf-8

"""
Reads in the SAT questions created by Peter Turney.
Reurns a list of objects of the type
{QID:str, ANS:ID, QUESTION:(A,B), CHOICES:[(a,b),(c,d),(e,f),(g,h),(i,j)]}
Where, ID is the answer of the ID (1-5), QUESTION contains the
pair given in the question, CHOICES are the five choices given
in the question.
"""

import os

pkg_dir = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.path.join(pkg_dir, "../benchmarks/SAT-package-V3.txt")
# contains a unique ID assigned to deduplicated sat word-pairs.
WORD_PAIR_ID_FILE = os.path.join(pkg_dir, "../benchmarks/sat_pairs")


class SAT:

    def __init__(self):
        self.SAT_QUESTIONS_FILE = QUESTIONS_FILE
        self.SAT_WORD_PAIR_ID_FILE = WORD_PAIR_ID_FILE
        self.id2pair = {}
        self.pair2id = {}
        #self.load_word_pairs_ids()
        pass
    

    def load_word_pairs_ids(self):
        if self.SAT_WORD_PAIR_ID_FILE:
            F = open(self.SAT_WORD_PAIR_ID_FILE, "r")
            for line in F:
                p = line.strip().split(",")
                id = int(p[0])
                first = p[1].strip()
                second = p[2].strip()
                self.id2pair[id] = (first, second)
                self.pair2id[(first, second)] = id
                pass
            F.close()
            pass
        pass


    def get_word_pair(self, id):
        return(self.id2pair[id])
    

    def get_id(self, first, second):
        return(self.pair2id[(first, second)])
    

    def getChoicePair(self, line):
        """
        Read the word pair and pos info.
        """
        p = line.strip().split()
        postags = p[2].split(':')
        return ({'word':p[0], 'pos':postags[0]}, {'word':p[1], 'pos':postags[1]})
    
            
    def getQuestions(self):
        F = open(self.SAT_QUESTIONS_FILE, "r")
        L = []
        ANS_IDs = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4}
        line = F.readline()
        while(line):
            H = {}
            H["QID"] = line.strip()
            H["QUESTION"] = self.getChoicePair(F.readline())
            H["CHOICES"]=[]
            for i in range(0, 5):
                line = F.readline()
                if not line.startswith("no choice"):
                    H["CHOICES"].append(self.getChoicePair(line))
                pass
            line=F.readline()
            ID = ANS_IDs[line.strip()]
            H["ANS"] = ID
            L.append(H)
            line = F.readline() #blank line
            line = F.readline() #next question
            pass
        F.close()
        return(L)
    pass


def dump_word_pairs(fname):
    """
    For each word-pair in SAT questions and aswers, assign
    an ID (starting from 1) and write to a file.
    """
    sat_pairs = open(fname, "w")
    S = SAT()
    id = 1
    L = S.getQuestions()
    N = len(L)
    H = {}
    duplicates = 0
    for i in range(0,N):
        (A,B) = L[i]["QUESTION"]
        A = A['word']
        B = B['word']
        if (A,B) not in H:
            sat_pairs.write("%d # %s # %s\n" % (id, A, B))
            id += 1
            H[(A,B)] = 1
        else:
            print A, B
            duplicates += 1
        for (C,D) in L[i]["CHOICES"]:
            C = C['word']
            D = D['word']
            if (C,D) not in H:
                #sat_pairs.write("%d # %s # %s\n" % (id,C,D))
                id += 1
                H[(C,D)] = 1
            else:
                print C, D
                duplicates += 1
    sat_pairs.close()
    pass


def main():
    """
    prints SAT questions.
    """
    S = SAT()
    L = S.getQuestions()
    print L[0]
    print len(L)
    pass

if __name__ == "__main__":
    main()
    #dump_word_pairs("../work/pairs")
    pass
            
          
            
        
        

