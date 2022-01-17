# Aligner.py
# aligns reference file and hypothesis file
# for MT metric preparation

# arg1: ref file (1 sentence per line, tokenized)
# arg2: hypotheses file (tokenized)
# arg3: output file (aligned file)

import argparse
import re
import sys
import warnings
import jiwer
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu,SmoothingFunction

class Align:

    def __init__(self,refs,hyps,refs_rest,hyp_rest,prev,nrofhyps,beamsize,sentbleu,sentbleulist,hypslower):
        self.refs=refs
        self.hyps=hyps
        self.refs_rest=refs_rest
        self.hyp_rest=hyp_rest
        self.prev=prev
        self.nrofhyps=nrofhyps
        self.beam=beamsize
        self.sentbleu=sentbleu
        self.sentbleulist=sentbleulist
        self.hypslower=hypslower
        if len(sentbleulist)>0:
            self.sentbleuavg=sum(sentbleulist)/len(sentbleulist)
        else:
            self.sentbleuavg=0
        if hasattr(self,"bleu"):
            pass
        elif (len(hyps)>0): 
            try: self.bleu=corpus_bleu(refs,hypslower,smoothing_function=SmoothingFunction().method4) 
            except: self.bleu=0
        else: self.bleu=0
        self.remainingwords=len(hyp_rest)
        self.complete=nrofhyps/(len(refs)+len(refs_rest))

    def print(self):
#        print('Align',file=sys.stderr)
        print('Refs')
        for ref in self.refs:
            print(ref,file=sys.stderr)
        print('Hyps')
        for hyp in self.hyps:
            print(hyp,file=sys.stderr)
#        print('ED',self.ed,file=sys.stderr)
#        print('Bleu',self.bleu,file=sys.stderr)
        if arguments.wer:
            print('1-WER',self.bleu)
        else:
            print('Bleu',self.bleu)
#        print('SentbleuAvg',self.sentbleuavg,file=sys.stderr)
        print('Nr of hyps',self.nrofhyps,file=sys.stderr)
#        print('Remaining words',self.remainingwords,file=sys.stderr)

    def printOutput(self,outfile):
        for hyp in self.hyps:
            line=' '.join(hyp)
            print(line,file=outfile)
        #remaining=self.remainingwords
        #print(' '.join(self.hyp_rest),file=outfile)
 
        
    def addRemainingWords(self):
        newRef=self.refs_rest[0]
        lowNewRef=[[x.lower() for x in newRef]]
        hypo=self.hyp_rest.copy()
        hyplower=[x.lower() for x in hypo]
        if arguments.wer:
            try: wer=jiwer.wer(' ',join(lowNewRef[0]),' ',join(hyplower))
            except: wer=1
            bleu=1-wer
        else:
            try: bleu=sentence_bleu(lowNewRef,hyplower,smoothing_function=SmoothingFunction().method7)
            except: bleu=0
        if re.search(r'[\.\?\!\,]$',' '.join(hypo)): 
            bleu=bleu * punctuation_weight
        newAlign=Align(self.refs+[lowNewRef],self.hyps+[hypo],[],[],self,self.nrofhyps+1,self.beam,bleu,self.sentbleulist+[bleu],self.hypslower+[hyplower])
        return newAlign
        

    def expand(self):
        refs_rest=self.refs_rest.copy()
        newRef=refs_rest.pop(0)
        lowNewRef=[[x.lower() for x in newRef]]
        lowNewRefString=' '.join(lowNewRef[0])
        lengthRef=len(newRef)
        toplist=[]
        lengthofhypo=0
        threshold=-1
        try: lookahead=int(arguments.lookahead)
        except: lookahead=lengthRef+(0.5*lengthRef)
        while lengthofhypo < lookahead:
            hyp_rest_copy=self.hyp_rest.copy()
            lengthofhypo+=1
            hypo=hyp_rest_copy[0:lengthofhypo]
            del hyp_rest_copy[0:lengthofhypo]
            if len(hyp_rest_copy) < len(self.refs_rest): break
            hyplower=[x.lower() for x in hypo]
            if arguments.wer:
                try: wer=jiwer.wer(lowNewRefString,' '.join(hyplower))
                except: wer=1
                bleu=1-wer
            else: 
                try: bleu=sentence_bleu(lowNewRef,hyplower,smoothing_function=SmoothingFunction().method3)
                except: bleu=0
            # if hypo ends in punctuation: multiply bleu by weight
            if re.search(r'[\.\?\!\,]$',' '.join(hypo)):
                bleu = bleu * punctuation_weight
            if bleu > threshold:
                # create new Align
                newAlign=Align(self.refs+[lowNewRef],self.hyps+[hypo],refs_rest,hyp_rest_copy,self,self.nrofhyps+1,self.beam,bleu,self.sentbleulist+[bleu],self.hypslower+[hyplower])
                toplist.append(newAlign)
                if len(toplist) > self.beam:
                    toplist.sort(key=lambda x: x.sentbleu,reverse=True)
                    toplist=toplist[0:self.beam]
                    threshold=toplist[-1].sentbleu
        nocachelist=[]
        for align in toplist:
            if align.inCache():
                pass
            else:
                align.putInCache()
                nocachelist.append(align)
        return nocachelist

    def isBetter(self,best):
        if self.nrofhyps > best.nrofhyps:
            return True
        elif self.nrofhyps == best.nrofhyps and self.bleu > best.bleu:
            return True
        else: return False
    
    def putInCache(self):
        nrhyps=str(self.nrofhyps)
        nrofwords=str(self.remainingwords)
        cache[nrhyps+'_'+nrofwords]=self.bleu

    def inCache(self):
        nrofhyps=str(self.nrofhyps)
        nrofwords=str(self.remainingwords)
        cachekey=nrofhyps+'_'+nrofwords
        if cachekey in cache:
            value=cache[cachekey]
            if value > self.bleu:
                return True
            else:
                return False
        else:
            return False


def specialSort(queue):
    returnqueue=[]
    while queue[0].nrofhyps <= breadthfirst_threshold:
        el=queue.pop()
        returnqueue.append(el)
    queue.sort(key=lambda x: (x.bleu*(1+(progress_weight*x.complete)),x.sentbleuavg),reverse=True)
    returnqueue=returnqueue+queue
    return returnqueue


parser=argparse.ArgumentParser(description='AlignerAStar.py')
parser.add_argument('-w','--wer',help="use WER optimizer instead of BLEU",action="store_true")
parser.add_argument('-p','--punctuation_weight',help="if the reference ends in a punctuation, multiply by this weight, giving more weight to alignment of punctuation (default = 1)")
parser.add_argument('-b','--beamsize',help='beamsize of the search algorithm (default=20)')
parser.add_argument('-f','--breadthfirst',help='Set the treshold on how many hypotheses should be aligned in a breadth first manner instead of A*  default=3')
parser.add_argument('-m','--maxexpand',help='The maximum nr of expansions before the system will stop (default = nr of reference sentences*500)')
parser.add_argument('-v','--verbose',help="Shows intermediate best hypotheses",action="store_true")
parser.add_argument('-l','--lookahead',help="Determines how many words more than the reference length should be checked. Default = 0.5 times reference length")
parser.add_argument('ref',help="The gold standard reference file")
parser.add_argument('hyp',help="The machine generated file (hypothesis) that needs to be aligned")
parser.add_argument('output',help="The name of the output file that will contain the aligned version of the hypothesis file")

arguments=parser.parse_args()

if arguments.punctuation_weight != None:
    try: punctuation_weight=float(arguments.punctuation_weight)
    except: print("Punctuation weight should be a number",file=stderr)
else:
    punctuation_weight=1.0  # if ref ends in punctuation, multiply be this weight
       
if arguments.beamsize != None:
    beamsize=int(arguments.beamsize)
else:
    beamsize=20

if arguments.breadthfirst != None:
    breadthfirst_threshold=int(arguments.breadthfirst)
else:
    breadthfirst_threshold=3


reffile= open(arguments.ref) #open(sys.argv[1])
refs=reffile.readlines()
nrofrefs=len(refs)

if arguments.maxexpand != None:
    maxexpands=int(arguments.maxexpand)
else:
    maxexpands=nrofrefs*500


progress_weight=1.0 # advantage for alignobjects closer to the end goal


tokenized_refs=[]
cache={}

for ref in refs:
    ref = ref.replace("'", "'")
    tokenized_refs.append(ref.split()) 
hypfile=open(arguments.hyp)#open(sys.argv[2])
hyp=hypfile.read()

outfile=open(arguments.output,'w')#open(sys.argv[3],'w')

tokenized_hyp=hyp.split() 

alignobject=Align([],[],tokenized_refs,tokenized_hyp,0,0,beamsize,0,[],[])
bestsolution=alignobject
q=[alignobject]
expandnr=0
aligncounter=0
bestbleu=0
incomplete=[]
solution=0

while len(q)>0:
    alignobject=q.pop(0)
    if alignobject.isBetter(bestsolution): 
        bestsolution=alignobject
        if arguments.verbose:
            print("\nCurrent best",file=sys.stderr)
            bestsolution.print()
    if alignobject.inCache():
        pass
    elif alignobject.nrofhyps == nrofrefs-1:
        complete=alignobject.addRemainingWords()
        if complete.bleu > bestbleu:
            solution=complete
            bestbleu=complete.bleu
            if (arguments.wer):
                print("\nBest 1-wer",solution.bleu,file=sys.stderr)
            else:
                print("\nBest bleu",solution.bleu,file=sys.stderr)
    else:
        expandnr+=1
        print("Expand nr",expandnr,"/",maxexpands,file=sys.stderr,end="\r")
        if expandnr > maxexpands:
            print("Maximum nr of expands reached, restarting from best solution:",maxexpands,file=sys.stderr)
            bestsolution.print()
            # Try again from best solution and empty q
            cache={}
            newq=[bestsolution]
            expandnr=0
            q=[]
        else :
            newq=alignobject.expand()
        if len(newq)<1:
            pass
        else:
            q=q+newq
            if q[0].nrofhyps > breadthfirst_threshold:
                q=specialSort(q)
                #q.sort(key=lambda x: (x.bleu*(1+(progress_weight*x.complete)),x.sentbleuavg),reverse=True)

if solution != 0 :
    print("ALIGNED SOLUTION:",file=sys.stderr)
    solution.print()
    solution.printOutput(outfile)
else:
    # show best solution
    print("Not finished -- try with a larger beam or larger maxexpands",file=sys.stderr)
    #bestsolution.printOutput()


    

