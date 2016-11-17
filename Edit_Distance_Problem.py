# -*- coding: utf-8 -*-
__author__ = "Bakary N'tji Diallo"
__email__ = "diallobakary4@gmail.com"
# -*- coding: utf-8 -*-
__author__ = "Bakary N'tji Diallo"
__email__ = "diallobakary4@gmail.com"
# RecursiveChange
# recursive solution to the minimum number of coin in change
def recursive_change(money, coins):
    if money == 0:
        return 0
    min_num_coins = money
    for i in range(1, len(coins)):
        if money >= coins[i-1]:
            num_coins = recursive_change(money - coins[i-1], coins)
            if num_coins + 1 < min_num_coins:
                min_num_coins = num_coins + 1
    return min_num_coins

# Test recursive_change
# print recursive_change(100,[5, 4, 1])

# Code Challenge: Solve the Change Problem. The DPChange pseudocode is reproduced below for your convenience.
#      Input: An integer money and an array Coins = (coin1, ..., coind).
#      Output: The minimum number of coins with denominations Coins that changes money.

def dpchange(money, coins):                                            # Coins (list of possible coins)"
    minNumCoins = {}                                                   # dico key: money value: MinNumCoin
    minNumCoins[0] = 0
    for m in range(1, money+1):
        minNumCoins[m] = float("inf")
        for i in range(len(coins)):
            if m >= coins[i]:
                if minNumCoins[(m - coins[i])] + 1 < minNumCoins[m]:
                    minNumCoins[m] = minNumCoins[(m - coins[i])] + 1
    return minNumCoins[money]

#Function test
# print dpchange(40,[50,25,20,10,5,1])
# print dpchange(24,[3,2])


# Code Challenge: Find the length of a longest path in the Manhattan Tourist Problem.
#      Input: Integers n and m, followed by an n × (m + 1) matrix Down and an (n + 1) × m matrix Right. The two matrices are separated by
#      the "-" symbol.
#      Output: The length of a longest path from source (0, 0) to sink (n, m) in the n × m rectangular grid whose edges are defined by the
#      matrices Down and Right.

def manhattanTourist(n, m, down, right):
    s = {}                                      # key = tuple of coordinate (O,O), value: lenght at that coordinate

    s[0,0] = 0
    for i in range(1, n+1):
        s[i,0] = s[i-1,0] + down[i][0]
    for j in range(1, m+1):
        s[0,j] = s[0, j-1] + right[0][j-1]              #-1 because list indexes start at zero
    for i in range(1, n+1):
        for j in range(1, m+1):
            s[i,j] = max([(s[i-1,j]+ down[i][j]),(s[i,j-1] + right[i][j-1])])

    return s[n,m]

# Test with sample
# n = 4
# m = 4
# down_weights = [0, [1, 0, 2, 4, 3], [4, 6, 5, 2, 1],[4, 4, 5 ,2, 1], [5, 6, 8, 5, 3]]  #down_weights[i][j]
# right_weights = [[3, 2, 4, 0],[3, 2, 4, 2],[0, 7, 3, 3],[3, 3, 0, 2],[1, 3, 2, 2]]      #right_weight[i][j+1]
# print manhattanTourist(n,m,down_weights,right_weights)

#Test with extra data set

with open("dataset_261_10.txt", "r") as data:
    #extracting m,n from file
    content = data.readline()
    mn = content.split()
    n = int(mn[0])
    m = int(mn[1])
    #extracting down from file
    content = data.readline()
    down_weights = [0]
    while "-" not in content:
        row = [int(e) for e in content.split()]
        down_weights.append(row)
        content = data.readline()
    #extracting right from file
    content = data.readline()
    right_weights = []
    while content != "":
        column = [int(e) for e in content.split()]
        right_weights.append(column)
        content = data.readline()

# print manhattanTourist(n,m,down_weights,right_weights)


# s dico; key = i,j, value = score (following the topological order)


# Code Challenge: Find the length of a longest path in the Manhattan Tourist Problem.
#      Input: Two sequences
#      Output:
BLOSUM62 = {'A': {'A': 4, 'C': 0, 'E': -1, 'D': -2, 'G': 0, 'F': -2, 'I': -1, 'H': -2, 'K': -1, 'M': -1, 'L': -1, 'N': -2, 'Q': -1, 'P': -1, 'S': 1, 'R': -1, 'T': 0, 'W': -3, 'V': 0, 'Y': -2}, 'C': {'A': 0, 'C': 9, 'E': -4, 'D': -3, 'G': -3, 'F': -2, 'I': -1, 'H': -3, 'K': -3, 'M': -1, 'L': -1, 'N': -3, 'Q': -3, 'P': -3, 'S': -1, 'R': -3, 'T': -1, 'W': -2, 'V': -1, 'Y': -2}, 'E': {'A': -1, 'C': -4, 'E': 5, 'D': 2, 'G': -2, 'F': -3, 'I': -3, 'H': 0, 'K': 1, 'M': -2, 'L': -3, 'N': 0, 'Q': 2, 'P': -1, 'S': 0, 'R': 0, 'T': -1, 'W': -3, 'V': -2, 'Y': -2}, 'D': {'A': -2, 'C': -3, 'E': 2, 'D': 6, 'G': -1, 'F': -3, 'I': -3, 'H': -1, 'K': -1, 'M': -3, 'L': -4, 'N': 1, 'Q': 0, 'P': -1, 'S': 0, 'R': -2, 'T': -1, 'W': -4, 'V': -3, 'Y': -3}, 'G': {'A': 0, 'C': -3, 'E': -2, 'D': -1, 'G': 6, 'F': -3, 'I': -4, 'H': -2, 'K': -2, 'M': -3, 'L': -4, 'N': 0, 'Q': -2, 'P': -2, 'S': 0, 'R': -2, 'T': -2, 'W': -2, 'V': -3, 'Y': -3}, 'F': {'A': -2, 'C': -2, 'E': -3, 'D': -3, 'G': -3, 'F': 6, 'I': 0, 'H': -1, 'K': -3, 'M': 0, 'L': 0, 'N': -3, 'Q': -3, 'P': -4, 'S': -2, 'R': -3, 'T': -2, 'W': 1, 'V': -1, 'Y': 3}, 'I': {'A': -1, 'C': -1, 'E': -3, 'D': -3, 'G': -4, 'F': 0, 'I': 4, 'H': -3, 'K': -3, 'M': 1, 'L': 2, 'N': -3, 'Q': -3, 'P': -3, 'S': -2, 'R': -3, 'T': -1, 'W': -3, 'V': 3, 'Y': -1}, 'H': {'A': -2, 'C': -3, 'E': 0, 'D': -1, 'G': -2, 'F': -1, 'I': -3, 'H': 8, 'K': -1, 'M': -2, 'L': -3, 'N': 1, 'Q': 0, 'P': -2, 'S': -1, 'R': 0, 'T': -2, 'W': -2, 'V': -3, 'Y': 2}, 'K': {'A': -1, 'C': -3, 'E': 1, 'D': -1, 'G': -2, 'F': -3, 'I': -3, 'H': -1, 'K': 5, 'M': -1, 'L': -2, 'N': 0, 'Q': 1, 'P': -1, 'S': 0, 'R': 2, 'T': -1, 'W': -3, 'V': -2, 'Y': -2}, 'M': {'A': -1, 'C': -1, 'E': -2, 'D': -3, 'G': -3, 'F': 0, 'I': 1, 'H': -2, 'K': -1, 'M': 5, 'L': 2, 'N': -2, 'Q': 0, 'P': -2, 'S': -1, 'R': -1, 'T': -1, 'W': -1, 'V': 1, 'Y': -1}, 'L': {'A': -1, 'C': -1, 'E': -3, 'D': -4, 'G': -4, 'F': 0, 'I': 2, 'H': -3, 'K': -2, 'M': 2, 'L': 4, 'N': -3, 'Q': -2, 'P': -3, 'S': -2, 'R': -2, 'T': -1, 'W': -2, 'V': 1, 'Y': -1}, 'N': {'A': -2, 'C': -3, 'E': 0, 'D': 1, 'G': 0, 'F': -3, 'I': -3, 'H': 1, 'K': 0, 'M': -2, 'L': -3, 'N': 6, 'Q': 0, 'P': -2, 'S': 1, 'R': 0, 'T': 0, 'W': -4, 'V': -3, 'Y': -2}, 'Q': {'A': -1, 'C': -3, 'E': 2, 'D': 0, 'G': -2, 'F': -3, 'I': -3, 'H': 0, 'K': 1, 'M': 0, 'L': -2, 'N': 0, 'Q': 5, 'P': -1, 'S': 0, 'R': 1, 'T': -1, 'W': -2, 'V': -2, 'Y': -1}, 'P': {'A': -1, 'C': -3, 'E': -1, 'D': -1, 'G': -2, 'F': -4, 'I': -3, 'H': -2, 'K': -1, 'M': -2, 'L': -3, 'N': -2, 'Q': -1, 'P': 7, 'S': -1, 'R': -2, 'T': -1, 'W': -4, 'V': -2, 'Y': -3}, 'S': {'A': 1, 'C': -1, 'E': 0, 'D': 0, 'G': 0, 'F': -2, 'I': -2, 'H': -1, 'K': 0, 'M': -1, 'L': -2, 'N': 1, 'Q': 0, 'P': -1, 'S': 4, 'R': -1, 'T': 1, 'W': -3, 'V': -2, 'Y': -2}, 'R': {'A': -1, 'C': -3, 'E': 0, 'D': -2, 'G': -2, 'F': -3, 'I': -3, 'H': 0, 'K': 2, 'M': -1, 'L': -2, 'N': 0, 'Q': 1, 'P': -2, 'S': -1, 'R': 5, 'T': -1, 'W': -3, 'V': -3, 'Y': -2}, 'T': {'A': 0, 'C': -1, 'E': -1, 'D': -1, 'G': -2, 'F': -2, 'I': -1, 'H': -2, 'K': -1, 'M': -1, 'L': -1, 'N': 0, 'Q': -1, 'P': -1, 'S': 1, 'R': -1, 'T': 5, 'W': -2, 'V': 0, 'Y': -2}, 'W': {'A': -3, 'C': -2, 'E': -3, 'D': -4, 'G': -2, 'F': 1, 'I': -3, 'H': -2, 'K': -3, 'M': -1, 'L': -2, 'N': -4, 'Q': -2, 'P': -4, 'S': -3, 'R': -3, 'T': -2, 'W': 11, 'V': -3, 'Y': 2}, 'V': {'A': 0, 'C': -1, 'E': -2, 'D': -3, 'G': -3, 'F': -1, 'I': 3, 'H': -3, 'K': -2, 'M': 1, 'L': 1, 'N': -3, 'Q': -2, 'P': -2, 'S': -2, 'R': -3, 'T': 0, 'W': -3, 'V': 4, 'Y': -1}, 'Y': {'A': -2, 'C': -2, 'E': -2, 'D': -3, 'G': -3, 'F': 3, 'I': -1, 'H': 2, 'K': -2, 'M': -1, 'L': -1, 'N': -2, 'Q': -1, 'P': -3, 'S': -2, 'R': -2, 'T': -2, 'W': 2, 'V': -1, 'Y': 7}}
#Python formatted version of BLOSUM62. To retrieve a value call blosum[row][col], e.g. blosum['A']['A'] returns 4


def lcs_backtrack(v,w, scoring_matrix):
    s = {}                                            # key = tuple of coordinate (O,O), value: lenght at that coordinate
    s[0,0] = 0
    backtrack = {}                                    # list of "↓" "→" "↘" to move back from sink to source
    sigma = 5                                         #penality for insertion deletion
    for i in range(1, len(v)+1):                      #Initialization of first row and column with indel penalities
        s[i,0] = s[i - 1,0] - sigma
    for j in range(1,len(w)+1):
        s[0,j] = s[0,j -1] - sigma

    for i in range(1, len(v)+1):
        for j in range(1, len(w)+1):

            s[i,j] = max(s[i-1,j] - sigma, s[i,j-1] - sigma, s[i-1,j-1] + scoring_matrix[v[i-1]][w[j-1]])


    for i in range(1, len(v)+1):
        for j in range(1, len(w)+1):
            if s[i,j] == s[i-1,j] - sigma:
                backtrack[i,j] = "↓"
            elif s[i,j] == s[i,j-1]- sigma:
                backtrack[i,j] = "→"
            elif s[i,j] == s[i-1,j-1] + scoring_matrix[v[i-1]][w[j-1]] :
                backtrack[i,j] = "↘"


    return backtrack, s[len(v),len(w)]          #return the backtrack symbols and the score



def outputLCS(backtrack, v, w):
    LCS = ""                            #the longes common sequence
    i, j = len(v), len(w)
    W2 = ""
    V2 = ""
    while i > 0 and j > 0:
        if backtrack[i,j] == "↘":       #"↘" means two letters are matching
            LCS += v[i-1]               #we add the matching letter to LCS
            W2 += w[j-1]
            V2 += v[i-1]
            i -= 1
            j -= 1
        elif backtrack[i,j] == "↓":
            V2 += v[i-1]
            W2 += "-"
            i -= 1
        elif backtrack[i,j] == "→":
            V2 += "-"
            W2 += w[j-1]
            j -= 1

    if i == 1 :
        V2 += v[i-1]
        W2 += "-"
    if j == 1 :
        V2 += "-"
        W2 += w[j-1]
    return LCS[::-1],  V2[::-1], W2[::-1]       #reurn longest CS, v aligned, w aligned

# Code Challenge: Find the longest common sequence w , v: the longest one
#      Input: Two sequences
#      Output: longest common sequence
def lcs(v, w):                                          #v: the longest one
    backtrack = lcs_backtrack(v, w, BLOSUM62)           #will return (backtrack, score)
    return backtrack[1], outputLCS(backtrack[0], v, w)      #return score, and the strings


#test           EXPECTED OUTPUT: GTAG
# v = "ILYPRQSMICMSFCFWDMWKKDVPVVLMMFLERRQMQSVFSWLVTVKTDCGKGIYNHRKYLGLPTMTAGDWHWIKKQNDPHEWFQGRLETAWLHSTFLYWKYFECDAVKVCMDTFGLFGHCDWDQQIHTCTHENEPAIAFLDLYCRHSPMCDKLYPVWDMACQTCHFHHSWFCRNQEMWMKGDVDDWQWGYHYHTINSAQCNQWFKEICKDMGWDSVFPPRHNCQRHKKCMPALYAGIWMATDHACTFMVRLIYTENIAEWHQVYCYRSMNMFTCGNVCLRCKSWIFVKNYMMAPVVNDPMIEAFYKRCCILGKAWYDMWGICPVERKSHWEIYAKDLLSFESCCSQKKQNCYTDNWGLEYRLFFQSIQMNTDPHYCQTHVCWISAMFPIYSPFYTSGPKEFYMWLQARIDQNMHGHANHYVTSGNWDSVYTPEKRAGVFPVVVPVWYPPQMCNDYIKLTYECERFHVEGTFGCNRWDLGCRRYIIFQCPYCDTMKICYVDQWRSIKEGQFRMSGYPNHGYWFVHDDHTNEWCNQPVLAKFVRSKIVAICKKSQTVFHYAYTPGYNATWPQTNVCERMYGPHDNLLNNQQNVTFWWKMVPNCGMQILISCHNKMKWPTSHYVFMRLKCMHVLMQMEYLDHFTGPGEGDFCRNMQPYMHQDLHWEGSMRAILEYQAEHHRRAFRAELCAQYDQEIILWSGGWGVQDCGFHANYDGSLQVVSGEPCSMWCTTVMQYYADCWEKCMFA"
# w = "ILIPRQQMGCFPFPWHFDFCFWSAHHSLVVPLNPQMQTVFQNRGLDRVTVKTDCHDHRWKWIYNLGLPTMTAGDWHFIKKHVVRANNPHQWFQGRLTTAWLHSTFLYKKTEYCLVRHSNCCHCDWDQIIHTCAFIAFLDLYQRHWPMCDKLYCHFHHSWFCRNQEMSMDWNQWFPWDSVPRANCLEEGALIALYAGIWANSMKRDMKTDHACTVRLIYVCELHAWLKYCYTSINMLCGNVCLRCKSWIFVKLFYMYAPVVNTIEANSPHYYKRCCILGQGICPVERKSHCEIYAKDLLSFESCCSQKQNCYTDNWGLEYRLFFQHIQMECTDPHANRGWTSCQTAKYWHFNLDDRPPKEFYMWLQATPTDLCMYQHCLMFKIVKQNFRKQHGHANPAASTSGNWDSVYTPEKMAYKDWYVSHPPVDMRRNGSKMVPVWYPPGIWHWKQSYKLTYECFFTVPGRFHVEGTFGCNRWDHQPGTRRDRQANHQFQCPYSDTMAIWEHAYTYVDQWRSIKEGQMPMSGYPNHGQWNVHDDHTNEQERSPICNQPVLAKFVRSKNVSNHEICKKSQTVFHWACEAQTNVCERMLNNQHVAVKRNVTFWWQMVPNCLWSCHNKMTWPTRPEQHRLFFVKMRLKCMHEYLDVAPSDFCRNMQAYMHSMRAILEYQADFDLKRRLRAIAPMDLCAQYDQEIILWSGGYIYDQSLQVVSCEGCSYYADCYVKCINVKEKCMFA"
# v = "PLEASANTLY"
# w = "MEANLY"
# my_result = lcs(v,w)
# print my_result[0]              #score of the alignment
# print my_result[1][1]           #w aligned to v
# print my_result[1][2]           #v aligned to w


 # Solve the Longest Path in a DAG Problem.
 #     Input: An integer representing the source node of a graph, followed by an integer representing the sink node of the graph, followed by
 #     a list of edges in the graph. The edge notation "0->1:7" indicates that an edge connects node 0 to node 1 with weight 7.
 #     Output: The length of a longest path in the graph, followed by a longest path. (If multiple longest paths exist, you may return any one.)

from topological_sort import *
with open("exam.txt","r") as content:
    data = content.readlines()
    source = int(data[0])
    sink = int(data[1])
    unsorted_graph = []
    edge_weights = {}
    for e in data[2:]:
        edge_a = int(e[:e.index('-')])
        edge_b = [int(e[e.index('>')+1:e.index(":")])]
        unsorted_graph.append((edge_a,edge_b))
        edge_weights[(edge_a, tuple(edge_b))] = int(e[e.index(":")+1:])

# Now change the data format from
# [(14, [29]), (14, [31]), (14, [30])] to [(14, [29, 31, 30])]
new_unsorted_graph = []
for edge1 in unsorted_graph:
    new_egde = [edge1[0],edge1[1][0]]
    for edge2 in unsorted_graph:
        if edge1[0] == edge2[0] and edge1 != edge2:
            new_egde.append(edge2[1][0])

    if new_egde[0] not in [e[0] for e in new_unsorted_graph]:
        new_unsorted_graph.append((new_egde[0],list(new_egde[1:])))

unsorted_graph = new_unsorted_graph

def longest_path_in_DAG(graph, weights, source, sink):
    # weights : {(19, (20,)): 13}
    score = {}                          #{0: -inf, 1: -inf} key node, value best score to the
    for edge in graph:
        score[edge[0]] = float("-inf")
        for node in edge[1]:
            score[node] = float("-inf")

    score[source] = 0
    sorted_graph = topolgical_sort(graph)
    # remove all nodes that have in-degrees equals
    # to 0 but is not the start node, until the sole node has in-degree equals to 0 is the start node.
    topo_order = []                     #building list of nodes in topological order
    for e in sorted_graph:
        if e[0] not in topo_order:
            topo_order.append(e[0])
            topo_order.extend([a for a in e[1] if a not in topo_order])
    topo_order = topo_order[::-1]
    # print(topo_order)                       #used just to print the order from the exam
    topo_order.remove(source)
    probable_path = [source]
    for nodeb in topo_order:                                      # [7, 1, 10, 8, 6, 5, 3, 0, 22, 34, ...]
        predecessors_nodeb = []                                    #built list of predecessor of nodeb
        for nodea in sorted_graph:                                 # (21, [35, 37]): format nodeb
            if nodeb in nodea[1] :                                 # nodea[0] is a predecessor of nodeb
                predecessors_nodeb.append(nodea[0])                # making list of scores of predecessors

        score_predecessors_nodeb = \
            [score[e] + weights[(e, (nodeb,))] for e in predecessors_nodeb]
        prede_score = dict(zip(predecessors_nodeb, score_predecessors_nodeb)) #key: prede, value= score

        if len(predecessors_nodeb) > 0:                             # the first node has no predecessor
            score[nodeb] = max(score_predecessors_nodeb)
            probable_path.append(max(prede_score, key = prede_score.get))

    probable_path.append(sink)
    longest_path = [sink]
    b = sink
    for e in probable_path[::-1]:
        pred = {}
        if (e, (b,)) in weights:
            pred[e] = score[e]
            longest_path.append(max(pred, key = pred.get))
            b = e
            if e == source:             # we have reached the source of the graph
                break

    longest_path.pop()
    longest_path.append(source)
    longest_path = [str(e) for e in longest_path[::-1]]
    return score[sink], '->'.join(longest_path)

#Function test for longest_path_in_DAG
# print longest_path_in_DAG(unsorted_graph, edge_weights, source, sink)


# Edit Distance Problem: Find the edit distance between two strings.
#      Input: Two strings.
#      Output: The edit distance between these strings

def edit_distance(v,w):
    s = {}                                            # key = tuple of coordinate (O,O), value: lenght at that coordinate
    s[0,0] = 0
    sigma = -1                                         #penality for insertion deletion
    for i in range(1, len(v)+1):                      #Initialization of first row and column with indel penalities
        s[i,0] = s[i - 1,0] - sigma
    for j in range(1,len(w)+1):
        s[0,j] = s[0,j -1] - sigma

    for i in range(1, len(v)+1):
        for j in range(1, len(w)+1):
            if v[i-1] == w[j-1]: match = 0
            elif v[i-1] != w[j-1]: match = -1
            s[i,j] = min(s[i-1,j] - sigma, s[i,j-1] - sigma, s[i-1,j-1] - match)

    return s[len(v),len(w)]

#Function test for edit_distance
# v = "PLEASANTLY"
# w = "MEANLY"
#test with dataset at http://bioinformaticsalgorithms.com/data/extradatasets/alignment/edit_distance.txt
v = "GGACRNQMSEVNMWGCWWASVWVSWCEYIMPSGWRRMKDRHMWHWSVHQQSSPCAKSICFHETKNQWNQDACGPKVTQHECMRRRLVIAVKEEKSRETKMLDLRHRMSGRMNEHNVTLRKSPCVKRIMERTTYHRFMCLFEVVPAKRQAYNSCDTYTMMACVAFAFVNEADWWKCNCAFATVPYYFDDSCRMVCGARQCYRLWQWEVNTENYVSIEHAEENPFSKLKQQWCYIPMYANFAWSANHMFWAYIANELQLDWQHPNAHPIKWLQNFLMRPYHPNCGLQHKERITPLHKSFYGMFTQHHLFCKELDWRIMAHANRYYCIQHGWHTNNPMDPIDTRHCCMIQGIPKRDHHCAWSTCDVAPLQGNWMLMHHCHHWNRVESMIQNQHEVAAGIKYWRLNRNGKLPVHTADNYGVLFQRWWFLGWYNFMMWHYSLHFFAVNFYFPELNAGQMPRFQDDQNRDDVYDTCIWYFAWSNTEFMEVFGNMMMYSRPMTKMGFHGMMLPYIAINGLRSISHVNKGIGPISGENCNLSTGLHHYGQLRMVMCGYCTPYRTEVKNQREMISAVHCHQHIDWRWIWCSGHWFGSNKCDLRIEDLQNYEPAKNKSNWPYMKECRKTEPYQDNIETMFFHQHDLARDSGYIANGWHENCRQHQDFSNTFAGGHKGTPKGEHMRRSLYVWDTDCVEKCQWVPELFALCWWTPLPDGVPVMLGTYRQYMFGLVVLYWFEVKYSCHNSWDYYNFHEGTMKDSDPENWCFWGMQIIQFHDHGKPEFFQDPMKQIIKTECTAYNSFMMGHIGKTTIVYLVSYIGRLWMKSCCLTWPPYATAPIKWAEETLLDFGQGPHPKYACHFTHQNMIRLAKLPMYWLWKLMFHE"
w = "GMWGFVQVSTQSRFRHMWHWSVHQQSSECAKSICHHEWKNQWNQDACGPKVTQHECMANMPMHKCNNWFWRLVIAVKEEKVRETKMLDLIHRHWLVLNQGRMNEHNVTLRKSPCVKRIMHKWKSRTTFHRFMCLMASEVVPAKRGAQCWRQLGTYATYTVYTMMACVAFAFEYQQDNDNEADWWKCNCAFVPVYFDDSCRPVVGAFQCYRLGLPFGTGWNYAEENPFSKLKQQMHRKTMGECKNMMIWAYCANELQLPIKWGSMYHEHDFQLPPYHPNRFHKIRITILHKSFYGMFTQHHLFCKELDWRIMAWANRYYCIQHGWHTNNPDDPITRHKCMIQGGQNSRNADIRHMPVQCGNWGHAIGLEMPMPMHHCHHANRVESMIQTQHYWGPKLNRNADWWFLGWQNFEIFRMPILRWMGAYEWHYSLHFFAVNFYFPELNAGQMPRFQDDQNNNACYDVWAWSNTEFMEVNGIKKLRFGNMMMYSRPMTKMGFHGMMKSRSISHVNKGIGPISGENCSTGLHHYGQLTEVKNQREMISAVHCHQHIWCKCDLRIEPAKNKGYWPYQKEFCWRKQINSRKTEPYQVAPVINIETMFFDFWYIANGMHENCRRTGHKPNPDCVEKCQWVPELFALCWWRAMPDGVPVMLGTMFGLVVYWFEVKYSCHNSLYRRVTDYYNFHEGTMKDHEVPWNWDNEHCHDHGKAEFFFQMLKIPICDPMKAIIPSTEMVNTPWHPFSFMMGHDGKTTIVYSGSYIGRLWVPSRWKPYAPANWKMPIKWAEETLLMVPHPHFTHQQLWGTTLRLAKLPMYWLWKLMFHHLFGVK"

v = "LKAWLKFNFLYHELTSSTRFVEGHLSNVSFGSTLNKGHCMNTEIMMPMQPRAVKCRKSRPMVQLTGHDHVFPIKLFWHELKGMRNGCDHSDPPQSQIRFGYEVLQTHSQEFCAQKQRFFIHPISCGTRRAEHNHCPHPKFFINPGESYLKALLEYGKHESMCPSTTAIRPKCACINAQTVECPVGCVNGQRWTWCNLSQAWRCSGNYTTSMVHRAMNPSKLWDPDSDWCDILFQLIMKNSAICTPHDFVPVNLSDMNEDVLNEPEGHSASQAMIAFNRALSHRLNCIVYSYCINNYQWIHSEIPWNDYQKMVMYRDAKSSKVFMNWLQVEQWYYCNNCFQHRFPDKDMSCQVENKWFCFSLKSLITEMVHHSKFIQCEQKGSDCVLPTREPWRWAQKCCSTQKFFCWWCYGQVYSFIATRDSEVSWTKDFGIWTPHLVMSWCSFYIFQSPSFTAKPHNTYCKFLKVQAAHTRTLYAIWINDCWRWAWVTIFHLCAAEGKFHKGMTQTPVIYPIEAMTHLFDNYTAFWCRLDEPLLFAWAILWDGPNAVDFFCGLEKLMMCVTQYTCARNTHCPHKELEKKKVKYIIAKMMNMLSIYDLEALTNDQLFSACRFLQNEKAENFNMMSAEIAGQQWKHWTTPLNNNHGDDAQSDDKRNGGNWLEYMECQFNQGNNNIREQQHLWWRWDQITDTTYVTIRPNFLEMVLKNTCNAKTFEHRPECLIGGHSKMNGNIKNRGYDAWCWAVDQRNSGTLQWPTWWGFCSVFIFFGYLDVVKKPVACRHLGFAWFAMFDFRMPWWSNVCDGCGSSWMHSLPQQWNKYAPTEMASYDLLVWYHKGDCNFTPWSYIADLNSPCQGYCTIHGRVHYFNNRKFRCAARQDFCT"
w = "LKAWLKFNFLYRKQYDLKELTSSAEGWKGVSDIDNRREFVEGILSNVSKGITEIMMPMQPRAVKCRSVQTGIKLFWHENKGMRNGCDNSDPPQYEVLQKNHDTHQWEFCANKQRFFIARISCGTRRAEEYNVCPHPKFLWALLDYGKKYLCCACINAQDVNQHMDICPCVNGQRWTWCNLWDICYWQAWRCNQEGHLQMFGNYTTSFVQRAMWLFQLIMKNAITPADFQPDEYLCDENLSDNNVNEDVLNTMAPAGHSASQAMIAQVENRALSHRIEIFLMQLTLTIVYSYSINNYQEIHSGHVIPWNDYQKMVMYRDAVTQELSLVGAKNFMNWLCNPDKDMSCEVENKWFCFSLKYFVAWWIRHTATACMLHHSKFIQCEQKGSDSALPTREPWRWAQKCCSTQKFFCWWCYGQWHELHYSCITRDSVFSEQSWTFDGGIWRIPHSTTPYIFQSPSFTAKPHNWYCKWLKVQSCNGAWIWINDCWRWAAAEGKFHKGNTQTPVIYPLFDNYTAFNHRPCHPTPWQFLFADGPNAVDFFCGLEKLMMCVAQYTCARNTHCPHKELEKIIYLWTNTKTKRIHAKMMNMCIWSIYTLEALRHDQACNDQNFSNCRCRFLKNEKAENFNMMSADLALPSIAGQQWKHITTPLNNNHGDDAQSDDKRNGGNWLEYMECQFNQDNNNIREQQHWQWETIDQITDTTYVDIRPNFLEMVLKNTWNAKTAQKGGHSKMNGNIKNRNEDAPDQRNSGLQWPTWWGFCSVFIFFPYLDVVKKPVACRHMCSMFAWFAMFDFRMPWWSNVCDGCGPQQWNKGAPTEMISYDLAVDCWYRIADLTQQYKTIHGRNNRHFRCAARQDFCT"
print edit_distance(v,w)
