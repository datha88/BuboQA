import os
import math
import pickle

from argparse import ArgumentParser
from collections import defaultdict
from util import clean_uri, processed_text, www2fb, rdf2fb


# Load up reachability graph

def load_index(filename):
    print("Loading index map from {}".format(filename))
    with open(filename, 'rb') as handler:
        index = pickle.load(handler)
    return index

# Load predicted MIDs and relations for each question in valid/test set
def get_mids(filename, hits):
    print("Entity Source : {}".format(filename))
    id2mids = defaultdict(list)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split(' %%%% ')
        lineid = items[0]
        cand_mids = items[1:][:hits]
        for mid_entry in cand_mids:
            mid, mid_name, mid_type, score = mid_entry.split('\t')
            id2mids[lineid].append((mid, mid_name, mid_type, float(score)))
    return id2mids

def get_rels(filename, hits):
    print("Relation Source : {}".format(filename))
    id2rels = defaultdict(list)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split(' %%%% ')
        lineid = items[0].strip()
        rel = www2fb(items[1].strip())
        label = items[2].strip()
        score = items[3].strip()
        if len(id2rels[lineid]) < hits:
            id2rels[lineid].append((rel, label, float(score)))
    return id2rels


def get_questions(filename):
    print("getting questions ...")
    id2questions = {}
    id2goldmids = {}
    fin =open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        lineid = items[0].strip()
        mid = items[1].strip()
        question = items[5].strip()
        rel = items[3].strip()
        id2questions[lineid] = (question, rel)
        id2goldmids[lineid] = mid
    return id2questions, id2goldmids

def get_mid2wiki(filename):
    print("Loading Wiki")
    mid2wiki = defaultdict(bool)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        sub = rdf2fb(clean_uri(items[0]))
        mid2wiki[sub] = True
    return mid2wiki

def calc_f1(gold_mid,gold_rel,pred_id2answers):
    precision = 0
    recall = 0
    f1 = 0
    if len(pred_id2answers) == 0:
        return (1, 0, 0)
    relation_mid_found = 0
    for i in range(len(pred_id2answers)):
        if (pred_id2answers[i][0] == gold_mid and pred_id2answers[i][1] == gold_rel):
            relation_mid_found = 1
            break
    if(relation_mid_found):
        precision = precision + 1
    
    for i in range(len(pred_id2answers)):
        if (pred_id2answers[i][0] == gold_mid and pred_id2answers[i][1] == gold_rel):
            recall = recall + 1
            break
    precision = precision /len(pred_id2answers)
    
    recall = recall / 1
    if (precision +recall>0 ):
        f1 = 2* recall *precision/(recall +precision)
   
    return precision, recall, f1

def evidence_integration(data_path, ent_path, rel_path, output_dir, index_reach, index_degrees, mid2wiki, is_heuristics, HITS_ENT, HITS_REL):
    
    id2questions, id2goldmids = get_questions(data_path)
    id2mids = get_mids(ent_path, HITS_ENT)
    id2rels = get_rels(rel_path, HITS_REL)
    file_base_name = os.path.basename(data_path)
    fout = open(os.path.join(output_dir, file_base_name), 'w')
    out_f = open(os.path.join(output_dir, 'error_analysis.txt'), 'w')
    id2answers = defaultdict(list)
    found, notfound_both, notfound_mid, notfound_rel = 0, 0, 0, 0
    retrieved, retrieved_top1, retrieved_top2, retrieved_top3 = 0, 0, 0, 0
    lineids_found1 = []
    lineids_found2 = []
    lineids_found3 = []

    correct_answers_count = 0
    avg_recall = 0
    avg_precision = 0
    avg_f1 = 0
    count = 0
    # for every lineid
    for line_id in id2goldmids:
        if line_id not in id2mids and line_id not in id2rels:
            notfound_both += 1
            continue
        elif line_id not in id2mids:
            notfound_mid += 1
            continue
        elif line_id not in id2rels:
            notfound_rel += 1
            continue

        found += 1
        question, truth_rel = id2questions[line_id]
        truth_rel = www2fb(truth_rel)
        truth_mid = id2goldmids[line_id]
        mids = id2mids[line_id]
        rels = id2rels[line_id]
        
        
        
        if is_heuristics:
            for (mid, mid_name, mid_type, mid_score) in mids:
                for (rel, rel_label, rel_log_score) in rels:
                    # if this (mid, rel) exists in FB
                    if rel in index_reach[mid]:
                        '''if (line_id == 'test-1'):
                            print(mid)
                            print(rel)
                            print(index_reach[mid])
                            print(int(index_degrees[mid][0]))'''
                        rel_score = math.exp(float(rel_log_score))
                        comb_score = (float(mid_score)**0.6) * (rel_score**0.1)
                        id2answers[line_id].append((mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score, int(mid2wiki[mid]), int(index_degrees[mid][0])))
                        #correct_answers_count = correct_answers_count + 1
                    # I cannot use retrieved here because I use contain different name_type
                    #if mid ==truth_mid and rel == truth_rel:
                    #    retrieved += 1
            id2answers[line_id].sort(key=lambda t: (t[6], t[3],  t[7], t[8]), reverse=True)
        else:
            id2answers[line_id] = [(mids[0][0], rels[0][0])]
        #print(id2answers[line_id])
        
        #id2answers[line_id] = [(mids[0][0], rels[0][0])]
        # write to file
        fout.write("{}".format(line_id))
        for answer in id2answers[line_id]:
            mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score, _, _ = answer
            fout.write(" %%%% {}\t{}\t{}\t{}\t{}".format(mid, rel, mid_name, mid_score, rel_score, comb_score))
        fout.write('\n')
        
        precision, recall, f1 = calc_f1(truth_mid, truth_rel, id2answers[line_id])
        avg_recall += recall
        avg_precision += precision
        avg_f1 += f1
        count += 1
        out_f.write(" %%%% {}\t{}\t{}\t".format(line_id, truth_mid, truth_rel, id2answers[line_id], f1))
        for i in range(len(id2answers[line_id])):
            out_f.write("{}\t{}\t".format(id2answers[line_id][i][0],id2answers[line_id][i][1]))
        out_f.write('{}'.format(f1))
        out_f.write('\n')
        
        if len(id2answers[line_id]) >= 1 and id2answers[line_id][0][0] == truth_mid \
                and id2answers[line_id][0][1] == truth_rel:
            retrieved_top1 += 1
            retrieved_top2 += 1
            retrieved_top3 += 1
            lineids_found1.append(line_id)
        elif len(id2answers[line_id]) >= 2 and id2answers[line_id][1][0] == truth_mid \
                and id2answers[line_id][1][1] == truth_rel:
            retrieved_top2 += 1
            retrieved_top3 += 1
            lineids_found2.append(line_id)
        elif len(id2answers[line_id]) >= 3 and id2answers[line_id][2][0] == truth_mid \
                and id2answers[line_id][2][1] == truth_rel:
            retrieved_top3 += 1
            lineids_found3.append(line_id)
        relation_mid_found = 0
        for i in range(len(id2answers[line_id])):
            if (id2answers[line_id][i][0] == truth_mid and id2answers[line_id][i][1] == truth_rel):
                relation_mid_found = 1
                break
        if relation_mid_found:
            correct_answers_count = correct_answers_count + 1

    print()
    print("found:              {}".format(found / len(id2goldmids) * 100.0))
    print("retrieved at top 1: {}".format(retrieved_top1 / len(id2goldmids) * 100.0))
    print("retrieved at top 2: {}".format(retrieved_top2 / len(id2goldmids) * 100.0))
    print("retrieved at top 3: {}".format(retrieved_top3 / len(id2goldmids) * 100.0))
    #print("retrieved at inf:   {}".format(retrieved / len(id2goldmids) * 100.0))
    
    if is_heuristics:
        print(correct_answers_count)
        print(len(id2goldmids))
        print("Accuracy : {}".format(correct_answers_count / len(id2goldmids) * 100.0))
        avg_recall = float(avg_recall) / count
        avg_precision = float(avg_precision) / count
        avg_f1 = float(avg_f1) / count
        print('Average Recall', avg_recall)
        print('Average Precision', avg_precision)
        print('Average F1', avg_f1)
    fout.close()
    out_f.close()
    return id2answers


if __name__=="__main__":
    parser = ArgumentParser(description='Perform evidence integration')
    parser.add_argument('--ent_type', type=str, required=True, help="options are [crf|lstm|gru]")
    parser.add_argument('--rel_type', type=str, required=True, help="options are [lr|cnn|lstm|gru]")
    parser.add_argument('--index_reachpath', type=str, default="../indexes/reachability_2M.pkl",
                        help='path to the pickle for the reachability index')
    parser.add_argument('--index_degreespath', type=str, default="../indexes/degrees_2M.pkl",
                        help='path to the pickle for the index with the degree counts')
    parser.add_argument('--data_path', type=str, default="../data/processed_simplequestions_dataset/test.txt")
    parser.add_argument('--ent_path', type=str, default="../entity_linking/results/crf/test-h100.txt", help='path to the entity linking results')
    parser.add_argument('--rel_path', type=str, default="../relation_prediction/nn/results/cnn/test.txt", help='path to the relation prediction results')
    parser.add_argument('--wiki_path', type=str, default="../data/fb2w.nt")
    parser.add_argument('--hits_ent', type=int, default=50, help='the hits here has to be <= the hits in entity linking')
    parser.add_argument('--hits_rel', type=int, default=5, help='the hits here has to be <= the hits in relation prediction retrieval')
    parser.add_argument('--no_heuristics', action='store_false', help='do not use heuristics', dest='heuristics')
    parser.add_argument('--output_dir', type=str, default="./results")
    args = parser.parse_args()
    print(args)

    ent_type = args.ent_type.lower()
    rel_type = args.rel_type.lower()
    assert(ent_type == "crf" or ent_type == "lstm" or ent_type == "gru")
    assert(rel_type == "lr" or rel_type == "cnn" or rel_type == "lstm" or rel_type == "gru")
    output_dir = os.path.join(args.output_dir, "{}-{}".format(ent_type, rel_type))
    os.makedirs(output_dir, exist_ok=True)

    index_reach = load_index(args.index_reachpath)
    index_degrees = load_index(args.index_degreespath)
    mid2wiki = get_mid2wiki(args.wiki_path)

    test_answers = evidence_integration(args.data_path, args.ent_path, args.rel_path, output_dir, index_reach, index_degrees, mid2wiki, args.heuristics, args.hits_ent, args.hits_rel)







