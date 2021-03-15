# System tools
import os
import sys
sys.path.append(os.path.join(".."))
import numpy as np
import pandas as pd

from collections import Counter
from itertools import combinations 
from tqdm import tqdm

import spacy
nlp = spacy.load("en_core_web_sm")
import argparse

#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

import networkx as nx


#creating the main funktion, to be called from the commandline.
def main():
    
    
    '''
    Defining commandline arguments:
    '''
    #
    ap = argparse.ArgumentParser(description = "[INFO] creating network analysis")
    #first argument: the path to the csv file that one which to create a network analysis from:
    ap.add_argument("-f", "--filepath", required=True, type=str, help="path to file")
    ap.add_argument("-e", "--edges", required=True, type=int, help="The number of edges a node must have to be in the network.")
    args = vars(ap.parse_args())
    
    
    '''
    Defining filepath
    '''
    
    #Første argparse
    input_file = os.path.join(args["filepath"])
    data = pd.read_csv(input_file)
    
    
    real_df = data[data["label"]=="REAL"]["text"]
    
    
    post_entities = []

    for text in tqdm(real_df[:100]):
        # create temporary list 
        tmp_entities = []
        # create doc object
        doc = nlp(text)
        # for every named entity
        for entity in doc.ents:
            # if that entity is a person
            if entity.label_ == "PERSON":
                # append to temp list
                tmp_entities.append(entity.text)
        # append temp list to main list
        #post_entities.append(tmp_entities)
        post_entities.append(set(sorted(tmp_entities)))
    
    '''
    Creating edgelist
    '''
    edgelist = []
    # iterate over every document
    for text in post_entities:
        # use itertools.combinations() to create edgelist
        edges = list(combinations(text, 2))
        # for each combination - i.e. each pair of 'nodes'
        for edge in edges:
            # append this to final edgelist
            edgelist.append(tuple(sorted(edge)))
    
    
    #EBT: weighted edgelist (Tror jeg)
    counted_edges = []
    for key, value in Counter(edgelist).items():
        source = key[0]
        target = key[1]
        weight = value
        counted_edges.append((source, target, weight))
    
    
    edges_df = pd.DataFrame(counted_edges, columns=["nodeA", "nodeB", "weight"])
    
    
    
    '''
    Filtereing based on edgeweight
    '''
    
    
    filtered = edges_df[edges_df["weight"]>args["edges"]]
    G=nx.from_pandas_edgelist(filtered, 'nodeA', 'nodeB', ["weight"])
    outpath_viz = os.path.join("outpath/network.png")
    
    nx.draw_random(G, with_labels=True, node_size=50, font_size=10)
  
    plt.savefig(outpath_viz, dpi=300, bbox_inches='tight')
    #plt.show()
    
    dg = nx.degree_centrality(G)
    ev = nx.eigenvector_centrality(G)
    bc = nx.betweenness_centrality(G)
    
    dg_df = pd.DataFrame(dg.items()).sort_values(weight, ascending=False)
    ev_df = pd.DataFrame(ev.items()).sort_values(weight, ascending=False)
    bc_df = pd.DataFrame(bc.items()).sort_values(weight, ascending=False)
    

    dg_df.columns = ["Name", "degree_centrality"]
    
    bc_df.columns = ["Name", "betweenness_centrality"]

    ev_df.columns = ["Name", "eigenvector_centrality"]

    final_df = pd.merge(bc_df, ev_df, on='Name')
    
    final_df = pd.merge(dg_df, final_df, on='Name')
    
    final_df.to_csv('outpath/final_df.csv') 

    print(final_df.head(10))
    
if __name__ =='__main__':
    main()