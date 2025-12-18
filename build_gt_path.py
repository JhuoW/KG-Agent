import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import argparse
from tqdm import tqdm
import os.path as osp
from datasets import load_dataset, Dataset
from multiprocessing import Pool
from functools import partial
from utils.graph_utils import build_graph, get_truth_paths

def process(sample, undirected = False):
    graph = build_graph(sample['graph'], undirected=undirected)
    start_nodes = sample['q_entity']
    answer_nodes = sample['a_entity']
    paths_list = get_truth_paths(start_nodes, answer_nodes, graph) # all shortest paths from q_entity to a_entity as ground truth reasoning path
    sample['ground_truth_paths'] = paths_list  # all ground truth paths of a sample question
    return sample


def index_graph(args, dataset_name):
    input_file = osp.join(args.data_path, dataset_name)  # rmanluo/RoG-webqsp or rmanluo/RoG-cwq
    data_path = f"{dataset_name}_undirected" if args.undirected else dataset_name
    output_dir = osp.join(args.output_path, data_path, args.split)
    # download rmanluo/RoG-webqsp train split from huggingface
    dataset = load_dataset(input_file, split=args.split)  
    results = []
    with Pool(processes=args.num_processes) as pool:
        for res in tqdm(pool.imap_unordered(partial(process, undirected = args.undirected), dataset)): 
            # question: what is the name of justin bieber brother
            # answer: ["Jaxon Bieber"] 
            # q_entity: ["Justin Bieber"]    
            # a_entity: ["Jaxon Bieber"]  
            # graph: [[
            #     "P!nk",
            #     "freebase.valuenotation.is_reviewed",
            #     "Gender"
            #   ],
            #   [
            #     "1Club.FM: Power",
            #     "broadcast.content.artist",
            #     "P!nk"
            #   ],
            #   [
            #     "Somebody to Love",
            #     "music.recording.contributions",
            #     "m.0rqp4h0"
            #   ],
            #   [
            #     "Rudolph Valentino",
            #     "freebase.valuenotation.is_reviewed",
            #     "Place of birth"
            #   ],
            results.append(res)
    
    index_dataset = Dataset.from_list(results)
    index_dataset.save_to_disk(output_dir) # add ground truth paths to each sample and save to disk



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='rmanluo') 
    parser.add_argument('--dataset', '-d', type=str, default='RoG-webqsp', choices=['RoG-webqsp', 'RoG-cwq'])
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_path', type = str, default='data/shortest_gt_paths')
    parser.add_argument('--undirected', action='store_true', help='whether the graph is undirected') # False
    parser.add_argument('--num_processes',type=int, default=8, help='number of processes')
    args = parser.parse_args()

    for dataset_name in ['RoG-webqsp', 'RoG-cwq']:
        index_graph(args, dataset_name)