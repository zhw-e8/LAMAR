import pandas as pd
import numpy as np
import json
import tqdm


def label_seq(strand, tx_start, tx_end, exon_starts, exon_ends, method):
    """
    Label each nucleotide for pre-mRNA sequence.
    Consider splice site (method='splice site'):
        0: Not splice site
        1: The first nucleotide of 5' splice site (G in GT)
        2: The second nucleotide of 3' splice site (G in AG)
    Consider splice site (method='exon'), used in SpliceAI:
        0: Not splice site
        1: The end position of exon (pre-mRNA in 5' -> 3')
        2: The start position of exon (pre-mRNA in 5' -> 3')
    Args:
        strand(str): strand
        tx_start(int): start position of pre-mRNA
        tx_end(int): end position of pre-mRNA
        exon_starts(np.array): start position of exon
        exon_ends(np.array): end position of exon
    Returns:
        label(np.array): label of pre-mRNA
    """
    seq_len = tx_end - tx_start + 1
    if method == 'splice site':
        if strand == '+':
            ss5s = exon_ends - tx_start + 1
            ss3s = exon_starts - tx_start - 1
        elif strand == '-':
            ss3s = tx_end - exon_ends - 1
            ss5s = tx_end - exon_starts + 1
    elif method == 'exon':
        if strand == '+':
            ss5s = exon_ends - tx_start
            ss3s = exon_starts - tx_start
        elif strand == '-':
            ss3s = tx_end - exon_ends
            ss5s = tx_end - exon_starts
    label = np.zeros(seq_len)
    label[ss5s] = 2
    label[ss3s] = 1
    return label


def pad_seq(seq, label, pred_len, flank_len):
    """
    Padding pre-mRNA sequence for fine-tuning.
    First, padding pre-mRNA with Ns in the 3' until the length is multiple of pred_len.
    Second, padding pre-mRNA with flank_len Ns in the two side.
    Args:
        seq(str): input pre-mRNA sequence
        label(np.array): input labels
        pred_len(int): the length of sequence predicted by the model
        flank_len(int): the length of flanking region on the two side of pre-mRNA
    Returns:
        seq(str): output padded sequence
        label(np.array): output updated labels
    """
    seq_len = len(seq)
    pad_len = pred_len - seq_len % pred_len
    seq = ['N'] * flank_len + list(seq) + ['N'] * pad_len + ['N'] * flank_len
    label = [-100] * flank_len + list(label) + [-100] * pad_len + [-100] * flank_len
    seq = "".join(seq)
    label = np.array(label)
    return seq, label


def split_seq(seq, label, pred_len, flank_len):
    """
    Split padded pre-mRNA sequences into segments of pred_len+2*flank_len nt.
    The segments are overlapped (2*flank_len nt).
    Args:
        seq(str): input padded pre-mRNA sequence
        label(np.array): input padded labels
        pred_len(int): the length of sequence predicted by the model
        flank_len(int): the length of flanking region on the two side of pre-mRNA
    Returns:
        sub_seqs(list(str)): output segments
        sub_labels(list(list(int))): output labels of segments
    """
    n = int((len(seq) - 2 * flank_len) / pred_len)
    window_len = pred_len + 2 * flank_len
    starts = pred_len * np.arange(n)
    ends = starts + window_len
    sub_seqs, sub_labels = [], []
    for i in range(n):
        sub_seq = seq[starts[i]:ends[i]]
        sub_label = label[starts[i]:ends[i]].copy()
        sub_label[0:flank_len] = -100
        sub_label[-flank_len:] = -100
        sub_label = [-100] + list(map(int, list(sub_label))) + [-100] # Transform np.float to int, because json does not support np.int
        sub_seqs.append(sub_seq)
        sub_labels.append(sub_label)
    return sub_seqs, sub_labels


def process_seq(fa_path, info_path, json_path, method, pred_len=768, flank_len=128):
    """
    Label, pad and split the pre-mRNAs from SpliceAI project.
    Args:
        fa_path(str): input fasta file containing pre-mRNAs
        info_path(str): input file containing positions of splice sites
        json_path(str): output json file containing processed sequences and labels
    """
    pre_mrnas = []
    with open(fa_path) as f:
        for line in tqdm.tqdm(f):
            if not line.startswith('>'):
                pre_mrnas.append(line.strip())
    
    results = []
    info_df = pd.read_csv(info_path, sep='\t')
    for i in tqdm.trange(len(info_df)):
        pre_mrna = pre_mrnas[i]
        strand = info_df.iloc[i, 2]
        tx_start = info_df.iloc[i, 3]
        tx_end = info_df.iloc[i, 4]
        exon_starts = np.array(info_df.iloc[i, 6].split(',')[:-1], dtype='int')
        exon_ends = np.array(info_df.iloc[i, 5].split(',')[:-1], dtype='int')
        # Labeling
        pre_mrna_label = label_seq(strand, tx_start, tx_end, exon_starts, exon_ends, method)
        # Padding
        padding_seq, padding_label = pad_seq(pre_mrna, pre_mrna_label, pred_len, flank_len)
        # Splitting
        sub_seqs, sub_labels = split_seq(padding_seq, padding_label, pred_len, flank_len)
        for j in range(len(sub_seqs)):
            result = {'seq': sub_seqs[j], 'label': sub_labels[j]}
            results.append(result)
    with open(json_path, 'w', encoding='utf-8') as fo:
        json.dump(results, fo, separators=[',', ':'])