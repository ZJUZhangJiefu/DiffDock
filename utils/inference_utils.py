import os
import esm
import yaml
import torch
import pandas as pd

from esm import pretrained, FastaBatchedDataset
from argparse import ArgumentParser, Namespace
from torch_geometric.data import Dataset
from Bio.PDB import PDBParser
from tqdm import tqdm

from utils.utils import get_model

# argument parser for inference
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
    parser.add_argument('--complex_name', type=str, default='1a0q', help='Name that the complex will be saved with')
    parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
    parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
    parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')

    parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
    parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
    parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

    parser.add_argument('--model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained score model and hyperparameters')
    parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
    parser.add_argument('--confidence_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')
    return parser.parse_args()       

# load the arguments of score model and confidence model  
def load_model_arguments(model_dir, confidence_model_dir):
    with open(f'{model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    if confidence_model_dir is not None:
        with open(f'{confidence_model_dir}/model_parameters.yml') as f:
            confidence_model_args = Namespace(**yaml.full_load(f))
    else:
        confidence_model_args = None
    return score_model_args, confidence_model_args

def set_nones(l):
    return [s if str(s) != 'nan' else None for s in l]

def load_complex_info_from_csv(args):
    if args.protein_ligand_csv is not None:
        df = pd.read_csv(args.protein_ligand_csv)
        complex_name_list = set_nones(df['complex_name'].tolist())
        protein_path_list = set_nones(df['protein_path'].tolist())
        protein_sequence_list = set_nones(df['protein_sequence'].tolist())
        ligand_description_list = set_nones(df['ligand_description'].tolist())
    else:
        complex_name_list = [args.complex_name]
        protein_path_list = [args.protein_path]
        protein_sequence_list = [args.protein_sequence]
        ligand_description_list = [args.ligand_description]
    complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
    for name in complex_name_list:
        write_dir = f'{args.out_dir}/{name}'
        os.makedirs(write_dir, exist_ok=True)
    return complex_name_list, protein_path_list, protein_sequence_list, ligand_description_list

def three_to_one(residue_name):
    look_up_table = {'ALA':	'A', 'ARG':	'R', 'ASN':	'N', 'ASP':	'D',
    'CYS':	'C', 'GLN':	'Q', 'GLU':	'E', 'GLY':	'G',
    'HIS':	'H', 'ILE':	'I', 'LEU':	'L', 'LYS':	'K',
    'MET':	'M', 'MSE': 'M', 'PHE':	'F', 'PRO':	'P',
    'PYL':	'O', 'SER':	'S', 'SEC':	'U', 'THR':	'T',
    'TRP':	'W', 'TYR':	'Y', 'VAL':	'V', 'ASX':	'B',
    'GLX':	'Z', 'XAA':	'X','XLE':	'J'}
    try:
        new_name = three_to_one[residue_name]
    except Exception as e:
        new_name =  '-'
        print("encountered unknown AA: ", residue_name, ' in the complex. Replacing it with a dash - .')
    return new_name
        

def get_sequence_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                seq += three_to_one(residue.get_resname())

        if sequence is None:
            sequence = seq
        else:
            sequence += (":" + seq)

    return sequence

def update_protein_sequences(protein_sequences, protein_files):
    updated_protein_sequences = []
    for i in range(len(protein_files)):
        if protein_files[i] is None:
            updated_protein_sequences.append(protein_sequences[i])
        else:
            updated_protein_sequences.append(get_sequence_from_pdbfile(protein_files[i]))
    return updated_protein_sequences

def original_ESM_embedding(model, alphabet, labels, sequences, device):
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    include = "per_tok"
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device=device, non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1: truncate_len + 1].clone()
    return embeddings

def generate_esm_embedding_from_residue_string(generate_lm_embeddings, protein_sequences, protein_files, complex_names, device):
    if generate_lm_embeddings:
        # load esm pretrained language model 
        print("Generating ESM language model embeddings")
        model_location = "esm2_t33_650M_UR50D"
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()
        if(torch.cuda.is_available()):
            model = model.to(device)
        
        # update the protein sequences from protein file
        protein_sequences = update_protein_sequences(protein_sequences, protein_files)
        
        # prepare labels, sequences to feed the esm pretrained model
        labels, sequences = [], []
        for i in range(len(protein_sequences)):
            s = protein_sequences[i].split(':')
            sequences.extend(s)
            labels.extend([complex_names[i] + '_chain_' + str(j) for j in range(len(s))])
        
        # call esm to generate the original embedding
        original_embeddings = original_ESM_embedding(model=model, alphabet = alphabet, labels=labels,
                                                    sequences=sequences, device=device)
        
        # adjust the format of embedding data 
        lm_embeddings = []
        for i in range(len(protein_sequences)):
            s = protein_sequences[i].split(':')
            lm_embeddings.append([original_embeddings[f'{complex_names[i]}_chain_{j}'] for j in range(len(s))])
        
        # release the memory of gpu after language model has been used
        del model
    else:
        lm_embeddings = [None] * len(complex_names)
    return lm_embeddings

def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None

class InferenceDataset(Dataset):
    def __init__(self, out_dir, complex_names, protein_files, ligand_descriptions,
                 protein_sequences, generate_lm_embeddings, device, 
                 receptor_radius=30, c_alpha_max_neighbors=None, 
                 remove_hs=False, all_atoms=False, atom_radius=5, atom_max_neighbors=None):
        # arguments belonging to the InferenceDataset class
        self.atom_max_neighbors = atom_max_neighbors # max number of neighbors of the atoms
        self.atom_radius = atom_radius # radius of atoms 
        self.all_atoms = all_atoms # keep all atoms
        self.remove_hs = remove_hs # remove all the hydrogens  
        self.receptor_radius = receptor_radius # radius of receptor
        self.c_alpha_max_neighbors = c_alpha_max_neighbors # max num of neighbors of the residues(identified by c-alpha)
        
        self.out_dir = out_dir
        self.complex_names = complex_names
        self.protein_files = protein_files
        self.ligand_descriptions = ligand_descriptions
        self.protein_sequences = protein_sequences
        self.device = device
        
        # generate ESM embeddings of protein strings  
        self.lm_embeddings = generate_esm_embedding_from_residue_string(generate_lm_embeddings, protein_sequences, protein_files,
                                                                        complex_names, device)
        
        # generate missing protein structure with esmfold
        self.generate_missing_protein_structure()
        return
        
    def generate_missing_protein_structure(self):
        if None in self.protein_files:
            print("generating missing structures with ESMFold")
            model = esm.pretrained.esmfold_v1()
            model = model.eval().to(self.device)

            for i in range(len(self.protein_files)):
                if self.protein_files[i] is None:
                    self.protein_files[i] = f"{self.out_dir}/{self.complex_names[i]}/{self.complex_names[i]}_esmfold.pdb"
                    if not os.path.exists(self.protein_files[i]):
                        print("generating", self.protein_files[i])
                        generate_ESM_structure(model, self.protein_files[i], self.protein_sequences[i])
        return
    def len(self):
        return len(self.complex_names)
    def get(self, id):
        pass
    
def get_confidence_model(args, confidence_args, t_to_sigma, device):
    if args.confidence_model_dir is not None:
        confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
        state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None
        confidence_args = None
    return confidence_model, confidence_args
    
def inference(test_dataset, test_loader, args):
    failures = []
    skipped = []
    
    N = args.samples_per_complex
    print('Size of test dataset: ', len(test_dataset))
    for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
        if not orig_complex_graph.success[0]:
            skipped += 1
            print(f"HAPPENING | The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
            continue
        
    
    
    
    
    print(f'Failed for {failures} complexes')
    print(f'Skipped {skipped} complexes')
    print(f'Results are in {args.out_dir}')
    return