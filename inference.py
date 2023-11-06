import os
import torch

from functools import partial
from torch_geometric.loader import DataLoader

from utils.utils import get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.inference_utils import parse_arguments, load_model_arguments, load_complex_info_from_csv
from utils.inference_utils import InferenceDataset, inference, get_confidence_model

# please set the gpu you want to use
gpu_to_use = "cuda:1"

# set gpu
os.environ["CUDA_VISIBLE_DEVICE"] = '1'  
device = torch.device(gpu_to_use if torch.cuda.is_available() else 'cpu')  

# parse arguments, make the output directory and load model arguments from file
args = parse_arguments()
os.makedirs(args.out_dir, exist_ok=True)
score_model_args, confidence_model_args = load_model_arguments(args.model_dir, args.confidence_model_dir)

# load complex information from csv file
complex_name_list, protein_path_list, protein_sequence_list, ligand_description_list = load_complex_info_from_csv(args)

# preprocessing the complexes into geometric graphs, and load them into test_loader
test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                lm_embeddings=score_model_args.esm_embeddings_path is not None,
                                receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                atom_max_neighbors=score_model_args.atom_max_neighbors)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# load test dataset for the confidence model, if required in args
if args.confidence_model_dir is not None and not confidence_model_args.use_original_model_cache:
    print('HAPPENING | confidence model uses different type of graphs than the score model. '
          'Loading (or creating if not existing) the data for the confidence model now.')
    confidence_test_dataset = \
        InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                         ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                         lm_embeddings=confidence_model_args.esm_embeddings_path is not None,
                         receptor_radius=confidence_model_args.receptor_radius, remove_hs=confidence_model_args.remove_hs,
                         c_alpha_max_neighbors=confidence_model_args.c_alpha_max_neighbors,
                         all_atoms=confidence_model_args.all_atoms, atom_radius=confidence_model_args.atom_radius,
                         atom_max_neighbors=confidence_model_args.atom_max_neighbors,
                         precomputed_lm_embeddings=test_dataset.lm_embeddings)
else:
    confidence_test_dataset = None

# bound the function t_to_sigma_compl with the specified arguments in score_model_args
t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

# load the score model for inference
model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

# load the confidence model
confidence_model, confidence_model_args = get_confidence_model(args=args, confidence_args=confidence_model_args, 
                                                               t_to_sigma=t_to_sigma, device=device)

# perform inference
inference(test_loader=test_loader, args=args)