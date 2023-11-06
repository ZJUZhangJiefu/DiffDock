from utils.diffusion_utils import get_timestep_embedding
from torch_geometric.nn.data_parallel import DataParallel

from models.all_atom_score_model import TensorProductScoreModel as AAScoreModel
from models.score_model import TensorProductScoreModel as CGScoreModel

def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False):
    if 'all_atoms' in args and args.all_atoms:
        # tensor product score model for all atoms
        model_class = AAScoreModel
    else:
        # default choice 
        model_class = CGScoreModel
    
    # get timestep embedding function
    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type,
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale)

    # the embedding type of residue strings, e.g. 'esm'
    lm_embedding_type = None
    if args.esm_embeddings_path is not None: lm_embedding_type = 'esm'

    # instantialize a model from the required model class
    model = model_class(t_to_sigma=t_to_sigma,
                        device=device,
                        no_torsion=args.no_torsion,
                        timestep_emb_func=timestep_emb_func,
                        num_conv_layers=args.num_conv_layers,
                        lig_max_radius=args.max_radius,
                        scale_by_sigma=args.scale_by_sigma,
                        sigma_embed_dim=args.sigma_embed_dim,
                        ns=args.ns, nv=args.nv,
                        distance_embed_dim=args.distance_embed_dim,
                        cross_distance_embed_dim=args.cross_distance_embed_dim,
                        batch_norm=not args.no_batch_norm,
                        dropout=args.dropout,
                        use_second_order_repr=args.use_second_order_repr,
                        cross_max_distance=args.cross_max_distance,
                        dynamic_max_cross=args.dynamic_max_cross,
                        lm_embedding_type=lm_embedding_type,
                        confidence_mode=confidence_mode,
                        num_confidence_outputs=len(
                            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                            args.rmsd_classification_cutoff, list) else 1)

    # parallel computing
    if device.type == 'cuda' and not no_parallel:
        model = DataParallel(model)
        
    # put model into gpu and return
    model.to(device)
    return model