import torch.nn as nn
import torch
import numpy as np
import math

from models.modules import ParityBackbone, SynapseUNET, Squeeze, SuperLinear, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)

class ContinuousThoughtMachinePrefrontalCortex(nn.Module):
    """
    Continuous Thought Machine (CTM).

    Technical report: https://arxiv.org/abs/2505.05522

    Interactive Website: https://pub.sakana.ai/ctm/

    Blog: https://sakana.ai/ctm/

    Thought takes time and reasoning is a process. 
    
    The CTM consists of three main ideas:
    1. The use of internal recurrence, enabling a dimension over which a concept analogous to thought can occur. 
    1. Neuron-level models, that compute post-activations by applying private (i.e., on a per-neuron basis) MLP 
       models to a history of incoming pre-activations.
    2. Synchronisation as representation, where the neural activity over time is tracked and used to compute how 
       pairs of neurons synchronise with one another over time. This measure of synchronisation is the representation 
       with which the CTM takes action and makes predictions.


    Args:
        iterations (int): Number of internal 'thought' ticks (T, in paper).
        d_model (int): Core dimensionality of the CTM's latent space (D, in paper).
                       NOTE: Note that this is NOT the representation used for action or prediction, but rather that which
                       is fully internal to the model and not directly connected to data.
        d_input (int): Dimensionality of projected attention outputs or direct input features.
        heads (int): Number of attention heads.
        n_synch_out (int): Number of neurons used for output synchronisation (D_out, in paper).
        n_synch_cortical_action (int): Number of neurons used for action/attention synchronisation (D_action, in paper).
        synapse_depth (int): Depth of the synapse model (U-Net if > 1, else MLP).
        memory_length (int): History length for Neuron-Level Models (M, in paper).
        deep_nlms (bool): Use deeper (2-layer) NLMs if True, else linear.
                        NOTE: we almost always use deep NLMs, but a linear NLM is faster.
        memory_hidden_dims (int): Hidden dimension size for deep NLMs.
        do_layernorm_nlm (bool): Apply LayerNorm within NLMs.
                        NOTE: we never set this to true in the paper. If you set this to true you will get strange behaviour,
                        but you can potentially encourage more periodic behaviour in the dynamics. Untested; be careful.
        backbone_type (str): Type of feature extraction backbone (e.g., 'resnet18-2', 'none').
        positional_embedding_type (str): Type of positional embedding for backbone features.
        out_dims (int): Output dimension size.
                        NOTE: projected from synchronisation!
        prediction_reshaper (list): Shape for reshaping predictions before certainty calculation (task-specific).
                        NOTE: this is used to compute certainty and is needed when applying softmax for probabilities
        dropout (float): Dropout rate.
        neuron_select_type (str): Neuron selection strategy ('first-last', 'random', 'random-pairing').
                        NOTE: some of this is legacy from our experimentation, but all three strategies are valid and useful. 
                            We dilineate exactly which strategies we use per experiment in the paper. 
                        - first-last: build a 'dense' sync matrix for output from the first D_out neurons and action from the 
                                      last D_action neurons. Flatten this matrix into the synchronisation representation. 
                                      This approach shares relationships for neurons and bottlenecks the gradients through them.
                                      NOTE: the synchronisation size will be (D_out/action * (D_out/action + 1))/2
                        - random: randomly select D_out neurons for the 'i' side pairings, and also D_out for the 'j' side pairings,
                                      also pairing those accross densely, resulting in a bottleneck roughly 2x as wide.
                                      NOTE: the synchronisation size will be (D_out/action * (D_out/action + 1))/2
                        - random-pairing (DEFAULT!): randomly select D_out neurons and pair these with another D_out neurons. 
                                      This results in much less bottlenecking and is the most up-to-date variant.
                                      NOTE: the synchronisation size will be D_out in this case; better control. 
        n_random_pairing_self (int): Number of neurons to select for self-to-self synch when random-pairing is used.
                        NOTE: when using random-pairing, i-to-i (self) synchronisation is rare, meaning that 'recovering a
                        snapshot representation' (see paper) is difficult. This alleviates that. 
                        NOTE: works fine when set to 0.
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_pfc_model,
                 d_input,
                 heads,
                 # batch_size,
                 n_synch_cortical_action,
                 n_synch_summary,
                 n_synch_out,
                 synapse_depth,
                 pfc_synapse_depth,
                 memory_length,
                 pfc_memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 summary_dims,
                 out_dims,
                 pfc_action_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 ):
        super(ContinuousThoughtMachinePrefrontalCortex, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.prediction_reshaper = prediction_reshaper
        self.n_synch_cortical_action = n_synch_cortical_action
        self.n_synch_summary = n_synch_summary
        self.n_synch_out = n_synch_out
        # self.batch_size = batch_size
        self.backbone_type = backbone_type
        self.out_dims = out_dims
        self.positional_embedding_type = positional_embedding_type
        self.neuron_select_type = neuron_select_type
        self.memory_length = memory_length
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm

        # --- Assertions ---
        self.verify_args()

        # --- Input Processing  ---
        d_backbone = self.get_d_backbone()
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input)) if heads else None
        self.q_proj = nn.LazyLinear(self.d_input) if heads else None
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True) if heads else None

        # --- Cortical CTM ---
        self.cortical_synapse_model = self.get_synapses(synapse_depth, d_model, dropout)
        self.cortical_neuron_level_model = self.get_neuron_level_models(deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout_nlm)
        self.register_parameter('cortical_start_activated_state', nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model)))))
        self.register_parameter('cortical_start_trace', nn.Parameter(torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length)))))
        self.synch_representation_size_cortical_action = self.calculate_synch_representation_size(n_synch_cortical_action)
        self.synch_representation_size_summary         = self.calculate_synch_representation_size(n_synch_summary)
        self.synch_representation_size_out             = self.calculate_synch_representation_size(n_synch_out)
        for synch_type, size in [ ('cortical_action', self.synch_representation_size_cortical_action ),
                                  ('summary',         self.synch_representation_size_summary         ),
                                  ('out',             self.synch_representation_size_out             ) ]:
            print(f"Cortical Synch representation size '{synch_type}': {size}")
        self.set_synchronisation_parameters('cortical_action', n_synch_cortical_action, n_random_pairing_self)
        self.set_synchronisation_parameters('summary',         n_synch_summary,         n_random_pairing_self)
        self.set_synchronisation_parameters('out',             n_synch_out,             n_random_pairing_self)
        self.summary_projector = nn.Sequential(nn.LazyLinear(summary_dims))
        self.output_projector  = nn.Sequential(nn.LazyLinear(self.out_dims))

        # --- PrefrontalCortex CTM ---
        pfc_model = 'ctm'  # Currently only CTM is supported for PFC
        if pfc_model == "ctm":
            self.pfc_synapse_model = self.get_synapses(pfc_synapse_depth, d_pfc_model, dropout)
            self.pfc_neuron_level_model = self.get_neuron_level_models(deep_nlms, do_layernorm_nlm, pfc_memory_length, memory_hidden_dims, d_pfc_model, dropout_nlm)
            self.register_parameter('pfc_start_activated_state', nn.Parameter(torch.zeros((d_pfc_model)).uniform_(-math.sqrt(1/(d_pfc_model)), math.sqrt(1/(d_pfc_model)))))
            self.register_parameter('pfc_start_trace', nn.Parameter(torch.zeros((d_pfc_model, pfc_memory_length)).uniform_(-math.sqrt(1/(d_pfc_model+pfc_memory_length)), math.sqrt(1/(d_pfc_model+pfc_memory_length)))))
            n_synch_pfc = (d_pfc_model * (d_pfc_model + 1)) // 2
            print(f"PrefrontalCortex Synch representation sizes: {n_synch_pfc}")
            self.set_synchronisation_parameters('pfc_action', n_synch_pfc, n_random_pairing_self)
            self.set_synchronisation_parameters('gate',       n_synch_pfc, n_random_pairing_self)
            self.pfc_action_projector = nn.Sequential(nn.LazyLinear(pfc_action_dims))
            self.gate_projector = nn.Sequential(nn.LazyLinear(
                self.synch_representation_size_cortical_action +
                self.synch_representation_size_summary +
                self.synch_representation_size_out
            ), nn.Sigmoid())
        else:
            raise NotImplementedError

    # --- Core CTM Methods ---

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        """
        Computes synchronisation to be used as a vector representation. 

        A neuron has what we call a 'trace', which is a history (time series) that changes with internal
        recurrence. i.e., it gets longer with every internal tick. There are pre-activation traces
        that are used in the NLMs and post-activation traces that, in theory, are used in this method. 

        We define sychronisation between neuron i and j as the dot product between their respective
        time series. Since there can be many internal ticks, this process can be quite compute heavy as it
        involves many dot products that repeat computation at each step.
        
        Therefore, in practice, we update the synchronisation based on the current post-activations,
        which we call the 'activated state' here. This is possible because the inputs to synchronisation 
        are only updated recurrently at each step, meaning that there is a linear recurrence we can
        leverage. 
        
        See Appendix TODO of the Technical Report (TODO:LINK) for the maths that enables this method.
        """

        if synch_type in ('cortical_action', 'summary', 'out'):

            if synch_type == 'cortical_action':
                neuron_indices_left  = self.cortical_action_neuron_indices_left
                neuron_indices_right = self.cortical_action_neuron_indices_right
            elif synch_type == 'summary':
                neuron_indices_left  = self.summary_neuron_indices_left
                neuron_indices_right = self.summary_neuron_indices_right
            elif synch_type == 'out':
                neuron_indices_left  = self.out_neuron_indices_left
                neuron_indices_right = self.out_neuron_indices_right

            if self.neuron_select_type in ('first-last', 'random'):
                left  = activated_state[:, neuron_indices_left]
                right = activated_state[:, neuron_indices_right]
                outer = left.unsqueeze(2) * right.unsqueeze(1)
                N = left.shape[1]
                i, j = torch.triu_indices(N, N)
                pairwise_product = outer[:, i, j]

            else:
                # B = activated_state.shape[0]
                # K = neuron_indices_left.shape[1]
                # batch_idx = torch.arange(B).unsqueeze(-1).expand(-1, K).to(activated_state.device)
                # left  = activated_state[batch_idx, neuron_indices_left ]
                # right = activated_state[batch_idx, neuron_indices_right]
                left  = activated_state[:, neuron_indices_left]
                right = activated_state[:, neuron_indices_right]
                pairwise_product = left * right

        elif synch_type in ('pfc_action', 'gate'):
            N = activated_state.shape[1]
            # Compute outer product of selected neurons
            outer = activated_state.unsqueeze(2) * activated_state.unsqueeze(1)
            # Resulting matrix is symmetric, so we only need the upper triangle
            i, j = torch.triu_indices(N, N)
            pairwise_product = outer[:, i, j]

        else:
            raise ValueError(f"Invalid synch_type: {synch_type}")
        
        # Compute synchronisation recurrently
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        
        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        """
        Compute the key-value features from the input data using the backbone. 
        """
        initial_rgb = self.initial_rgb(x)
        self.kv_features = self.backbone(initial_rgb)
        pos_emb = self.positional_embedding(self.kv_features)
        combined_features = (self.kv_features + pos_emb).flatten(2).transpose(1, 2)
        kv = self.kv_proj(combined_features)
        return kv

    def compute_certainty(self, current_prediction):
        """
        Compute the certainty of the current prediction.
        
        We define certainty as being 1-normalised entropy.

        For legacy reasons we stack that in a 2D vector as this can be used for optimisation later.
        """
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    # --- Setup Methods ---

    def set_initial_rgb(self):
        """
        This is largely to accommodate training on grayscale images and is legacy, but it
        doesn't hurt the model in any way that we can tell.
        """
        if 'resnet' in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1) # Adapts input channels lazily
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        """
        Get the dimensionality of the backbone output, to be used for positional embedding setup.

        This is a little bit complicated for resnets, but the logic should be easy enough to read below.        
        """
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            if '18' in self.backbone_type or '34' in self.backbone_type: 
                if self.backbone_type.split('-')[1]=='1': return 64
                elif self.backbone_type.split('-')[1]=='2': return 128
                elif self.backbone_type.split('-')[1]=='3': return 256
                elif self.backbone_type.split('-')[1]=='4': return 512
                else:
                    raise NotImplementedError
            else:
                if self.backbone_type.split('-')[1]=='1': return 256
                elif self.backbone_type.split('-')[1]=='2': return 512
                elif self.backbone_type.split('-')[1]=='3': return 1024
                elif self.backbone_type.split('-')[1]=='4': return 2048
                else:
                    raise NotImplementedError
        elif self.backbone_type == 'none':
            return None
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        """
        Set the backbone module based on the specified type.
        """
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        """
        Get the positional embedding module.

        For Imagenet and mazes we used NO positional embedding, and largely don't think
        that it is necessary as the CTM can build up its own internal world model when
        observing.

        LearnableFourierPositionalEncoding:
            Implements Algorithm 1 from "Learnable Fourier Features for Multi-Dimensional
            Spatial Positional Encoding" (https://arxiv.org/pdf/2106.02795.pdf).
            Provides positional information for 2D feature maps.      

            (MultiLearnableFourierPositionalEncoding uses multiple feature scales)

        CustomRotationalEmbedding:
            Simple sinusoidal embedding to encourage interpretability
        """
        if self.positional_embedding_type == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        elif self.positional_embedding_type == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif self.positional_embedding_type == 'none':
            return lambda x: 0  # Default no-op
        else:
            raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout):
        """
        Neuron level models are one of the core innovations of the CTM. They apply separate MLPs/linears to 
        each neuron.
        NOTE: the name 'SuperLinear' is largely legacy, but its purpose is to apply separate linear layers
            per neuron. It is sort of a 'grouped linear' function, where the group size is equal to 1. 
            One could make the group size bigger and use fewer parameters, but that is future work.

        NOTE: We used GLU() nonlinearities because they worked well in practice. 
        """
        if deep_nlms:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        else:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )

    def get_synapses(self, synapse_depth, d_model, dropout):
        """
        The synapse model is the recurrent model in the CTM. It's purpose is to share information
        across neurons. If using depth of 1, this is just a simple single layer with nonlinearity and layernomr.
        For deeper synapse models we use a U-NET structure with many skip connections. In practice this performs
        better as it enables multi-level information mixing.

        The intuition with having a deep UNET model for synapses is that the action of synaptic connections is
        not necessarily a linear one, and that approximate a synapose 'update' step in the brain is non trivial. 
        Hence, we set it up so that the CTM can learn some complex internal rule instead of trying to approximate
        it ourselves.
        """
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, synapse_depth, 16, dropout)  # hard-coded minimum width of 16; future work TODO.

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):
        """
        1. Set the buffers for selecting neurons so that these indices are saved into the model state_dict.
        2. Set the parameters for learnable exponential decay when computing synchronisation between all neurons.
        """
        assert synch_type in ('cortical_action', 'summary', 'out', 'pfc_action', 'gate'), \
            f"Invalid synch_type: {synch_type}"
        if synch_type in ('cortical_action', 'summary', 'out'):
            left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
            self.register_buffer(f'{synch_type}_neuron_indices_left',  left)
            self.register_buffer(f'{synch_type}_neuron_indices_right', right)
        if synch_type == 'cortical_action':
            synch_representation_size = self.synch_representation_size_cortical_action
        elif synch_type == 'summary':
            synch_representation_size = self.synch_representation_size_summary
        elif synch_type == 'out':
            synch_representation_size = self.synch_representation_size_out
        else:
            synch_representation_size = n_synch
        self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
        """
        Initialize the left and right neuron indices based on the neuron selection type.
        This complexity is owing to legacy experiments, but we retain that these types of
        neuron selections are interesting to experiment with.
        """
        if self.neuron_select_type=='first-last':
            if synch_type == 'cortical_action':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
            if synch_type == 'summary':
                neuron_indices_left = neuron_indices_right = torch.arange(self.n_synch_cortical_action, self.n_synch_cortical_action+n_synch)
            elif synch_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)

        elif self.neuron_select_type=='random':
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))

        elif self.neuron_select_type=='random-pairing':
            assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for random-pairing."
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self))))

        # neuron_indices_left  = neuron_indices_left.unsqueeze(0).repeat(self.batch_size, 1)
        # neuron_indices_right = neuron_indices_right.unsqueeze(0).repeat(self.batch_size, 1)

        device = self.cortical_start_activated_state.device
        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    # --- Utilty Methods ---

    def verify_args(self):
        """
        Verify the validity of the input arguments to ensure consistent behaviour. 
        Specifically when selecting neurons for sychronisation using 'first-last' or 'random',
        one needs the right number of neurons
        """
        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron selection type: {self.neuron_select_type}"
        
        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"
        
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"
        
        # if self.neuron_select_type == 'first-last':
        #     assert self.d_model >= (self.n_synch_out + self.n_synch_cortical_action), \
        #         "d_model must be >= n_synch_out + n_synch_cortical_action for neuron subsets"

        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")

    def calculate_synch_representation_size(self, n_synch):
        """
        Calculate the size of the synchronisation representation based on neuron selection type.
        """
        if self.neuron_select_type == 'random-pairing':
            synch_representation_size = n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            synch_representation_size = (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return synch_representation_size



    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        cortical_pre_activations_tracking  = []
        cortical_post_activations_tracking = []
        # pfc_pre_activations_tracking       = []
        # pfc_post_activations_tracking      = []
        synch_action_tracking              = []
        # synch_summary_tracking             = []
        synch_out_tracking                 = []
        # synch_pfc_action_tracking          = []
        # synch_gate_tracking                = []
        attention_tracking                 = []
        gate_tracking                      = []

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        cortical_state_trace     = self.cortical_start_trace.unsqueeze(0).expand(B, -1, -1)          # Shape: (B, H, T)
        cortical_activated_state = self.cortical_start_activated_state.unsqueeze(0).expand(B, -1)    # Shape: (B, H)
        pfc_state_trace          = self.pfc_start_trace.unsqueeze(0).expand(B, -1, -1)
        pfc_activated_state      = self.pfc_start_activated_state.unsqueeze(0).expand(B, -1)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations,             device=device, dtype=torch.float32)

        # --- Initialise Recurrent Synch Values  ---
        self.decay_params_cortical_action.data = torch.clamp(self.decay_params_cortical_action, 0, 15)
        self.decay_params_summary.data         = torch.clamp(self.decay_params_summary,         0, 15)
        self.decay_params_out.data             = torch.clamp(self.decay_params_out,             0, 15)
        self.decay_params_pfc_action.data      = torch.clamp(self.decay_params_pfc_action,      0, 15)
        self.decay_params_gate.data            = torch.clamp(self.decay_params_gate,            0, 15)
        r_cortical_action = torch.exp(-self.decay_params_cortical_action).unsqueeze(0).repeat(B, 1)
        r_summary         = torch.exp(-self.decay_params_summary        ).unsqueeze(0).repeat(B, 1)
        r_out             = torch.exp(-self.decay_params_out            ).unsqueeze(0).repeat(B, 1)
        r_pfc_action      = torch.exp(-self.decay_params_pfc_action     ).unsqueeze(0).repeat(B, 1)
        r_gate            = torch.exp(-self.decay_params_gate           ).unsqueeze(0).repeat(B, 1)
        decay_alpha_action,     decay_beta_action     = None, None
        decay_alpha_summary,    decay_beta_summary    = None, None
        decay_alpha_out,        decay_beta_out        = self.compute_synchronisation(cortical_activated_state, None, None, r_out, synch_type='out')[1:]
        decay_alpha_pfc_action, decay_beta_pfc_action = None, None
        decay_alpha_gate,       decay_beta_gate       = None, None
        
        current_gate_action  = torch.ones((B, self.synch_representation_size_cortical_action), device=device)
        current_gate_summary = torch.ones((B, self.synch_representation_size_summary),         device=device)
        current_gate_out     = torch.ones((B, self.synch_representation_size_out),             device=device)

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(cortical_activated_state, decay_alpha_action, decay_beta_action, r_cortical_action, synch_type='cortical_action')
            synchronisation_action = synchronisation_action * current_gate_action

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            cortical_pre_synapse_input = torch.concatenate((attn_out, cortical_activated_state), dim=-1)

            # --- Apply Cortical Synapses ---
            state = self.cortical_synapse_model(cortical_pre_synapse_input)
            cortical_state_trace = torch.cat((cortical_state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Cortical Neuron-Level Models ---
            cortical_activated_state = self.cortical_neuron_level_model(cortical_state_trace)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(cortical_activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')
            synchronisation_out = synchronisation_out * current_gate_out

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty  = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Calculate Synchronisation for Summary ---
            synchronisation_summary, decay_alpha_summary, decay_beta_summary = self.compute_synchronisation(cortical_activated_state, decay_alpha_summary, decay_beta_summary, r_summary, synch_type='summary')
            synchronisation_summary = synchronisation_summary * current_gate_summary
            current_summary = self.summary_projector(synchronisation_summary)

            # --- Calculate Synchronisation for PFC action ---
            synchronisation_pfc_action, decay_alpha_pfc_action, decay_beta_pfc_action = self.compute_synchronisation(pfc_activated_state, decay_alpha_pfc_action, decay_beta_pfc_action, r_pfc_action, synch_type='pfc_action')
            current_pfc_action = self.pfc_action_projector(synchronisation_pfc_action)
            
            # --- Apply PFC Synapses ---
            pfc_pre_synapse_input = torch.concatenate((current_summary, current_pfc_action, pfc_activated_state), dim=-1)
            state = self.pfc_synapse_model(pfc_pre_synapse_input)
            pfc_state_trace = torch.cat((pfc_state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply PFC Neuron-Level Models ---
            pfc_activated_state = self.pfc_neuron_level_model(pfc_state_trace)
            synchronisation_gate, decay_alpha_gate, decay_beta_gate = self.compute_synchronisation(pfc_activated_state, decay_alpha_gate, decay_beta_gate, r_gate, synch_type='gate')
            current_gate = self.gate_projector(synchronisation_gate)
            current_gate_action, current_gate_summary, current_gate_out = torch.split(current_gate, [
                self.synch_representation_size_cortical_action,
                self.synch_representation_size_summary,
                self.synch_representation_size_out
            ], dim=-1)

            # --- Tracking ---
            if track:
                cortical_pre_activations_tracking.append(cortical_state_trace[:,:,-1].detach().cpu().numpy())
                cortical_post_activations_tracking.append(cortical_activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())
                gate_tracking.append(current_gate.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(cortical_pre_activations_tracking), np.array(cortical_post_activations_tracking), np.array(attention_tracking), np.array(gate_tracking)
        return predictions, certainties, synchronisation_out
