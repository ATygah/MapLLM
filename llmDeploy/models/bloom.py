import torch
import sys
import os
import yaml
from typing import Dict, Union, Tuple, List, Optional, Any
import pandas as pd

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.llm import LLM

class BLOOM:
    """
    Implementation of BLOOM model using the LLM framework.
    """
    def __init__(
        self,
        noc_rows: int = 400,
        noc_cols: int = 400,
        pe_memory_size: int = 128 * 1024,
        mapping_strategy: str = "grid_wise",
        split_strategy: str = "hybrid_split",
        data_type: str = "float16",
        config_path: str = "../../config/models/bloom_175b.yaml",
        allow_wrapping: bool = False,
        optimization_level: int = 1,
        reuse_pe_for_aggregation: bool = True,
        row_aggregation_enabled: bool = True,
        column_aggregation_enabled: bool = False,
        channel_bandwidth: float = 32.0
    ):
        """
        Initialize a BLOOM model.
        
        Args:
            noc_rows: Number of rows in the NoC grid
            noc_cols: Number of columns in the NoC grid
            pe_memory_size: Memory size of each PE in bytes
            mapping_strategy: Strategy for mapping networks to PEs
            split_strategy: Strategy for splitting computation
            data_type: Data type for computations
            config_path: Path to the model configuration YAML
            allow_wrapping: Whether to allow wrapping around the NoC edges
            optimization_level: Level of optimization (0=none, 1=skip embeddings and normalizations)
            reuse_pe_for_aggregation: Whether to reuse PEs for aggregation
            row_aggregation_enabled: Whether row aggregation is enabled
            column_aggregation_enabled: Whether column aggregation is enabled
            channel_bandwidth: Bandwidth of communication channels
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Model parameters from config
        self.model_name = self.config.get('model_name', 'bloom-176b')
        self.attention_type = self.config.get('attention_type', 'self')
        self.vocab_size = self.config.get('vocab_size', 250880)
        self.embed_dim = self.config.get('embed_dim', 14336)
        self.max_seq_len = self.config.get('max_seq_len', 2048)
        self.batch_size = self.config.get('batch_size', 1)
        self.seq_len = self.config.get('seq_len', 2048)
        self.kv_seq_len = self.config.get('kv_seq_len', 2048)
        self.dtype = self.config.get('dtype', 'float16')
        self.num_heads = self.config.get('num_heads', 112)
        self.num_layers = self.config.get('num_layers', 70)
        self.mlp_expansion_factor = self.config.get('mlp_expansion_factor', 4)
        self.mlp_activation = self.config.get('mlp_activation', 'gelu')
        self.norm_type = self.config.get('norm_type', 'prelayernorm')
        self.dropout_rate = self.config.get('dropout_rate', 0.0)
        self.architecture = self.config.get('architecture', 'gpt2')
        self.positional_encoding = self.config.get('positional_encoding', 'rotary')
        self.optimization_level = optimization_level
        
        # Create base LLM
        self.llm = LLM(
            seq_len=self.seq_len,
            pe_memory_size=pe_memory_size,
            noc_rows=noc_rows,
            noc_cols=noc_cols,
            mapping_strategy=mapping_strategy,
            split_strategy=split_strategy,
            data_type=data_type,
            allow_wrapping=allow_wrapping,
            reuse_pe_for_aggregation=reuse_pe_for_aggregation,
            row_aggregation_enabled=row_aggregation_enabled,
            column_aggregation_enabled=column_aggregation_enabled,
            channel_bandwidth=channel_bandwidth
        )
        
        # Networks dictionary for easier tracking
        self.networks = {}
        
        # Build the model's network structure
        self._build_model()
        
        # Ensure all networks have active_pes attribute
        self._ensure_active_pes_in_networks()
    
    def _ensure_active_pes_in_networks(self):
        """Ensure all networks have active_pes attribute."""
        for name, network in self.networks.items():
            if not hasattr(network, 'active_pes'):
                print(f"Adding active_pes to network {name}")
                network.active_pes = set()
                # Try to populate from mapper if available
                if hasattr(network, 'mapper') and hasattr(network.mapper, 'pe_layer_map'):
                    for pe_coords in network.mapper.pe_layer_map.keys():
                        network.active_pes.add(pe_coords)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            # Return default BLOOM configurations
            return {
                'model_name': 'bloom-176b',
                'attention_type': 'self',
                'vocab_size': 250880,
                'embed_dim': 14336,
                'max_seq_len': 2048,
                'batch_size': 1,
                'seq_len': 2048,
                'kv_seq_len': 2048,
                'dtype': 'float16',
                'num_heads': 112,
                'num_layers': 70,
                'mlp_expansion_factor': 4,
                'mlp_activation': 'gelu',
                'norm_type': 'prelayernorm',
                'dropout_rate': 0.0,
                'architecture': 'gpt2',
                'positional_encoding': 'rotary'
            }
    
    def _build_model(self):
        """Build the complete BLOOM model structure."""
        # Calculate dimensions
        hidden_dim = self.embed_dim * self.mlp_expansion_factor
        head_dim = self.embed_dim // self.num_heads
        
        # Create the embedding layer (skip if optimization_level > 0)
        if self.optimization_level == 0:
            self.networks['token_embedding'] = self.llm.create_fc_network(
                name="token_embedding",
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                seq_len=self.seq_len
            )
        
        # Create layers
        execution_order = []
        
        for layer_idx in range(self.num_layers):
            layer_prefix = f"layer_{layer_idx}_"
            
            # BLOOM uses pre-layer normalization (different from GPT2)
            if self.optimization_level == 0:
                # 1. First layer normalization (pre-attention)
                self.networks[f'{layer_prefix}ln1'] = self.llm.create_arithmetic_network(
                    name=f"{layer_prefix}ln1",
                    seq_len=self.seq_len,
                    d_model=self.embed_dim,
                    operation="norm"
                )
                
                # 2. Second layer normalization (pre-MLP)
                self.networks[f'{layer_prefix}ln2'] = self.llm.create_arithmetic_network(
                    name=f"{layer_prefix}ln2",
                    seq_len=self.seq_len,
                    d_model=self.embed_dim,
                    operation="norm"
                )
            
            # 3. QKV projections
            self.networks[f'{layer_prefix}q_proj'] = self.llm.create_fc_network(
                name=f"{layer_prefix}q_proj",
                input_dim=self.embed_dim,
                output_dim=self.embed_dim,
                seq_len=self.seq_len
            )
            
            self.networks[f'{layer_prefix}k_proj'] = self.llm.create_fc_network(
                name=f"{layer_prefix}k_proj",
                input_dim=self.embed_dim,
                output_dim=self.embed_dim,
                seq_len=self.seq_len
            )
            
            self.networks[f'{layer_prefix}v_proj'] = self.llm.create_fc_network(
                name=f"{layer_prefix}v_proj",
                input_dim=self.embed_dim,
                output_dim=self.embed_dim,
                seq_len=self.seq_len
            )
            
            # 4. Multi-head attention with rotary positional encodings
            attention_heads = []
            for head_idx in range(self.num_heads):
                head_name = f"{layer_prefix}attention_head_{head_idx}"
                self.networks[head_name] = self.llm.create_arithmetic_network(
                    name=head_name,
                    seq_len=self.seq_len,
                    d_model=head_dim,
                    operation="attention"
                )
                attention_heads.append(head_name)
            
            # 5. Output projection
            self.networks[f'{layer_prefix}output_proj'] = self.llm.create_fc_network(
                name=f"{layer_prefix}output_proj",
                input_dim=self.embed_dim,
                output_dim=self.embed_dim,
                seq_len=self.seq_len
            )
            
            # 6. MLP layers
            self.networks[f'{layer_prefix}mlp1'] = self.llm.create_fc_network(
                name=f"{layer_prefix}mlp1",
                input_dim=self.embed_dim,
                output_dim=hidden_dim,
                seq_len=self.seq_len
            )
            
            self.networks[f'{layer_prefix}mlp2'] = self.llm.create_fc_network(
                name=f"{layer_prefix}mlp2",
                input_dim=hidden_dim,
                output_dim=self.embed_dim,
                seq_len=self.seq_len
            )
            
            # Connect networks within this layer (respecting pre-layer norm architecture)
            if layer_idx == 0:
                # First layer connections
                if self.optimization_level == 0:
                    # Connect token embedding to first layer norm
                    self.llm.connect_networks("token_embedding", f"{layer_prefix}ln1")
                    # Connect LN1 to QKV projections
                    self.llm.connect_networks(f"{layer_prefix}ln1", f"{layer_prefix}q_proj")
                    self.llm.connect_networks(f"{layer_prefix}ln1", f"{layer_prefix}k_proj")
                    self.llm.connect_networks(f"{layer_prefix}ln1", f"{layer_prefix}v_proj")
                # If using optimization, projections are fed directly during inference
            else:
                # For subsequent layers, connect from the previous layer's output
                prev_layer_prefix = f"layer_{layer_idx-1}_"
                if self.optimization_level == 0:
                    # With full model, connect from previous mlp2 to current ln1, then to projections
                    self.llm.connect_networks(f"{prev_layer_prefix}mlp2", f"{layer_prefix}ln1")
                    self.llm.connect_networks(f"{layer_prefix}ln1", f"{layer_prefix}q_proj")
                    self.llm.connect_networks(f"{layer_prefix}ln1", f"{layer_prefix}k_proj")
                    self.llm.connect_networks(f"{layer_prefix}ln1", f"{layer_prefix}v_proj")
                else:
                    # With optimization, connect directly from previous mlp2 to projections
                    self.llm.connect_networks(f"{prev_layer_prefix}mlp2", f"{layer_prefix}q_proj")
                    self.llm.connect_networks(f"{prev_layer_prefix}mlp2", f"{layer_prefix}k_proj")
                    self.llm.connect_networks(f"{prev_layer_prefix}mlp2", f"{layer_prefix}v_proj")
            
            # Connect projections to attention heads
            for head_idx in range(self.num_heads):
                head_name = f"{layer_prefix}attention_head_{head_idx}"
                slice_start = head_idx * head_dim
                slice_end = (head_idx + 1) * head_dim
                
                # Connect Q, K, V projections to this attention head
                self.llm.connect_networks(
                    f"{layer_prefix}q_proj", head_name, 
                    connection_type="attention_q",
                    source_range=(slice_start, slice_end)
                )
                
                self.llm.connect_networks(
                    f"{layer_prefix}k_proj", head_name, 
                    connection_type="attention_k",
                    source_range=(slice_start, slice_end)
                )
                
                self.llm.connect_networks(
                    f"{layer_prefix}v_proj", head_name, 
                    connection_type="attention_v",
                    source_range=(slice_start, slice_end)
                )
                
                # Connect attention head to output projection
                self.llm.connect_networks(
                    head_name, f"{layer_prefix}output_proj",
                    dest_range=(slice_start, slice_end)
                )
            
            # For BLOOM's pre-norm architecture
            if self.optimization_level == 0:
                # Connect input to output projection (residual)
                if layer_idx == 0:
                    self.llm.connect_networks("token_embedding", f"{layer_prefix}output_proj", connection_type="residual")
                else:
                    prev_layer_prefix = f"layer_{layer_idx-1}_"
                    self.llm.connect_networks(f"{prev_layer_prefix}mlp2", f"{layer_prefix}output_proj", connection_type="residual")
                
                # Connect output projection to second layer norm
                self.llm.connect_networks(f"{layer_prefix}output_proj", f"{layer_prefix}ln2")
                self.llm.connect_networks(f"{layer_prefix}ln2", f"{layer_prefix}mlp1")
                
                # Connect output projection to mlp2 (residual)
                self.llm.connect_networks(f"{layer_prefix}output_proj", f"{layer_prefix}mlp2", connection_type="residual")
            else:
                # With optimization, connect directly to MLP
                self.llm.connect_networks(f"{layer_prefix}output_proj", f"{layer_prefix}mlp1")
            
            # Connect MLP layers
            self.llm.connect_networks(f"{layer_prefix}mlp1", f"{layer_prefix}mlp2")
            
            # Add to execution order
            if layer_idx == 0 and self.optimization_level == 0:
                execution_order.append(["token_embedding"])
            
            # Layer execution order
            layer_execution = []
            
            if self.optimization_level == 0:
                # Full model execution order (pre-layer norm architecture)
                layer_execution = [
                    [f"{layer_prefix}ln1"],
                    [f"{layer_prefix}q_proj", f"{layer_prefix}k_proj", f"{layer_prefix}v_proj"],  # Run projections in parallel
                    attention_heads,  # Run attention heads in parallel
                    [f"{layer_prefix}output_proj"],
                    [f"{layer_prefix}ln2"],
                    [f"{layer_prefix}mlp1"],
                    [f"{layer_prefix}mlp2"]
                ]
            else:
                # Optimized execution order (no layer norms)
                layer_execution = [
                    [f"{layer_prefix}q_proj", f"{layer_prefix}k_proj", f"{layer_prefix}v_proj"],  # Run projections in parallel
                    attention_heads,  # Run attention heads in parallel
                    [f"{layer_prefix}output_proj"],
                    [f"{layer_prefix}mlp1"],
                    [f"{layer_prefix}mlp2"]
                ]
            
            execution_order.extend(layer_execution)
        
        # Create the final layer norm (skip if optimization_level > 0)
        if self.optimization_level == 0:
            self.networks['final_ln'] = self.llm.create_arithmetic_network(
                name="final_ln",
                seq_len=self.seq_len,
                d_model=self.embed_dim,
                operation="norm"
            )
            
            # Connect the last layer to the final layer norm
            last_layer_prefix = f"layer_{self.num_layers-1}_"
            self.llm.connect_networks(f"{last_layer_prefix}mlp2", "final_ln")
            
            # Add final layer norm to execution order
            execution_order.append(["final_ln"])
        
        # Set the execution order
        self.llm.set_execution_order(execution_order)
    
    def run_inference(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run inference on the model.
        
        Args:
            input_tensor: Input tensor for inference
            
        Returns:
            Dictionary of output tensors
        """
        if self.optimization_level == 0:
            # For regular model, input goes to token embedding
            inputs = {"token_embedding": input_tensor}
        else:
            # For optimized model, input goes directly to first layer's projections
            # Assumes input is already in the embedding space (dimension [seq_len, embed_dim])
            inputs = {
                "layer_0_q_proj": input_tensor,
                "layer_0_k_proj": input_tensor,
                "layer_0_v_proj": input_tensor
            }
        
        outputs = self.llm.run_inference(inputs)
        return outputs

    def get_traffic_table(self) -> pd.DataFrame:
        """Get traffic table from the model's NoC."""
        return self.llm.noc.scheduler.get_traffic_table()
    
    def get_pe_utilization(self) -> Dict[str, Any]:
        """Get PE utilization statistics."""
        return self.llm.get_pe_utilization()
    
    def get_pe_mapping_details(self) -> pd.DataFrame:
        """Get detailed PE mapping information."""
        if hasattr(self.llm, 'get_pe_mapping_details'):
            return self.llm.get_pe_mapping_details()
        return pd.DataFrame() 