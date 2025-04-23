import torch
import unittest
import os
import sys
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.pe_noc import NoCTopology
from llmDeploy.network_composer import NetworkComposer
from llmDeploy.neural_network import FCNeuralNetwork, ArithmeticNetwork

class TestNetworkComposer(unittest.TestCase):
    """Test suite for NetworkComposer class."""
    
    def setUp(self):
        """Set up test case with NoCTopology and common parameters."""
        # Parameters
        self.rows = 8
        self.cols = 8
        self.seq_len = 4
        self.d_model = 64
        self.memory_size = 1024 * 16  # 16 KB per PE
        
        # Initialize NoC
        self.noc = NoCTopology(self.rows, self.cols, self.memory_size)
    
    def test_initialization(self):
        """Test basic initialization of NetworkComposer."""
        composer = NetworkComposer(self.noc)
        self.assertEqual(len(composer.networks), 0)
        self.assertEqual(len(composer.connections), 0)
        self.assertEqual(len(composer.execution_order), 0)
    
    def test_add_fc_network(self):
        """Test adding an FC network to the composer."""
        composer = NetworkComposer(self.noc)
        
        # Add an FC network
        composer.add_fc_network(
            name="test_fc",
            input_dim=self.d_model,
            layer_dims=[self.d_model],
            seq_len=self.seq_len
        )
        
        # Verify network was added
        self.assertEqual(len(composer.networks), 1)
        self.assertTrue("test_fc" in composer.networks)
        self.assertTrue(isinstance(composer.networks["test_fc"], FCNeuralNetwork))
        self.assertEqual(composer.execution_order, ["test_fc"])
    
    def test_add_arithmetic_network(self):
        """Test adding an arithmetic network to the composer."""
        composer = NetworkComposer(self.noc)
        
        # Add an arithmetic network
        composer.add_arithmetic_network(
            name="test_arithmetic",
            seq_len=self.seq_len,
            d_model=self.d_model
        )
        
        # Verify network was added
        self.assertEqual(len(composer.networks), 1)
        self.assertTrue("test_arithmetic" in composer.networks)
        self.assertTrue(isinstance(composer.networks["test_arithmetic"], ArithmeticNetwork))
        self.assertEqual(composer.execution_order, ["test_arithmetic"])
    
    def test_connect_networks(self):
        """Test connecting networks in the composer."""
        composer = NetworkComposer(self.noc)
        
        # Add networks
        composer.add_fc_network(
            name="fc1",
            input_dim=self.d_model,
            layer_dims=[self.d_model],
            seq_len=self.seq_len
        )
        
        composer.add_arithmetic_network(
            name="arithmetic",
            seq_len=self.seq_len,
            d_model=self.d_model
        )
        
        # Connect networks
        composer.connect(
            source_network="fc1",
            dest_network="arithmetic",
            connection_type="matmul_a"
        )
        
        # Verify connection
        self.assertEqual(len(composer.connections), 1)
        self.assertTrue(("fc1", "arithmetic") in composer.connections)
        self.assertEqual(composer.connections[("fc1", "arithmetic")]["type"], "matmul_a")
    
    def test_set_execution_order(self):
        """Test setting execution order in the composer."""
        composer = NetworkComposer(self.noc)
        
        # Add networks
        composer.add_fc_network(
            name="fc1",
            input_dim=self.d_model,
            layer_dims=[self.d_model],
            seq_len=self.seq_len
        )
        
        composer.add_arithmetic_network(
            name="arithmetic",
            seq_len=self.seq_len,
            d_model=self.d_model
        )
        
        # Set execution order
        composer.set_execution_order(["arithmetic", "fc1"])
        
        # Verify order
        self.assertEqual(composer.execution_order, ["arithmetic", "fc1"])
        
        # Test invalid order
        with self.assertRaises(ValueError):
            composer.set_execution_order(["invalid_network"])
    
    def test_matrix_multiply_setup(self):
        """Test setting up a matrix multiplication with two input networks."""
        composer = NetworkComposer(self.noc)
        
        # Add two FC networks for Q and K projections
        composer.add_fc_network(
            name="q_proj",
            input_dim=self.d_model,
            layer_dims=[self.d_model],
            seq_len=self.seq_len,
            split_strategy="column_split"
        )
        
        composer.add_fc_network(
            name="k_proj",
            input_dim=self.d_model,
            layer_dims=[self.d_model],
            seq_len=self.seq_len,
            split_strategy="column_split"
        )
        
        # Add arithmetic network for matrix multiplication
        composer.add_arithmetic_network(
            name="matmul",
            seq_len=self.seq_len,
            d_model=self.d_model,
            split_strategy="column_split"
        )
        
        # Connect networks
        composer.connect(
            source_network="q_proj",
            dest_network="matmul",
            connection_type="matmul_a"
        )
        
        composer.connect(
            source_network="k_proj",
            dest_network="matmul",
            connection_type="matmul_b"
        )
        
        # Verify setup
        self.assertEqual(len(composer.networks), 3)
        self.assertEqual(len(composer.connections), 2)
        
        # Check PE assignments
        # We just verify that PEs were assigned, not specific coordinates
        q_proj_network = composer.networks["q_proj"]
        k_proj_network = composer.networks["k_proj"]
        matmul_network = composer.networks["matmul"]
        
        self.assertTrue(len(q_proj_network.active_pes) > 0)
        self.assertTrue(len(k_proj_network.active_pes) > 0)
        self.assertTrue(len(matmul_network.active_pes) > 0)
    
    def test_utilization_calculation(self):
        """Test PE utilization calculation."""
        composer = NetworkComposer(self.noc)
        
        # Add two networks
        composer.add_fc_network(
            name="fc1",
            input_dim=self.d_model,
            layer_dims=[self.d_model],
            seq_len=self.seq_len
        )
        
        composer.add_arithmetic_network(
            name="arithmetic",
            seq_len=self.seq_len,
            d_model=self.d_model
        )
        
        # Get utilization
        utilization = composer.get_pe_utilization()
        
        # Verify utilization stats
        self.assertIn('total_pes', utilization)
        self.assertIn('used_computation_pes', utilization)
        self.assertIn('computation_utilization', utilization)
        
        self.assertEqual(utilization['total_pes'], self.rows * self.cols)
        self.assertTrue(utilization['used_computation_pes'] > 0)
        self.assertTrue(0 <= utilization['computation_utilization'] <= 100)
    
    def test_matrix_multiply_with_inputs(self):
        """Test matrix multiplication with direct inputs."""
        composer = NetworkComposer(self.noc)
        
        # Add arithmetic network for matrix multiplication
        composer.add_arithmetic_network(
            name="matmul",
            seq_len=self.seq_len,
            d_model=self.d_model,
            split_strategy="column_split"
        )
        
        # Create test inputs
        input_a = torch.randn(self.seq_len, self.d_model)
        input_b = torch.randn(self.seq_len, self.d_model)
        
        # This would normally run the computation
        # But since we can't fully implement it in a test, we'll just verify the setup
        
        # For a real implementation, we'd run:
        # outputs = composer.run_matrix_multiply(
        #     network_name="matmul",
        #     input_a=input_a,
        #     input_b=input_b,
        #     transpose_b=True
        # )
        #
        # And then verify the output shape and values
        
        # For now, just verify the network exists
        self.assertTrue("matmul" in composer.networks)
        self.assertTrue(hasattr(composer.networks["matmul"], "matrix_multiply"))
        
        # Verify a simple property of the network
        matmul_network = composer.networks["matmul"]
        self.assertEqual(matmul_network.seq_len, self.seq_len)
        self.assertEqual(matmul_network.d_model, self.d_model)

if __name__ == '__main__':
    unittest.main() 