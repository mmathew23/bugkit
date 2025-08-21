"""
Specialized analysis utilities for HuggingFace models
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap

from ...core.base import BaseDebugTool
from ...testing.tensor_compare import TensorComparer
from ..storage import DiskTensorStorage


class HuggingFaceAnalyzer(BaseDebugTool):
    """Specialized analyzer for HuggingFace model insights"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        save_visualizations: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.save_visualizations = save_visualizations
        self.analysis_results: Dict[str, Any] = {}
        
        # Create visualization directory
        if save_visualizations:
            self.viz_dir = self.output_dir / "visualizations"
            self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def enable(self) -> None:
        """Enable analyzer"""
        self.enabled = True
        if self.verbose:
            self.logger.info("HuggingFace analyzer enabled")
    
    def disable(self) -> None:
        """Disable analyzer and save results"""
        self.enabled = False
        self._save_analysis_results()
        if self.verbose:
            self.logger.info("HuggingFace analyzer disabled")
    
    def analyze_attention_patterns(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        layer_name: str = "attention",
        head_analysis: bool = True,
        pattern_analysis: bool = True,
        save_heatmap: bool = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive attention pattern analysis
        
        Args:
            attention_weights: Attention weights tensor [batch, heads, seq_len, seq_len]
            tokens: Token strings for visualization
            layer_name: Name of the attention layer
            head_analysis: Whether to analyze individual attention heads
            pattern_analysis: Whether to analyze attention patterns
            save_heatmap: Whether to save attention heatmaps
        """
        if not self.enabled:
            raise RuntimeError("HuggingFaceAnalyzer is not enabled")
        
        if save_heatmap is None:
            save_heatmap = self.save_visualizations
        
        # Ensure tensor is on CPU and detached
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu()
        
        analysis_result = {
            "layer_name": layer_name,
            "timestamp": time.time(),
            "tensor_info": {
                "shape": list(attention_weights.shape),
                "dtype": str(attention_weights.dtype),
                "device": str(attention_weights.device),
            },
            "global_statistics": {},
            "head_analysis": {},
            "pattern_analysis": {},
            "visualizations": [],
        }
        
        # Global attention statistics
        analysis_result["global_statistics"] = self._calculate_attention_statistics(attention_weights)
        
        # Per-head analysis
        if head_analysis and attention_weights.dim() >= 3:
            analysis_result["head_analysis"] = self._analyze_attention_heads(attention_weights)
        
        # Pattern analysis
        if pattern_analysis:
            analysis_result["pattern_analysis"] = self._analyze_attention_patterns(attention_weights, tokens)
        
        # Generate visualizations
        if save_heatmap:
            viz_paths = self._create_attention_visualizations(
                attention_weights, tokens, layer_name
            )
            analysis_result["visualizations"] = viz_paths
        
        # Store results
        self.analysis_results[f"attention_{layer_name}_{int(time.time())}"] = analysis_result
        
        if self.verbose:
            self.logger.info(f"Attention analysis completed for {layer_name}")
        
        return analysis_result
    
    def _calculate_attention_statistics(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Calculate global attention statistics"""
        # Flatten for global statistics
        flat_attention = attention_weights.flatten()
        
        # Basic statistics
        stats = {
            "mean": float(torch.mean(flat_attention)),
            "std": float(torch.std(flat_attention)),
            "min": float(torch.min(flat_attention)),
            "max": float(torch.max(flat_attention)),
            "median": float(torch.median(flat_attention)),
        }
        
        # Attention-specific metrics
        if attention_weights.dim() >= 3:
            # Entropy calculation (over last dimension)
            eps = 1e-8
            attention_probs = attention_weights + eps
            entropy = -(attention_probs * torch.log(attention_probs)).sum(-1)
            
            stats.update({
                "mean_entropy": float(torch.mean(entropy)),
                "std_entropy": float(torch.std(entropy)),
                "sparsity": float(torch.mean((attention_weights < 0.01).float())),
                "diagonal_dominance": self._calculate_diagonal_dominance(attention_weights),
            })
        
        return stats
    
    def _calculate_diagonal_dominance(self, attention_weights: torch.Tensor) -> float:
        """Calculate how much attention focuses on the diagonal (self-attention)"""
        if attention_weights.dim() < 3:
            return 0.0
        
        # Get diagonal elements for each head
        min_dim = min(attention_weights.shape[-2], attention_weights.shape[-1])
        diagonal_elements = []
        
        for i in range(min_dim):
            diagonal_elements.append(attention_weights[..., i, i])
        
        if diagonal_elements:
            diagonal_tensor = torch.stack(diagonal_elements, dim=-1)
            return float(torch.mean(diagonal_tensor))
        
        return 0.0
    
    def _analyze_attention_heads(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze individual attention heads"""
        if attention_weights.dim() < 4:
            return {"error": "Insufficient dimensions for head analysis"}
        
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        head_analysis = {
            "num_heads": num_heads,
            "head_statistics": {},
            "head_similarities": {},
            "dominant_heads": [],
        }
        
        # Analyze each head
        for head_idx in range(num_heads):
            head_attention = attention_weights[:, head_idx, :, :]
            
            # Calculate head-specific statistics
            head_stats = {
                "mean_attention": float(torch.mean(head_attention)),
                "max_attention": float(torch.max(head_attention)),
                "entropy": float(torch.mean(-(head_attention * torch.log(head_attention + 1e-8)).sum(-1))),
                "sparsity": float(torch.mean((head_attention < 0.01).float())),
                "attention_variance": float(torch.var(head_attention)),
            }
            
            # Classify attention pattern
            head_stats["attention_pattern"] = self._classify_attention_pattern(head_attention)
            
            head_analysis["head_statistics"][f"head_{head_idx}"] = head_stats
        
        # Find dominant heads (highest variance = most selective)
        head_variances = [
            (head_idx, stats["attention_variance"])
            for head_idx, stats in enumerate(head_analysis["head_statistics"].values())
        ]
        head_variances.sort(key=lambda x: x[1], reverse=True)
        
        head_analysis["dominant_heads"] = [
            {"head_index": head_idx, "variance": var}
            for head_idx, var in head_variances[:3]  # Top 3 most selective heads
        ]
        
        # Calculate head similarities
        head_analysis["head_similarities"] = self._calculate_head_similarities(attention_weights)
        
        return head_analysis
    
    def _classify_attention_pattern(self, head_attention: torch.Tensor) -> str:
        """Classify the attention pattern of a head"""
        # Calculate various pattern metrics
        seq_len = head_attention.shape[-1]
        
        # Diagonal dominance
        diagonal_score = self._calculate_diagonal_dominance(head_attention.unsqueeze(1)).item()
        
        # Local vs global attention
        # Create distance matrix
        positions = torch.arange(seq_len).float()
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        
        # Weight by distance
        local_attention = torch.sum(head_attention * (distance_matrix <= 3))  # Within 3 positions
        global_attention = torch.sum(head_attention * (distance_matrix > 3))
        
        local_ratio = local_attention / (local_attention + global_attention + 1e-8)
        
        # Classify pattern
        if diagonal_score > 0.3:
            return "self_focused"
        elif local_ratio > 0.7:
            return "local_attention"
        elif local_ratio < 0.3:
            return "global_attention"
        else:
            return "mixed_attention"
    
    def _calculate_head_similarities(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Calculate similarities between attention heads"""
        if attention_weights.dim() < 4:
            return {"error": "Insufficient dimensions for head similarity"}
        
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Average across batch dimension
        avg_attention = torch.mean(attention_weights, dim=0)  # [num_heads, seq_len, seq_len]
        
        # Calculate pairwise similarities
        similarities = {}
        similarity_matrix = torch.zeros(num_heads, num_heads)
        
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                # Calculate cosine similarity
                head_i = avg_attention[i].flatten()
                head_j = avg_attention[j].flatten()
                
                similarity = torch.cosine_similarity(head_i, head_j, dim=0)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
                
                similarities[f"head_{i}_vs_head_{j}"] = float(similarity)
        
        # Find most similar heads
        upper_triangle = torch.triu(similarity_matrix, diagonal=1)
        max_similarity = torch.max(upper_triangle)
        max_indices = torch.where(upper_triangle == max_similarity)
        
        most_similar = []
        for i in range(len(max_indices[0])):
            head_i, head_j = int(max_indices[0][i]), int(max_indices[1][i])
            most_similar.append({
                "head_pair": [head_i, head_j],
                "similarity": float(max_similarity),
            })
        
        return {
            "pairwise_similarities": similarities,
            "similarity_matrix": similarity_matrix.tolist(),
            "most_similar_heads": most_similar,
            "average_similarity": float(torch.mean(upper_triangle[upper_triangle > 0])),
        }
    
    def _analyze_attention_patterns(
        self, 
        attention_weights: torch.Tensor, 
        tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze high-level attention patterns"""
        pattern_analysis = {
            "sequence_patterns": {},
            "token_patterns": {},
            "structural_patterns": {},
        }
        
        # Average across batch and heads for pattern analysis
        if attention_weights.dim() == 4:
            avg_attention = torch.mean(attention_weights, dim=(0, 1))  # [seq_len, seq_len]
        else:
            avg_attention = torch.mean(attention_weights, dim=0)  # Assume already averaged
        
        seq_len = avg_attention.shape[0]
        
        # Sequence-level patterns
        pattern_analysis["sequence_patterns"] = {
            "start_token_attention": float(torch.mean(avg_attention[:, 0])),  # Attention to first token
            "end_token_attention": float(torch.mean(avg_attention[:, -1])),   # Attention to last token
            "positional_bias": self._calculate_positional_bias(avg_attention),
            "attention_span": self._calculate_attention_span(avg_attention),
        }
        
        # Token-specific patterns (if tokens provided)
        if tokens:
            pattern_analysis["token_patterns"] = self._analyze_token_patterns(avg_attention, tokens)
        
        # Structural patterns
        pattern_analysis["structural_patterns"] = {
            "band_structure": self._detect_band_structure(avg_attention),
            "block_structure": self._detect_block_structure(avg_attention),
            "sparsity_pattern": self._analyze_sparsity_pattern(avg_attention),
        }
        
        return pattern_analysis
    
    def _calculate_positional_bias(self, attention_matrix: torch.Tensor) -> Dict[str, float]:
        """Calculate positional bias in attention"""
        seq_len = attention_matrix.shape[0]
        
        # Create position indices
        positions = torch.arange(seq_len).float()
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Calculate bias towards different relative positions
        forward_bias = torch.mean(attention_matrix * (pos_diff > 0).float())
        backward_bias = torch.mean(attention_matrix * (pos_diff < 0).float())
        current_bias = torch.mean(attention_matrix * (pos_diff == 0).float())
        
        return {
            "forward_bias": float(forward_bias),
            "backward_bias": float(backward_bias),
            "current_position_bias": float(current_bias),
            "directional_preference": "forward" if forward_bias > backward_bias else "backward",
        }
    
    def _calculate_attention_span(self, attention_matrix: torch.Tensor) -> Dict[str, float]:
        """Calculate effective attention span"""
        seq_len = attention_matrix.shape[0]
        
        # For each query position, calculate the attention span
        spans = []
        for i in range(seq_len):
            attention_dist = attention_matrix[i, :]
            
            # Calculate weighted average distance
            positions = torch.arange(seq_len).float()
            distances = torch.abs(positions - i)
            
            weighted_span = torch.sum(attention_dist * distances)
            spans.append(float(weighted_span))
        
        return {
            "mean_attention_span": float(np.mean(spans)),
            "std_attention_span": float(np.std(spans)),
            "min_attention_span": float(np.min(spans)),
            "max_attention_span": float(np.max(spans)),
        }
    
    def _analyze_token_patterns(self, attention_matrix: torch.Tensor, tokens: List[str]) -> Dict[str, Any]:
        """Analyze attention patterns for specific tokens"""
        if len(tokens) != attention_matrix.shape[0]:
            return {"error": "Token count doesn't match sequence length"}
        
        token_patterns = {
            "high_attention_tokens": [],
            "low_attention_tokens": [],
            "token_type_analysis": {},
        }
        
        # Calculate attention received by each token
        attention_received = torch.sum(attention_matrix, dim=0)  # Sum over query positions
        
        # Find high and low attention tokens
        for i, token in enumerate(tokens):
            attention_score = float(attention_received[i])
            
            if attention_score > torch.mean(attention_received) + torch.std(attention_received):
                token_patterns["high_attention_tokens"].append({
                    "token": token,
                    "position": i,
                    "attention_score": attention_score,
                })
            elif attention_score < torch.mean(attention_received) - torch.std(attention_received):
                token_patterns["low_attention_tokens"].append({
                    "token": token,
                    "position": i,
                    "attention_score": attention_score,
                })
        
        # Analyze by token type (simple heuristics)
        special_tokens = ["[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"]
        punctuation = [".", ",", "!", "?", ";", ":"]
        
        special_attention = []
        punct_attention = []
        word_attention = []
        
        for i, token in enumerate(tokens):
            attention_score = float(attention_received[i])
            
            if token in special_tokens:
                special_attention.append(attention_score)
            elif any(p in token for p in punctuation):
                punct_attention.append(attention_score)
            else:
                word_attention.append(attention_score)
        
        token_patterns["token_type_analysis"] = {
            "special_tokens_avg_attention": float(np.mean(special_attention)) if special_attention else 0,
            "punctuation_avg_attention": float(np.mean(punct_attention)) if punct_attention else 0,
            "words_avg_attention": float(np.mean(word_attention)) if word_attention else 0,
        }
        
        return token_patterns
    
    def _detect_band_structure(self, attention_matrix: torch.Tensor) -> Dict[str, Any]:
        """Detect band/diagonal structures in attention"""
        seq_len = attention_matrix.shape[0]
        
        # Calculate attention along different diagonals
        diagonal_sums = {}
        
        for offset in range(-min(5, seq_len-1), min(6, seq_len)):
            diagonal = torch.diagonal(attention_matrix, offset=offset)
            diagonal_sums[f"diagonal_{offset}"] = float(torch.mean(diagonal))
        
        # Detect if there's a band structure
        main_diagonal = diagonal_sums.get("diagonal_0", 0)
        off_diagonal_avg = np.mean([v for k, v in diagonal_sums.items() if k != "diagonal_0"])
        
        return {
            "diagonal_strengths": diagonal_sums,
            "main_diagonal_strength": main_diagonal,
            "off_diagonal_avg": off_diagonal_avg,
            "has_band_structure": main_diagonal > 2 * off_diagonal_avg,
            "band_width": self._estimate_band_width(attention_matrix),
        }
    
    def _estimate_band_width(self, attention_matrix: torch.Tensor) -> int:
        """Estimate the effective bandwidth of attention"""
        seq_len = attention_matrix.shape[0]
        
        # For each row, find the effective span of non-negligible attention
        spans = []
        threshold = 0.01  # Attention values below this are considered negligible
        
        for i in range(seq_len):
            row = attention_matrix[i, :]
            significant_positions = torch.where(row > threshold)[0]
            
            if len(significant_positions) > 0:
                span = int(torch.max(significant_positions) - torch.min(significant_positions) + 1)
                spans.append(span)
        
        return int(np.mean(spans)) if spans else seq_len
    
    def _detect_block_structure(self, attention_matrix: torch.Tensor) -> Dict[str, Any]:
        """Detect block structures in attention matrix"""
        # Simplified block detection using variance
        seq_len = attention_matrix.shape[0]
        
        # Divide into blocks and calculate within-block vs between-block attention
        block_size = max(1, seq_len // 4)  # Divide into 4 blocks
        
        within_block_attention = 0
        between_block_attention = 0
        
        for i in range(0, seq_len, block_size):
            for j in range(0, seq_len, block_size):
                block_attention = torch.mean(
                    attention_matrix[i:i+block_size, j:j+block_size]
                )
                
                if i == j:  # Same block
                    within_block_attention += block_attention
                else:  # Different blocks
                    between_block_attention += block_attention
        
        num_blocks = (seq_len + block_size - 1) // block_size
        within_block_attention /= num_blocks
        between_block_attention /= (num_blocks * (num_blocks - 1))
        
        return {
            "block_size": block_size,
            "num_blocks": num_blocks,
            "within_block_attention": float(within_block_attention),
            "between_block_attention": float(between_block_attention),
            "block_structure_strength": float(within_block_attention / (between_block_attention + 1e-8)),
            "has_block_structure": within_block_attention > 1.5 * between_block_attention,
        }
    
    def _analyze_sparsity_pattern(self, attention_matrix: torch.Tensor) -> Dict[str, Any]:
        """Analyze sparsity patterns in attention"""
        # Different sparsity thresholds
        thresholds = [0.01, 0.05, 0.1]
        sparsity_analysis = {}
        
        for threshold in thresholds:
            sparse_mask = attention_matrix < threshold
            sparsity_ratio = float(torch.mean(sparse_mask.float()))
            
            sparsity_analysis[f"sparsity_at_{threshold}"] = {
                "ratio": sparsity_ratio,
                "num_sparse_elements": int(torch.sum(sparse_mask)),
                "total_elements": attention_matrix.numel(),
            }
        
        return sparsity_analysis
    
    def _create_attention_visualizations(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]],
        layer_name: str,
    ) -> List[str]:
        """Create attention heatmap visualizations"""
        visualization_paths = []
        
        # Average across batch dimension if needed
        if attention_weights.dim() == 4:
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Create overall attention heatmap (averaged across heads)
            avg_attention = torch.mean(attention_weights[0], dim=0)  # First batch, avg heads
            
            viz_path = self._create_single_heatmap(
                avg_attention, tokens, f"{layer_name}_average_attention"
            )
            visualization_paths.append(viz_path)
            
            # Create per-head heatmaps (limit to first few heads)
            max_heads_to_plot = min(6, num_heads)
            for head_idx in range(max_heads_to_plot):
                head_attention = attention_weights[0, head_idx]
                
                viz_path = self._create_single_heatmap(
                    head_attention, tokens, f"{layer_name}_head_{head_idx}"
                )
                visualization_paths.append(viz_path)
        
        else:
            # Single attention matrix
            viz_path = self._create_single_heatmap(
                attention_weights, tokens, f"{layer_name}_attention"
            )
            visualization_paths.append(viz_path)
        
        return visualization_paths
    
    def _create_single_heatmap(
        self,
        attention_matrix: torch.Tensor,
        tokens: Optional[List[str]],
        title: str,
    ) -> str:
        """Create a single attention heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Convert to numpy
        attention_np = attention_matrix.detach().cpu().numpy()
        
        # Create heatmap
        im = ax.imshow(attention_np, cmap='Blues', aspect='auto')
        
        # Set labels
        if tokens and len(tokens) == attention_matrix.shape[0]:
            # Limit token labels if too many
            if len(tokens) > 50:
                token_labels = [tokens[i] if i % (len(tokens) // 25) == 0 else '' for i in range(len(tokens))]
            else:
                token_labels = tokens
            
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=45, ha='right')
            ax.set_yticklabels(token_labels)
        
        ax.set_xlabel('Key Positions')
        ax.set_ylabel('Query Positions')
        ax.set_title(f'Attention Heatmap: {title}')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Save plot
        viz_path = self.viz_dir / f"{title.replace(' ', '_')}.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(viz_path)
    
    def compare_layer_outputs(
        self,
        layer_outputs: Dict[str, torch.Tensor],
        layer_names: Optional[List[str]] = None,
        comparison_name: str = "layer_comparison",
    ) -> Dict[str, Any]:
        """Compare outputs across different layers"""
        if not self.enabled:
            raise RuntimeError("HuggingFaceAnalyzer is not enabled")
        
        if layer_names is None:
            layer_names = list(layer_outputs.keys())
        
        comparison_result = {
            "comparison_name": comparison_name,
            "timestamp": time.time(),
            "layers_compared": layer_names,
            "pairwise_comparisons": {},
            "similarity_analysis": {},
            "activation_analysis": {},
            "recommendations": [],
        }
        
        # Pairwise tensor comparisons
        comparer = TensorComparer(verbose=False)
        comparer.enable()
        
        try:
            for i, layer1 in enumerate(layer_names):
                for j, layer2 in enumerate(layer_names[i+1:], i+1):
                    if layer1 in layer_outputs and layer2 in layer_outputs:
                        comparison = comparer.compare(
                            layer_outputs[layer1],
                            layer_outputs[layer2],
                            name1=layer1,
                            name2=layer2,
                        )
                        
                        comparison_key = f"{layer1}_vs_{layer2}"
                        comparison_result["pairwise_comparisons"][comparison_key] = comparison
        finally:
            comparer.disable()
        
        # Activation pattern analysis
        for layer_name in layer_names:
            if layer_name in layer_outputs:
                tensor = layer_outputs[layer_name]
                
                activation_stats = {
                    "mean_activation": float(torch.mean(tensor)),
                    "std_activation": float(torch.std(tensor)),
                    "sparsity": float(torch.mean((tensor == 0).float())),
                    "negative_ratio": float(torch.mean((tensor < 0).float())),
                    "activation_range": float(torch.max(tensor) - torch.min(tensor)),
                }
                
                comparison_result["activation_analysis"][layer_name] = activation_stats
        
        # Generate recommendations
        comparison_result["recommendations"] = self._generate_layer_comparison_recommendations(comparison_result)
        
        # Store results
        self.analysis_results[f"layer_comparison_{int(time.time())}"] = comparison_result
        
        return comparison_result
    
    def _generate_layer_comparison_recommendations(self, comparison_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on layer comparison"""
        recommendations = []
        
        # Check for highly similar layers
        similar_layers = []
        for comp_key, comp_data in comparison_result["pairwise_comparisons"].items():
            if comp_data.get("match_percentage", 0) > 95:
                similar_layers.append(comp_key)
        
        if similar_layers:
            recommendations.append(
                f"High similarity detected between layers: {', '.join(similar_layers[:3])}. "
                "Consider layer pruning or architectural changes."
            )
        
        # Check for dead neurons (high sparsity)
        high_sparsity_layers = []
        for layer_name, stats in comparison_result["activation_analysis"].items():
            if stats.get("sparsity", 0) > 0.8:
                high_sparsity_layers.append(layer_name)
        
        if high_sparsity_layers:
            recommendations.append(
                f"High sparsity in layers: {', '.join(high_sparsity_layers)}. "
                "Check for dead neurons or consider pruning."
            )
        
        return recommendations
    
    def profile_memory_usage(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """Profile memory usage of model layers"""
        if not self.enabled:
            raise RuntimeError("HuggingFaceAnalyzer is not enabled")
        
        memory_profile = {
            "timestamp": time.time(),
            "model_class": model.__class__.__name__,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "layer_memory_usage": {},
            "peak_memory_mb": 0,
            "memory_timeline": [],
        }
        
        # Hook for memory profiling
        layer_memory = {}
        
        def memory_hook(name):
            def hook_fn(module, input, output):
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                    layer_memory[name] = current_memory
                    memory_profile["memory_timeline"].append({
                        "layer": name,
                        "memory_mb": current_memory,
                        "timestamp": time.time(),
                    })
            return hook_fn
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(memory_hook(name))
                hooks.append(hook)
        
        try:
            # Run forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Calculate memory usage per layer
            memory_profile["layer_memory_usage"] = layer_memory
            memory_profile["peak_memory_mb"] = max(layer_memory.values()) if layer_memory else 0
            
            # Calculate memory growth
            if len(memory_profile["memory_timeline"]) > 1:
                memory_growth = []
                for i in range(1, len(memory_profile["memory_timeline"])):
                    growth = (memory_profile["memory_timeline"][i]["memory_mb"] - 
                             memory_profile["memory_timeline"][i-1]["memory_mb"])
                    memory_growth.append({
                        "layer": memory_profile["memory_timeline"][i]["layer"],
                        "growth_mb": growth,
                    })
                
                memory_profile["memory_growth"] = memory_growth
                
                # Find memory-hungry layers
                memory_profile["top_memory_layers"] = sorted(
                    memory_growth, key=lambda x: x["growth_mb"], reverse=True
                )[:5]
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Store results
        self.analysis_results[f"memory_profile_{int(time.time())}"] = memory_profile
        
        return memory_profile
    
    def _save_analysis_results(self) -> None:
        """Save all analysis results"""
        if self.analysis_results:
            self.save_json(self.analysis_results, "hf_analysis_results.json")
            
            # Generate summary report
            summary = {
                "total_analyses": len(self.analysis_results),
                "analysis_types": {},
                "timestamp": time.time(),
            }
            
            # Count analysis types
            for key in self.analysis_results.keys():
                analysis_type = key.split("_")[0]
                summary["analysis_types"][analysis_type] = summary["analysis_types"].get(analysis_type, 0) + 1
            
            self.save_json(summary, "hf_analysis_summary.json")
            
            if self.verbose:
                self.logger.info(f"Saved {len(self.analysis_results)} analysis results")


# Convenience functions
def analyze_attention_patterns(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    layer_name: str = "attention",
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Quick attention pattern analysis"""
    analyzer = HuggingFaceAnalyzer(output_dir=output_dir)
    analyzer.enable()
    
    try:
        return analyzer.analyze_attention_patterns(
            attention_weights, tokens, layer_name, **kwargs
        )
    finally:
        analyzer.disable()


def compare_layer_outputs(
    layer_outputs: Dict[str, torch.Tensor],
    layer_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Quick layer output comparison"""
    analyzer = HuggingFaceAnalyzer(output_dir=output_dir)
    analyzer.enable()
    
    try:
        return analyzer.compare_layer_outputs(layer_outputs, layer_names, **kwargs)
    finally:
        analyzer.disable()


def profile_memory_usage(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Quick memory usage profiling"""
    analyzer = HuggingFaceAnalyzer(output_dir=output_dir)
    analyzer.enable()
    
    try:
        return analyzer.profile_memory_usage(model, inputs, **kwargs)
    finally:
        analyzer.disable()