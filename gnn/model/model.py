
import torch
import torch.nn as nn

class OrbFrozenMLP(nn.Module):
    def __init__(
        self,
        orb_model,
        hidden_dim=128,
        output_dim=1,
        freeze_orb=True,
        task_type="regression",
    ):
        """
        Args:
            orb_model (nn.Module): The pretrained ORB model (e.g., orbff.model).
            hidden_dim (int): Dimension of the hidden layer in the MLP.
            output_dim (int): Number of output features (e.g., 1 for regression, 
                              number of classes for classification).
            freeze_orb (bool): If True, the ORB model parameters are frozen.
            task_type (str): "binary", "multiclass", "regression"
        """
        super().__init__()
        
        # Optionally freeze the ORB model parameters
        if freeze_orb:
            for param in orb_model.parameters():
                param.requires_grad = False
        
        self.orb_model = orb_model
        self.task_type = task_type
        
        # Simple MLP readout (customize as needed)
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Apply Xavier uniform initialization to all Linear layers
        self._initialize_weights()

    def _initialize_weights(self):
        """ Apply Xavier Uniform initialization to Linear layers """
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, graph):
        """
        Forward pass:
          1. Featurize edges/nodes with orb_model.
          2. Pass through GNN stacks.
          3. Take the mean of node features.
          4. Pass the pooled feature through MLP.
          5. Return either a log-softmax (classification) or raw output (regression).
        """
        # 1) Featurize with ORB
        graph = self.orb_model.featurize_edges(graph)
        graph = self.orb_model.featurize_nodes(graph)
        graph = self.orb_model._encoder(graph)
        
        # 2) Pass through GNN stacks
        for gnn in self.orb_model.gnn_stacks:
            graph = gnn(graph)
        
        # 3) Node feature pooling
        node_feat = graph.node_features['feat']  # shape: (num_nodes, feature_dim)
        splitted_feats = torch.split(node_feat, graph.n_node.tolist(), dim=0)
        pooled_feats = [sub_feat.mean(dim=0) for sub_feat in splitted_feats]
        pooled_feats = torch.stack(pooled_feats, dim=0)

        # 4) Readout MLP
        out = self.mlp(pooled_feats)  # shape: (batch_size, output_dim)
        
        # 5) Classification or regression
        if self.task_type == "binary":
            # We might want raw logits to feed into BCEWithLogitsLoss, 
            # so no activation here. If you do want a sigmoid for inference:
            # out = torch.sigmoid(out)
            pass
        elif self.task_type == "multiclass":
            # If you plan to use CrossEntropyLoss, you do not apply softmax here;
            # CrossEntropyLoss expects raw logits. 
            # If you want log-softmax for inference:
            out = torch.nn.functional.log_softmax(out, dim=1)
            pass
        elif self.task_type == "regression":
            # Keep it raw for MSELoss or similar
            pass

        return out

    def loss_fn(self, predictions, targets):
        """
        Compute the loss for the given task type.
        """
        if self.task_type == "binary":
            loss = nn.BCEWithLogitsLoss()(predictions.squeeze(dim=-1), targets)
        elif self.task_type == "multiclass":
            targets = targets.long()
            loss = nn.NLLLoss()(predictions, targets)
        elif self.task_type == "regression":
            loss = nn.MSELoss()(predictions.squeeze(dim=-1), targets)
        else:
            raise ValueError(f"Invalid task_type: {self.task_type}")
        
        return loss
    