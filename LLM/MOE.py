import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.expert = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.expert(x)
        return x

class SparseMOE(nn.Module):
    def __init__(self, input_dim, output_dim, top_k, num_experts):
        super().__init__()

        self.expert = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.k = top_k
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        expert_outputs = [layer(x) for layer in self.expert]
        expert_outputs = torch.stack(expert_outputs, dim=1) #(batch_size, num_experts, output_dim)

        gate_logits = self.gate(x)
        values, indices = torch.topk(gate_logits, k=self.k, dim=-1)
        gates = torch.full_like(gate_logits, float('-inf')).scatter(1, indices, values)
        gates = torch.softmax(gates, dim=-1) #(batch_size, num_experts)

        x = gates.unsqueeze(1) @ expert_outputs
        x = x.squeeze(1)
        return x

if __name__ == '__main__':
    input_dim = 64
    output_dim = 128
    num_experts = 8
    k = 2
    batch_size = 4

    moe = SparseMOE(input_dim, output_dim, k, num_experts)
    x = torch.randn(batch_size, input_dim)
    output = moe(x)
