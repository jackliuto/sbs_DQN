import torch

# Example tensor
tensor_3d = torch.tensor([[[1, 2, 3], [4, 5, 6]],
                          [[7, 8, 9], [10, 11, 12]],
                          [[13, 14, 15], [16, 17, 18]]])

# Tensor of indices
list_of_indices = torch.tensor([[0, 1, 1],
                                [1, 0, 2]])

print(list_of_indices.dtype)

# Transpose and convert indices to a list of tensors
indices = list(list_of_indices.t())


# print(indices)

# Use the tensors to index into tensor_3d
# extracted_values = tensor_3d[indices[0], indices[1], indices[2]]

extracted_values = tensor_3d[indices]

print(extracted_values)