import torch

def sparsify_matrix(matrix : torch.tensor, density : float, block_size : list[int, int], keep_diagonal : bool=False, return_mask : bool=False):
    """ sparsifies matrix with specified block size

        Parameters:
        -----------
        matrix : torch.tensor
            matrix to sparsify
        density : int
            target density
        block_size : [int, int]
            block size dimensions
        keep_diagonal : bool
            If true, the diagonal will be kept. This option requires block_size[0] == block_size[1] and defaults to False
    """

    m, n   = matrix.shape
    m1, n1 = block_size

    if m % m1 or n % n1:
        raise ValueError(f"block size {(m1, n1)} does not divide matrix size {(m, n)}")

    # extract diagonal if keep_diagonal = True
    if keep_diagonal:
        if m != n:
            raise ValueError("Attempting to sparsify non-square matrix with keep_diagonal=True")

        to_spare = torch.diag(torch.diag(matrix))
        matrix   = matrix - to_spare
    else:
        to_spare = torch.zeros_like(matrix)

    # calculate energy in sub-blocks
    x = torch.reshape(matrix, (m // m1, m1, n // n1, n1))
    x = x ** 2
    block_energies = torch.sum(torch.sum(x, dim=3), dim=1)

    number_of_blocks = (m * n) // (m1 * n1)
    number_of_survivors = round(number_of_blocks * density)

    # masking threshold
    if number_of_survivors == 0:
        threshold = 0
    else:
        threshold = torch.sort(torch.flatten(block_energies)).values[-number_of_survivors]

    # create mask
    mask = torch.ones_like(block_energies)
    mask[block_energies < threshold] = 0
    mask = torch.repeat_interleave(mask, m1, dim=0)
    mask = torch.repeat_interleave(mask, n1, dim=1)

    # perform masking
    masked_matrix = mask * matrix + to_spare

    if return_mask:
        return masked_matrix, mask
    else:
        return masked_matrix

def calculate_gru_flops_per_step(gru, sparsification_dict=dict(), drop_input=False):
    input_size = gru.input_size
    hidden_size = gru.hidden_size
    flops = 0

    input_density = (
        sparsification_dict.get('W_ir', [1])[0]
        + sparsification_dict.get('W_in', [1])[0]
        + sparsification_dict.get('W_iz', [1])[0]
    ) / 3

    recurrent_density = (
        sparsification_dict.get('W_hr', [1])[0]
        + sparsification_dict.get('W_hn', [1])[0]
        + sparsification_dict.get('W_hz', [1])[0]
    ) / 3

    # input matrix vector multiplications
    if not drop_input:
        flops += 2 * 3 * input_size * hidden_size * input_density

    # recurrent matrix vector multiplications
    flops += 2 * 3 * hidden_size * hidden_size * recurrent_density

    # biases
    flops += 6 * hidden_size

    # activations estimated by 10 flops per activation
    flops += 30 * hidden_size

    return flops