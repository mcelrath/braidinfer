// Quantization constants — shared between standalone and megakernel kernels.
// Included by linear_proj.hip and megakernel.hip.

#ifndef QUANT_CONSTS_H
#define QUANT_CONSTS_H

// NF4 codebook: 16 quantile-matched levels for N(0,1), from QLoRA
__constant__ float NF4_TABLE[16] = {
    -1.0f, -0.6961928f, -0.5250731f, -0.3949175f,
    -0.2844414f, -0.1847734f, -0.0910500f,  0.0f,
     0.0795803f,  0.1609302f,  0.2461123f,  0.3379152f,
     0.4407098f,  0.5626170f,  0.7229568f,  1.0f,
};

#endif // QUANT_CONSTS_H
