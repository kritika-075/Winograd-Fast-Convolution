# Winograd-Fast-Convolution
# Winograd
Winograd's fast convolution algorithms transform input and filters into another space where convolution becomes element-wise multiplication. The fourier transform also turns convolutions into element-wise multiplications, but uses complex numbers. Complex number multiplication requires 3 real multiplications, and hermitian symmetry in the discrete fourier transform (DFT) of real valued data effectively reduces this to about 1.5 multiplications per input. Winograd minimal algorithms only need 1 real multiplication per input, which is essentially the computational advantage that Winograd convolution has over DFT.

It is worth noting that while 3/2 = 1.5 real multiplications per complex multiplication is achieved in theory, 4/2 = 2 is used more often for DFT based CNN acceleration, because the former uses more workspace memory, which might outweigh the benefits of reduced mutliplications.

Wincnn generates a subset of the Winograd convolution algorithms that are called modified Cook-Toom algorithms. These use the provably minimal number of multiplications for convolution. Wincnn uses the Lagrange Interpolating Polynomial to transform polynomial multiplication, which is equivalent to convolution, into element-wise multiplication of the values that the polynomials take at a fixed number of interpolation points. The down-side of Cook-Toom algorithms is that the transforms quickly become unstable as the transform size increases. However they are a good match for the small 3x3 convolutions used in convolutional neural networks.

# Is the total number of arithmetic operations actually reduced?
Basically the number of multiply-accumulates in the multiplication stage dominates the number of arithmetic operations (additions, multiplications, or multiply-accumulates) in the transform stages, provided that the dimensions of the neural network layer are all large enough.

Example: A 3x3 convolutional layer has C input channels and K output channels, and spatial dimensions HxW. The direct algorithm use HWCK9 multiply accumulates. F(4x4, 3x3) uses (H/4)(W/4)CK(36) = HWCK(2.25) multiply accumulates, in addition to (H/4)(W/4)C(144) arithmetic instructions for the data transform, CK(72) for the filter transform, and (H/4)(W/4)K(100) for the inverse transform.

So the reason the number of arithmetic instructions in the transforms do not matter is that the multiplication stage is O(HWCK) while the transforms are O(HWC), O(CK), and O(HWK), respectively. So each of the transforms is less by a factor of K, HW, or C. If all of these dimensions are large, then the amount of arithmetic in the transforms is dominated by the multiplication stage.

# Can Winograd fast convolution algorithms be used with strided convolutions?
It is possible to use Winograd or any convolution algorithm with strides > 1 by decimating the input and filter to make un-strided convolutions, then adding the results.



