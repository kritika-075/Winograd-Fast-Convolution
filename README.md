# Winograd-Fast-Convolution
# Background
The success of convolutional neural networks is limited by how fast we can compute them. We use the winograd's minimal filtering algorithms for small, 3*3 filters on CNN for intel architecture, specifically the Knights Landing Architecture. However, the reduction of arithmetic operations in Winograd algorithm comes at the cost of complicating the memory accesses.
