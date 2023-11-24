import torch 

class ConvolutionalAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(ConvolutionalAttention, self).__init__()
        
        self.query_convolution = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.key_convolution = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.value_convolution = torch.nn.Conv2d(in_channels, in_channels, 1)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        q, k, v = x 

        if q.shape != k.shape or k.shape != v.shape:
            raise ValueError("q, k, v must have the same shape")

        previous_shape = q.shape

        q = self.query_convolution(q)
        k = self.key_convolution(k)
        v = self.value_convolution(v)

        q = self.activation(q)
        k = self.activation(k)
        v = self.activation(v)

        # Flatten q, k, v to get (B, C, H*W)
        q = torch.flatten(q, start_dim=2)
        k = torch.flatten(k, start_dim=2)
        v = torch.flatten(v, start_dim=2)

        k = torch.transpose(k, 1, 2) # (B, H*W, C)
        attention = torch.bmm(k, q) # (B, H*W, H*W)
        attention = torch.softmax(attention, dim=-1)
        attention = torch.bmm(v, attention) # (B, C, H*W)
        attention = torch.reshape(attention, previous_shape)

        return attention
    
def main():
    attention = ConvolutionalAttention(3)
    x = torch.randn(1, 3, 4, 6)
    y = torch.randn(1, 3, 4, 6)
    z = torch.randn(1, 3, 4, 6)
    output = attention((x, y, z))
    print(output.shape)

if __name__ == "__main__":
    main()

        