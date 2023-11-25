import torch
from zeta.structs import (
    AutoregressiveWrapper,
    Decoder,
    Transformer,
)


class GPT4(torch.nn.Module):
    """
    GPT4 is a transformer architecture that uses a ViT encoder and a transformer decoder.

    Args:

        image_size (int): Size of the image.
        patch_size (int): Size of the patch.
        encoder_dim (int): Dimension of the encoder.
        encoder_depth (int): Depth of the encoder.
        encoder_heads (int): Number of heads in the encoder.
        num_tokens (int): Number of tokens.
        max_seq_len (int): Maximum sequence length.
        decoder_dim (int): Dimension of the decoder.
        decoder_depth (int): Depth of the decoder.
        decoder_heads (int): Number of heads in the decoder.
        alibi_num_heads (int): Number of heads in the alibi attention.
        attn_kv_heads (int): Number of heads in the attention key-value projection.
        use_abs_pos_emb (bool): Whether to use absolute positional embeddings.
        cross_attend (bool): Whether to cross attend in the decoder.
        alibi_pos_bias (bool): Whether to use positional bias in the alibi attention.
        rotary_xpos (bool): Whether to use rotary positional embeddings.
        attn_flash (bool): Whether to use attention flash.
        qk_norm (bool): Whether to normalize the query and key in the attention layer.

    Returns:

            torch.Tensor: The output of the model.

    Usage:

            >>> img = torch.randn(1, 3, 256, 256)
            >>> text = torch.randint(0, 20000, (1, 1024))
            >>> model = GPT4()
            >>> output = model(img, text)
            >>> print(output)

    """

    def __init__(
        self,
        num_tokens=20000,
        max_seq_len=1024,
        decoder_dim=512,
        decoder_depth=6,
        decoder_heads=8,
        alibi_num_heads=4,
        attn_kv_heads=2,
        use_abs_pos_emb=False,
        alibi_pos_bias=True,
        rotary_xpos=True,
        attn_flash=True,
        qk_norm=True,
    ):
        super(GPT4, self).__init__()
        # palm model architecture
        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_kv_heads=attn_kv_heads,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            ),
        )

        # autoregressive wrapper to enable generation of tokens
        self.decoder = AutoregressiveWrapper(self.decoder)

    def forward(self, text: torch.Tensor):
        """Forward pass of the model."""
        try:
            return self.decoder(text)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise
