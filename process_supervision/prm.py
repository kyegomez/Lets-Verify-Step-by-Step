import torch
from zeta.structs import (
  AutoregressiveWrapper,
  Decoder,
  Encoder,
  Transformer,
  ViTransformerWrapper,
)
from torch import nn

class PRM(torch.nn.Module):
    """
    PRM is a transformer architecture that uses a ViT encoder and a transformer decoder.

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
            >>> model = PRM()
            >>> output = model(img, text)
            >>> print(output)

    """

    def __init__(
        self,
        image_size=256,
        patch_size=32,
        encoder_dim=512,
        encoder_depth=6,
        encoder_heads=8,
        num_tokens=20000,
        max_seq_len=1024,
        decoder_dim=512,
        decoder_depth=6,
        decoder_heads=8,
        alibi_num_heads=4,
        attn_kv_heads=2,
        use_abs_pos_emb=False,
        cross_attend=True,
        alibi_pos_bias=True,
        rotary_xpos=True,
        attn_flash=True,
        qk_norm=True,
        num_classes: int = 10
    ):
        super(PRM, self).__init__()

        # vit architecture
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim, depth=encoder_depth, heads=encoder_heads
            ),
        )

        # palm model architecture
        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=cross_attend,
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


        self.classification_head = nn.Linear(decoder_dim, num_classes)


    def forward(self, img: torch.Tensor, text: torch.Tensor):
        try:
            encoded = self.encoder(img, return_embeddings=True)
            decoder_output = self.decoder(text, context=encoded)

            # Check and process the output tensor
            output_tensor = decoder_output[0] if isinstance(decoder_output, tuple) else decoder_output
            # Ensure that output_tensor is of shape [batch_size, decoder_dim]
            # Modify the line below if the output tensor needs reshaping
            cls_token_output = output_tensor[:, 0]

            # Pass through the classification head
            class_logits = self.classification_head(cls_token_output)

            return class_logits
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise



# Usage with random inputs
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

# Initiliaze the model
model = PRM()
output = model(img, text)
print(output)

