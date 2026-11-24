"""Exercise training/save/reload with a tiny random CLIP and synthetic fixtures.

This is a software smoke test, NOT trained research weights or a quality result.
All fixture data and weights disappear with the temporary directory.
"""
import argparse
import json
from pathlib import Path
import tempfile


def main():
    import torch
    from PIL import Image
    from transformers import (CLIPConfig, CLIPModel, CLIPProcessor,
                              CLIPImageProcessor, CLIPTokenizer)
    from train_sticker_encoder import train
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='xiaoyu-training-smoke-') as temporary:
        root = Path(temporary)
        base, data, output = root/'base', root/'data', root/'output'
        base.mkdir(); data.mkdir()
        vocab = {'<|startoftext|>': 0, '<|endoftext|>': 1}
        for char in 'abcdefghijklmnopqrstuvwxyz':
            vocab[char] = len(vocab)
            vocab[char+'</w>'] = len(vocab)
        (base/'vocab.json').write_text(json.dumps(vocab))
        (base/'merges.txt').write_text('#version: 0.2\n')
        tokenizer = CLIPTokenizer(vocab_file=str(base/'vocab.json'),
                                  merges_file=str(base/'merges.txt'), model_max_length=32)
        processor = CLIPProcessor(tokenizer=tokenizer,
            image_processor=CLIPImageProcessor(size={'shortest_edge': 32},
                                               crop_size={'height': 32, 'width': 32}))
        config = CLIPConfig(projection_dim=16,
            text_config={'vocab_size': len(vocab), 'hidden_size': 32,
                         'intermediate_size': 64, 'num_hidden_layers': 1,
                         'num_attention_heads': 4, 'max_position_embeddings': 32,
                         'bos_token_id': 0, 'eos_token_id': 1, 'pad_token_id': 1},
            vision_config={'hidden_size': 32, 'intermediate_size': 64,
                           'num_hidden_layers': 1, 'num_attention_heads': 4,
                           'image_size': 32, 'patch_size': 16})
        CLIPModel(config).save_pretrained(base, safe_serialization=True)
        processor.save_pretrained(base)
        for i in range(8):
            image = Image.new('RGB', (32, 32), (i*30, 250-i*20, i*10))
            image.putpixel((i, i), (255, 255, 255))
            image.save(data/(str(i)+'.png'))
        splits = {
            'train': [{'image': str(i)+'.png', 'text': text, 'reviewed': True}
                      for i, text in enumerate(['hello', 'goodbye'])],
            'validation': [{'left': '2.png', 'right': str(i)+'.png',
                            'label': int(i == 3), 'reviewed': True} for i in (3, 4)],
            'test': [{'left': '5.png', 'right': str(i)+'.png',
                      'label': int(i == 6), 'reviewed': True} for i in (6, 7)]}
        for split, rows in splits.items():
            (data/(split+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows))
        train(argparse.Namespace(dataset=str(data), base=str(base), output=str(output),
              epochs=1, batch_size=2, lr=1e-4, seed=42, device='cpu'))
        manifest = json.loads((output/'reproduction.json').read_text())
        assert manifest['state'] == 'trained_evaluated' and not manifest['author_original']
        model = CLIPModel.from_pretrained(output, local_files_only=True, use_safetensors=True)
        processed = CLIPProcessor.from_pretrained(output, local_files_only=True)(
            images=Image.open(data/'7.png'), return_tensors='pt')
        with torch.inference_mode():
            vector = model.get_image_features(**processed)
        assert vector.shape == (1, 16) and torch.isfinite(vector).all()
    print('PASS: synthetic forward/backward, validation, save and reload; no research model produced.')


if __name__ == '__main__':
    main()
