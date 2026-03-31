import torch

print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1', device='cuda')
    emb = model.encode(['test sentence'])
    print(f'Embedding shape: {emb.shape}, device used: cuda')
else:
    print('No CUDA device found')
