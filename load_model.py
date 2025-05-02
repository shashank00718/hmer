import pickle
import torch
from encoder import DenseMD
from decoder import AttnDecoderCausal

def load_trained_model(model_path='trained_model.pkl'):
    # Load the complete model state
    with open(model_path, 'rb') as f:
        model_state = pickle.load(f)
    
    # Create config object
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = Config(model_state['config'])
    
    # Initialize models
    encoder = DenseMD(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        hidden_channels=config.HIDDEN_SIZE
    ).to(config.DEVICE)
    
    decoder = AttnDecoderCausal(
        hidden_size=config.HIDDEN_SIZE,
        output_size=len(model_state['vocabulary'])
    ).to(config.DEVICE)
    
    # Load states
    encoder.load_state_dict(model_state['encoder_state'])
    decoder.load_state_dict(model_state['decoder_state'])
    
    return encoder, decoder, config, model_state['vocabulary'], model_state['metadata']

# Example usage
if __name__ == '__main__':
    encoder, decoder, config, vocab, metadata = load_trained_model()
    print(f"Loaded model with best accuracy: {metadata['best_accuracy']}")
    print(f"Best epoch: {metadata['best_epoch']}")
