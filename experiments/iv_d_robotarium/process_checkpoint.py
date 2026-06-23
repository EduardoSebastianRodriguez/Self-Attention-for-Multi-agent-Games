import torch
import numpy as np
from collections import OrderedDict

def process_checkpoint(checkpoint_path: str, output_path: str = 'actor_weights.npy'):
    """
    Loads a PyTorch checkpoint, extracts actor weights for two teams,
    and saves them to a .npy file.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint file.
        output_path (str): Path to save the output .npy file.
    """
    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        # Load the checkpoint onto the CPU
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print("Checkpoint loaded successfully.")

        policy_state_dict = checkpoint['collector']['policy_state_dict']

        # --- Process and collect weights ---
        team1_weights = {}
        team2_weights = {}
        
        prefix_team1 = 'module.0.module.0.module.0.'
        prefix_team2 = 'module.1.module.0.module.0.'

        print("\n--- Extracting Team 1 Actor Weights ---")
        for key, value in policy_state_dict.items():
            if key.startswith(prefix_team1):
                # Remove the prefix to get a clean layer name
                new_key = key[len(prefix_team1):]
                team1_weights[new_key] = value.numpy()
                print(f"Found layer: {new_key} with shape: {value.shape}")

        print("\n--- Extracting Team 2 Actor Weights ---")
        for key, value in policy_state_dict.items():
            if key.startswith(prefix_team2):
                # Remove the prefix to get a clean layer name
                new_key = key[len(prefix_team2):]
                team2_weights[new_key] = value.numpy()
                print(f"Found layer: {new_key} with shape: {value.shape}")

        # --- Save the collected weights ---
        if not team1_weights and not team2_weights:
            print("\nWarning: No weights were found with the specified prefixes. Nothing to save.")
            return

        # Combine into a single dictionary for saving
        all_weights_to_save = {
            'team1': team1_weights,
            'team2': team2_weights
        }

        np.save(output_path, all_weights_to_save)
        print(f"\n Successfully saved weights to {output_path}")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
    except KeyError as e:
        print(f"Error: Could not find key {e} in the checkpoint. Check the checkpoint structure.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    CHECKPOINT_FILE = 'checkpoint_3000000.pt'
    process_checkpoint(CHECKPOINT_FILE)