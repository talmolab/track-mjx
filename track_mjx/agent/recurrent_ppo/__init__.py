"""Recurrent PPO implementation with VAE-style intention encoding.

This module provides a recurrent variant of the intention-based PPO algorithm
where the decoder uses an RNN (SimpleCell, GRU, or LSTM) instead of a
feedforward MLP. The encoder remains an MLP that maps trajectory observations
to latent intentions.
"""
