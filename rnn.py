"""
Character-level Vanilla RNN model implementation in Tinygrad. Inspired by Andrej Karpathy
https://gist.github.com/karpathy/d4dee566867f8291f086
"""

data = open("el_quijote.txt", "r").read()
chars = list(set(data))
data_size = 