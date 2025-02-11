# Number Theoretic Transform (NTT) Implementation

This repository contains an implementation of the forward and backward Number Theoretic Transform (NTT) over Z_Q[X]/(X^N + 1), where N is a power of two and Q is an NTT-friendly prime satisfying Q ≡ 1 mod 2N. 

## Features

- **Base Task**: Implements NTT for primes up to 61 bits
- **Bonus Task 1**: Provides an optimized AVX2 implementation for 31-bit primes
- **Modular design** with support for custom parameters (prime modulus Q, root of unity ψ, and maximum transform size N)
- Includes unit tests to ensure correctness and benchmarks to measure performance

## How It Works

The implementation consists of:

- A base version that supports 61-bit primes using scalar operations
- An AVX2-optimized version for 31-bit primes, leveraging SIMD instructions for faster computation
- Uses Montgomery arithmetic for efficient modular multiplication
- Handles edge cases and ensures proper reduction during computations

## Usage

### Prerequisites

- Rust toolchain installed (rustc, cargo)
- For AVX2 optimizations, ensure your CPU supports AVX2 instructions

### Steps

**Run Tests**:
```bash
cargo test
```

**Run Benchmarks**:
```bash
cargo bench
```
Runs benchmarks to measure the performance of both the base and AVX2 implementations.

### Example

To use the NTT in your own code:

```rust
use ntt::Table;

fn main() {
    // Create a new NTT table with default parameters
    let ntt = Table::new();
    
    // Input vector (must have length equal to a power of two)
    let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    
    // Perform forward NTT
    ntt.forward_inplace(&mut data);
    
    // Perform backward NTT
    ntt.backward_inplace(&mut data);
    
    println!("Transformed data: {:?}", data);
}
```
