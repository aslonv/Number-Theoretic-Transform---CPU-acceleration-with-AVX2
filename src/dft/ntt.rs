// Copyright 2025 Begali Aslonov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Contact:
/// Email: bekali.aslonov@gmail.com
/// Github: github.com/aslonv
/// 𝕏: x.com/aslonv

/// Implementation of Number Theoretic Transform (NTT) over ZQ[X]/XN+1
/// where N is a power of two and Q is an NTT-friendly prime satisfying Q≡1mod2N.
/// This implementation provides both a base version for 61-bit primes and
/// an optimized AVX2 version for 31-bit primes.
use crate::dft::DFT;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct Table<O> {
    q: O,
    forward_twiddles: Vec<Vec<O>>,
    inverse_twiddles: Vec<Vec<O>>,
    inv_n_table: Vec<O>,
    #[cfg(target_arch = "x86_64")]
    montgomery: Option<Montgomery32>,
}

impl<O> Table<O> {
    pub fn q(&self) -> &O {
        &self.q
    }
}

#[cfg(target_arch = "x86_64")]
impl Table<u64> {
    /// NTT table using the reference 61-bit prime and root of unity.
    pub fn new() -> Self {
        const Q: u64 = 0x1fffffffffe00001;
        const PSI: u64 = 0x15eb043c7aa2b01f;
        let max_log_n = 17;
        
        let (forward_twiddles, inverse_twiddles) = Self::precompute_twiddles(Q, PSI, max_log_n);
        let inv_n_table = Self::precompute_inverses(Q, max_log_n);

        #[cfg(target_arch = "x86_64")]
        let montgomery = if Q < (1 << 31) {
            Some(Montgomery32::new(Q as u32))
        } else {
            None
        };

        Self {
            q: Q,
            forward_twiddles,
            inverse_twiddles,
            inv_n_table,
            #[cfg(target_arch = "x86_64")]
            montgomery,
        }
    }

    /// NTT table with custom parameters.
    pub fn with_params(q: u64, psi: u64, max_log_n: usize) -> Self {
        let (forward_twiddles, inverse_twiddles) = Self::precompute_twiddles(q, psi, max_log_n);
        let inv_n_table = Self::precompute_inverses(q, max_log_n);

        #[cfg(target_arch = "x86_64")]
        let montgomery = if q < (1 << 31) {
            Some(Montgomery32::new(q as u32))
        } else {
            None
        };

        Self {
            q,
            forward_twiddles,
            inverse_twiddles,
            inv_n_table,
            #[cfg(target_arch = "x86_64")]
            montgomery,
        }
    }

    fn precompute_twiddles(q: u64, psi: u64, max_log_n: usize) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        let mut forward_twiddles = Vec::with_capacity(max_log_n);
        let mut inverse_twiddles = Vec::with_capacity(max_log_n);

        for s in 1..=max_log_n {
            let m = 1 << s;
            let exponent = (1 << (max_log_n - s)) as u64;
            let w = pow_mod(psi, exponent, q);
            let w_inv = pow_mod(w, q - 2, q);

            let mut fwd = Vec::with_capacity(m >> 1);
            let mut inv = Vec::with_capacity(m >> 1);
            let mut current_fwd = 1u64;
            let mut current_inv = 1u64;

            for _ in 0..(m >> 1) {
                fwd.push(current_fwd);
                inv.push(current_inv);
                current_fwd = mul_mod(current_fwd, w, q);
                current_inv = mul_mod(current_inv, w_inv, q);
            }

            forward_twiddles.push(fwd);
            inverse_twiddles.push(inv);
        }

        (forward_twiddles, inverse_twiddles)
    }

    fn precompute_inverses(q: u64, max_log_n: usize) -> Vec<u64> {
        (0..=max_log_n)
            .map(|k| pow_mod(1 << k, q - 2, q))
            .collect()
    }

    fn bit_reverse(a: &mut [u64]) {
        let n = a.len();
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j >= bit {
                j -= bit;
                bit >>= 1;
            }
            j += bit;
            if i < j {
                a.swap(i, j);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn bit_reverse_u32(a: &mut [u32]) {
        let n = a.len();
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j >= bit {
                j -= bit;
                bit >>= 1;
            }
            j += bit;
            if i < j {
                a.swap(i, j);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn avx2_forward_inplace(&self, a: &mut [u64], mont: &Montgomery32) {
        let mut a32 = a.iter()
            .map(|x| {
                // Input is within 32-bit range before converting
                let x_reduced = (*x % self.q) as u32;
                mont.to_montgomery(x_reduced)
            })
            .collect::<Vec<u32>>();

        let n = a32.len();
        let log_n = n.trailing_zeros() as usize;
        Self::bit_reverse_u32(&mut a32);

        for s in 1..=log_n {
            let m = 1 << s;
            let m_half = m >> 1;
            let twiddles = &self.forward_twiddles[s - 1];

            for k in (0..n).step_by(m) {
                for j in (0..m_half).step_by(8) {
                    if j + 8 <= m_half {
                        let ptr = a32.as_mut_ptr().add(k + j);
                        let a_vec = _mm256_loadu_si256(ptr as _);
                        let b_vec = _mm256_loadu_si256(ptr.add(m_half) as _);

                        // Converts twiddles to Montgomery form, ensuring 32-bit bounds
                        let tw = [
                            mont.to_montgomery((twiddles[j] % self.q) as u32),
                            mont.to_montgomery((twiddles[j + 1] % self.q) as u32),
                            mont.to_montgomery((twiddles[j + 2] % self.q) as u32),
                            mont.to_montgomery((twiddles[j + 3] % self.q) as u32),
                            mont.to_montgomery((twiddles[j + 4] % self.q) as u32),
                            mont.to_montgomery((twiddles[j + 5] % self.q) as u32),
                            mont.to_montgomery((twiddles[j + 6] % self.q) as u32),
                            mont.to_montgomery((twiddles[j + 7] % self.q) as u32),
                        ];

                        let w = _mm256_set_epi32(
                            tw[7] as i32, tw[6] as i32,
                            tw[5] as i32, tw[4] as i32,
                            tw[3] as i32, tw[2] as i32,
                            tw[1] as i32, tw[0] as i32,
                        );

                        let t = mont.mul_avx2(b_vec, w);
                        let sum = mont.add_avx2(a_vec, t);
                        let diff = mont.sub_avx2(a_vec, t);

                        _mm256_storeu_si256(ptr as _, sum);
                        _mm256_storeu_si256(ptr.add(m_half) as _, diff);
                    } else {
                        // Scalar fallback for remaining elements
                        for j in j..m_half {
                            let idx1 = k + j;
                            let idx2 = k + j + m_half;
                            let t = mont.mul(
                                a32[idx2],
                                mont.to_montgomery((twiddles[j] % self.q) as u32)
                            );
                            a32[idx2] = mont.sub(a32[idx1], t);
                            a32[idx1] = mont.add(a32[idx1], t);
                        }
                    }
                }
            }
        }

        for (i, x) in a32.iter().enumerate() {
            let val = mont.from_montgomery(*x) as u64;
            a[i] = val % self.q;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn avx2_backward_inplace(&self, a: &mut [u64], mont: &Montgomery32) {
        let mut a32 = a.iter()
            .map(|x| {
                // Input is within 32-bit range
                let x_reduced = (*x % self.q) as u32;
                mont.to_montgomery(x_reduced)
            })
            .collect::<Vec<u32>>();

        let n = a32.len();
        let log_n = n.trailing_zeros() as usize;
        Self::bit_reverse_u32(&mut a32);

        for s in 1..=log_n {
            let m = 1 << s;
            let m_half = m >> 1;
            let twiddles = &self.inverse_twiddles[s - 1];

            for k in (0..n).step_by(m) {
                for j in (0..m_half).step_by(8) {
                    if j + 8 <= m_half {
                        let ptr = a32.as_mut_ptr().add(k + j);
                        let a_vec = _mm256_loadu_si256(ptr as _);
                        let b_vec = _mm256_loadu_si256(ptr.add(m_half) as _);

                        let tw = [
                            mont.to_montgomery((twiddles[j] % self.q) as u32),
                            mont.to_montgomery((twiddles[j+1] % self.q) as u32),
                            mont.to_montgomery((twiddles[j+2] % self.q) as u32),
                            mont.to_montgomery((twiddles[j+3] % self.q) as u32),
                            mont.to_montgomery((twiddles[j+4] % self.q) as u32),
                            mont.to_montgomery((twiddles[j+5] % self.q) as u32),
                            mont.to_montgomery((twiddles[j+6] % self.q) as u32),
                            mont.to_montgomery((twiddles[j+7] % self.q) as u32),
                        ];

                        let w = _mm256_set_epi32(
                            tw[7] as i32, tw[6] as i32,
                            tw[5] as i32, tw[4] as i32,
                            tw[3] as i32, tw[2] as i32,
                            tw[1] as i32, tw[0] as i32,
                        );

                        let t = mont.mul_avx2(b_vec, w);
                        let sum = mont.add_avx2(a_vec, t);
                        let diff = mont.sub_avx2(a_vec, t);

                        _mm256_storeu_si256(ptr as _, sum);
                        _mm256_storeu_si256(ptr.add(m_half) as _, diff);
                    } else {
                        for j in j..m_half {
                            let idx1 = k + j;
                            let idx2 = k + j + m_half;
                            let t = mont.mul(
                                a32[idx2],
                                mont.to_montgomery((twiddles[j] % self.q) as u32)
                            );
                            a32[idx2] = mont.sub(a32[idx1], t);
                            a32[idx1] = mont.add(a32[idx1], t);
                        }
                    }
                }
            }
        }

        // Inverse scaling in Montgomery form
        let inv_n = (self.inv_n_table[log_n] % self.q) as u32;
        let inv_n_mont = mont.to_montgomery(inv_n);
        for x in a32.iter_mut() {
            *x = mont.mul(*x, inv_n_mont);
        }

        // Converting back to normal form with proper reduction
        for (i, x) in a32.iter().enumerate() {
            let val = mont.from_montgomery(*x) as u64;
            a[i] = val % self.q;
        }
    }

    fn forward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        #[cfg(target_arch = "x86_64")]
        if let Some(mont) = &self.montgomery {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.avx2_forward_inplace(a, mont) };
            }
        }

        let q = self.q;
        let n = a.len();
        let log_n = n.trailing_zeros() as usize;

        Self::bit_reverse(a);

        for s in 1..=log_n {
            let m = 1 << s;
            let m_half = m >> 1;
            let twiddles = &self.forward_twiddles[s - 1];

            for k in (0..n).step_by(m) {
                for j in 0..m_half {
                    let idx1 = k + j;
                    let idx2 = k + j + m_half;
                    let t = mul_mod(twiddles[j], a[idx2], q);
                    let a1 = a[idx1];
                    a[idx2] = sub_mod::<LAZY>(a1, t, q);
                    a[idx1] = add_mod::<LAZY>(a1, t, q);
                }
            }
        }
    }

    fn backward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        #[cfg(target_arch = "x86_64")]
        if let Some(mont) = &self.montgomery {
            if is_x86_feature_detected!("avx2") {
                // AVX2 implementation
                return unsafe { self.avx2_backward_inplace(a, mont) };
            }
        }

        let q = self.q;
        let n = a.len();
        let log_n = n.trailing_zeros() as usize;

        Self::bit_reverse(a);

        for s in 1..=log_n {
            let m = 1 << s;
            let m_half = m >> 1;
            let twiddles = &self.inverse_twiddles[s - 1];

            for k in (0..n).step_by(m) {
                for j in 0..m_half {
                    let idx1 = k + j;
                    let idx2 = k + j + m_half;
                    let t = mul_mod(twiddles[j], a[idx2], q);
                    let a1 = a[idx1];
                    a[idx2] = sub_mod::<LAZY>(a1, t, q);
                    a[idx1] = add_mod::<LAZY>(a1, t, q);
                }
            }
        }

        if !LAZY {
            let inv_n = self.inv_n_table[log_n];
            for elem in a.iter_mut() {
                *elem = mul_mod(*elem, inv_n, q);
            }
        }
    }

}

/// For computing Discrete Fourier Transforms over finite fields
impl DFT<u64> for Table<u64> {
    fn forward_inplace(&self, a: &mut [u64]) {
        self.forward_inplace_core::<false>(a)
    }

    fn forward_inplace_lazy(&self, a: &mut [u64]) {
        self.forward_inplace_core::<true>(a)
    }

    fn backward_inplace(&self, a: &mut [u64]) {
        self.backward_inplace_core::<false>(a)
    }

    fn backward_inplace_lazy(&self, a: &mut [u64]) {
        self.backward_inplace_core::<true>(a)
    }
}

// Montgomery arithmetic to make modular operations more efficient
#[cfg(target_arch = "x86_64")]
struct Montgomery32 {
    q: u32,
    qinv: u32,
    r2: u32,
}

#[cfg(target_arch = "x86_64")]
impl Montgomery32 {
    fn new(q: u32) -> Self {
        assert!(q % 2 == 1 && q > 3 && q < (1 << 31));
        
        // Newton-Raphson for modular inverse
        let mut inv = q;
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(inv)));
        }
        let qinv = inv.wrapping_neg();
        let r2 = (1u128 << 64) % q as u128;
        Self { q, qinv, r2: r2 as u32 }
    }

    #[inline]
    fn to_montgomery(&self, x: u32) -> u32 {
        self.mul(x, self.r2)
    }

    #[inline]
    fn from_montgomery(&self, x: u32) -> u32 {
        self.mul(x, 1)
    }

    #[inline]
    fn mul(&self, a: u32, b: u32) -> u32 {
        let product = a as u64 * b as u64;
        let tmp = (product as u32).wrapping_mul(self.qinv) as u64;
        ((product + tmp * self.q as u64) >> 32) as u32
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(&self, a: __m256i, b: __m256i) -> __m256i {
        let q = _mm256_set1_epi32(self.q as i32);
        let qinv = _mm256_set1_epi32(self.qinv as i32);

        // Perform 32x32->64 bit multiplication
        let t = _mm256_mul_epu32(a, b);
        
        // For Montgomery reduction, we need to compute (t + (t * q_inv mod 2^32) * q) >> 32
        let t_low = _mm256_and_si256(t, _mm256_set1_epi64x(0xFFFFFFFF));
        let qinv_64 = _mm256_and_si256(qinv, _mm256_set1_epi64x(0xFFFFFFFF));
        let m = _mm256_mul_epu32(t_low, qinv_64);
        let mq = _mm256_mul_epu32(m, q);
        let t_plus_mq = _mm256_add_epi64(t, mq);
        let result = _mm256_srli_epi64(t_plus_mq, 32);
        
        // Pack back to 32-bit values
        let packed = _mm256_permutevar8x32_epi32(result, 
            _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
        
        self.reduce_avx2(packed)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(&self, a: __m256i, b: __m256i) -> __m256i {
        let sum = _mm256_add_epi32(a, b);
        self.reduce_avx2(sum)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sub_avx2(&self, a: __m256i, b: __m256i) -> __m256i {
        let q = _mm256_set1_epi32(self.q as i32);
        let a_plus_q = _mm256_add_epi32(a, q);
        let diff = _mm256_sub_epi32(a_plus_q, b);
        self.reduce_avx2(diff)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn reduce_avx2(&self, x: __m256i) -> __m256i {
        let q = _mm256_set1_epi32(self.q as i32);
        let mask = _mm256_cmpgt_epi32(x, _mm256_sub_epi32(q, _mm256_set1_epi32(1)));
        let x_minus_q = _mm256_sub_epi32(x, q);
        _mm256_blendv_epi8(x, x_minus_q, mask)
    }

    #[inline]
    fn add(&self, a: u32, b: u32) -> u32 {
        let sum = a + b;
        if sum >= self.q {
            sum - self.q
        } else {
            sum
        }
    }

    #[inline]
    fn sub(&self, a: u32, b: u32) -> u32 {
        if a < b {
            a + self.q - b
        } else {
            a - b
        }
    }
}

// HELPERs
#[inline]
fn pow_mod(a: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u128;
    let mut a = a as u128 % modulus as u128;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * a) % modulus as u128;
        }
        a = (a * a) % modulus as u128;
        exp >>= 1;
    }
    result as u64
}

#[inline]
fn mul_mod(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

#[inline(always)]
fn add_mod<const LAZY: bool>(a: u64, b: u64, modulus: u64) -> u64 {
    if LAZY {
        a + b
    } else {
        ((a as u128 + b as u128) % modulus as u128) as u64
    }
}

#[inline(always)]
fn sub_mod<const LAZY: bool>(a: u64, b: u64, modulus: u64) -> u64 {
    if LAZY {
        a.wrapping_sub(b)
    } else {
        let a = a as u128;
        let b = b as u128;
        let modulus = modulus as u128;
        ((a + modulus - b) % modulus) as u64
    }
}

// TESTS
#[cfg(test)]
mod tests {
    use super::*;

    /// Tests full NTT cycle (forward + backward) with base implementation
    #[test]
    fn test_full_cycle() {
        let ntt = Table::new();
        let q = ntt.q;
        
        let mut a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let original = a.clone();
        
        ntt.forward_inplace(&mut a);
        ntt.backward_inplace(&mut a);
        
        for (i, &val) in a.iter().enumerate() {
            assert_eq!(val, original[i] % q);
        }
    }

    /// Tests AVX2 implementation with 31-bit prime
    #[test]
    fn test_avx2_32bit() {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
               
                let q = 0x7FFFFFFFu64; 
                let n: u32 = 8; 
                let psi = pow_mod(7, (q - 1) / (2 * n as u64), q); // Different root of unity
                let ntt = Table::with_params(q, psi, (n as u32).ilog2() as usize);
                
                let mut a: Vec<u64> = (0..n).map(|x| (x as u64) % q).collect();
                let original = a.clone();
                
                ntt.forward_inplace(&mut a);
                ntt.backward_inplace(&mut a);
                
                for (i, &val) in a.iter().enumerate() {
                    assert_eq!(val % q, original[i] % q,
                        "Mismatch at index {}: got {}, expected {}",
                        i, val % q, original[i] % q);
                }
            }
        }
    }

    /// Tests edge cases in Montgomery arithmetic
    #[test]
    fn test_avx2_edge_cases() {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let q = 0x7FFFFFFFu32; // 31-bit prime
                let mont = Montgomery32::new(q);
                
                // Test with smaller values first
                let values = vec![
                    0u32,
                    1u32,
                    2u32,
                    q - 1,
                    q - 2,
                    q / 2,
                    q / 2 + 1
                ];
                
                for &x in &values {
                    for &y in &values {
                        let expected = ((x as u64 + y as u64) % q as u64) as u32;
                        let x_mont = mont.to_montgomery(x);
                        let y_mont = mont.to_montgomery(y);
                        
                        unsafe {
                            let vec_x = _mm256_set1_epi32(x_mont as i32);
                            let vec_y = _mm256_set1_epi32(y_mont as i32);
                            let result = mont.add_avx2(vec_x, vec_y);
                            let result_scalar = _mm256_extract_epi32(result, 0) as u32;
                            let final_result = mont.from_montgomery(result_scalar);
                            
                            assert_eq!(final_result, expected,
                                "Failed on x={}, y={}: got {}, expected {}",
                                x, y, final_result, expected);
                        }
                    }
                }
            }
        }
    }
}
