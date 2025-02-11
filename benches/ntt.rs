use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use app::dft::ntt::Table;
use app::dft::DFT;

fn ntt_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT");
    
    // Base 61-bit implementation
    let base_table = Table::new();
    for log_n in [11, 12, 13, 14, 15, 16] {
        let n = 1 << log_n;
        let mut a = vec![0u64; n];
        for i in 0..n {
            a[i] = i as u64 % base_table.q();
        }
        
        group.bench_with_input(BenchmarkId::new("Base", n), &n, |b, _| {
            b.iter(|| {
                base_table.forward_inplace(&mut a);
                base_table.backward_inplace(&mut a);
            })
        });
    }

    // AVX2 32-bit implementation
    #[cfg(target_arch = "x86_64")]
    {
        const AVX_Q: u64 = 0xFFF00001;
        const AVX_PSI: u64 = 0x7A329F10;
        let avx_table = Table::with_params(AVX_Q, AVX_PSI, 20);
        
        for log_n in [11, 12, 13, 14, 15, 16] {
            let n = 1 << log_n;
            let mut a = vec![0u64; n];
            for i in 0..n {
                a[i] = i as u64 % AVX_Q;
            }
            
            group.bench_with_input(BenchmarkId::new("AVX2", n), &n, |b, _| {
                b.iter(|| {
                    avx_table.forward_inplace(&mut a);
                    avx_table.backward_inplace(&mut a);
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, ntt_bench);
criterion_main!(benches);