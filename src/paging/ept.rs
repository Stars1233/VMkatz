//! Extended Page Table (EPT) walker for nested virtualization (VBS/Hyper-V).
//!
//! When VBS is enabled, Hyper-V runs inside the VM as a nested hypervisor.
//! ESXi captures L1 guest physical memory, but the Windows kernel runs in
//! L2 guest physical space mapped through Hyper-V's EPT (SLAT).
//!
//! This module finds and walks the EPT to reconstruct L2→L1 translation,
//! allowing us to scan L2 physical memory for EPROCESS structures.
//!
//! Optimization: EptLayer precomputes the full L2→L1 mapping on construction,
//! so subsequent reads use O(log n) binary search instead of 4-level walk.

use crate::error::{GovmemError, Result};
use crate::memory::PhysicalMemory;

const PAGE_SIZE: u64 = 4096;
const EPT_PRESENT_MASK: u64 = 0x7; // bits 2:0 = RWX
const EPT_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000; // bits 51:12
const EPT_LARGE_PAGE: u64 = 1 << 7; // bit 7 = large page

/// A contiguous L2→L1 mapping region from an EPT leaf entry.
#[derive(Debug, Clone, Copy)]
struct EptMapping {
    l2_base: u64, // L2 guest physical base address
    l1_base: u64, // L1 physical base address
    size: u64,    // Region size: 4KB, 2MB, or 1GB
}

/// EPT-translated physical memory layer with precomputed mappings.
/// Construction walks the full EPT once; reads use binary search.
pub struct EptLayer<'a, P: PhysicalMemory> {
    l1: &'a P,
    mappings: Vec<EptMapping>, // sorted by l2_base
    l2_size: u64,
    mapped_count: usize, // total number of mapped 4KB-equivalent pages
}

/// A scored EPT candidate, sorted by quality (non-zero translated pages).
#[derive(Debug)]
pub struct EptCandidate {
    pub pml4_addr: u64,
    pub l2_size: u64,
    pub valid_pml4e: u32,
    pub nonzero_pages: u32,
    pub total_sampled: u32,
}

impl<'a, P: PhysicalMemory> EptLayer<'a, P> {
    /// Create an EPT layer by precomputing all L2→L1 mappings.
    /// This walks the full 4-level EPT once, then all subsequent reads are O(log n).
    pub fn new(l1: &'a P, ept_pml4: u64, l2_size: u64) -> Self {
        let l1_size = l1.phys_size();
        let mut mappings = Vec::new();
        let mut mapped_pages: u64 = 0;

        let mut pml4_buf = [0u8; PAGE_SIZE as usize];
        if l1.read_phys(ept_pml4, &mut pml4_buf).is_ok() {
            for i in 0..512u64 {
                let pml4e = read_entry(&pml4_buf, i);
                if pml4e & EPT_PRESENT_MASK == 0 {
                    continue;
                }
                let pdpt_addr = pml4e & EPT_ADDR_MASK;
                if pdpt_addr >= l1_size {
                    continue;
                }

                let mut pdpt_buf = [0u8; PAGE_SIZE as usize];
                if l1.read_phys(pdpt_addr, &mut pdpt_buf).is_err() {
                    continue;
                }

                for j in 0..512u64 {
                    let pdpte = read_entry(&pdpt_buf, j);
                    if pdpte & EPT_PRESENT_MASK == 0 {
                        continue;
                    }

                    let l2_1g = (i << 39) | (j << 30);

                    // 1GB large page
                    if pdpte & EPT_LARGE_PAGE != 0 {
                        let l1_base = pdpte & 0x000F_FFFF_C000_0000;
                        if l1_base < l1_size {
                            mappings.push(EptMapping {
                                l2_base: l2_1g,
                                l1_base,
                                size: 1 << 30,
                            });
                            mapped_pages += 262144;
                        }
                        continue;
                    }

                    let pd_addr = pdpte & EPT_ADDR_MASK;
                    if pd_addr >= l1_size {
                        continue;
                    }

                    let mut pd_buf = [0u8; PAGE_SIZE as usize];
                    if l1.read_phys(pd_addr, &mut pd_buf).is_err() {
                        continue;
                    }

                    for k in 0..512u64 {
                        let pde = read_entry(&pd_buf, k);
                        if pde & EPT_PRESENT_MASK == 0 {
                            continue;
                        }

                        let l2_2m = l2_1g | (k << 21);

                        // 2MB large page
                        if pde & EPT_LARGE_PAGE != 0 {
                            let l1_base = pde & 0x000F_FFFF_FFE0_0000;
                            if l1_base < l1_size {
                                mappings.push(EptMapping {
                                    l2_base: l2_2m,
                                    l1_base,
                                    size: 1 << 21,
                                });
                                mapped_pages += 512;
                            }
                            continue;
                        }

                        let pt_addr = pde & EPT_ADDR_MASK;
                        if pt_addr >= l1_size {
                            continue;
                        }

                        let mut pt_buf = [0u8; PAGE_SIZE as usize];
                        if l1.read_phys(pt_addr, &mut pt_buf).is_err() {
                            continue;
                        }

                        for l in 0..512u64 {
                            let pte = read_entry(&pt_buf, l);
                            if pte & EPT_PRESENT_MASK == 0 {
                                continue;
                            }
                            let l1_addr = pte & EPT_ADDR_MASK;
                            if l1_addr >= l1_size {
                                continue;
                            }
                            mappings.push(EptMapping {
                                l2_base: l2_2m | (l << 12),
                                l1_base: l1_addr,
                                size: PAGE_SIZE,
                            });
                            mapped_pages += 1;
                        }
                    }
                }
            }
        }

        mappings.sort_by_key(|m| m.l2_base);

        log::info!(
            "EPT prebuilt: {} mapping regions, ~{} mapped pages ({} MB of L2 space)",
            mappings.len(),
            mapped_pages,
            mapped_pages * 4 / 1024,
        );

        Self {
            l1,
            mappings,
            l2_size,
            mapped_count: mapped_pages as usize,
        }
    }

    /// Number of mapped 4KB-equivalent pages.
    pub fn mapped_page_count(&self) -> usize {
        self.mapped_count
    }

    /// Translate L2 → L1 via binary search on precomputed mappings.
    fn translate_l2(&self, l2_phys: u64) -> Result<u64> {
        // Binary search: find the mapping region containing this L2 address
        let idx = self
            .mappings
            .partition_point(|m| m.l2_base + m.size <= l2_phys);

        if idx < self.mappings.len() {
            let m = &self.mappings[idx];
            if l2_phys >= m.l2_base && l2_phys < m.l2_base + m.size {
                let offset = l2_phys - m.l2_base;
                return Ok(m.l1_base + offset);
            }
        }

        Err(GovmemError::UnmappablePhysical(l2_phys))
    }

    /// Iterate over all mapped L2 page-aligned addresses and their L1 targets.
    /// Used for efficient EPROCESS scanning (scan L1 pages directly).
    pub fn mapped_pages(&self) -> MappedPageIter<'_> {
        MappedPageIter {
            mappings: &self.mappings,
            region_idx: 0,
            page_offset: 0,
        }
    }
}

/// Iterator over (L2_page_base, L1_page_addr) for all mapped pages.
pub struct MappedPageIter<'a> {
    mappings: &'a [EptMapping],
    region_idx: usize,
    page_offset: u64,
}

impl Iterator for MappedPageIter<'_> {
    type Item = (u64, u64); // (L2 page base, L1 page address)

    fn next(&mut self) -> Option<Self::Item> {
        while self.region_idx < self.mappings.len() {
            let m = &self.mappings[self.region_idx];
            if self.page_offset < m.size {
                let l2 = m.l2_base + self.page_offset;
                let l1 = m.l1_base + self.page_offset;
                self.page_offset += PAGE_SIZE;
                return Some((l2, l1));
            }
            self.region_idx += 1;
            self.page_offset = 0;
        }
        None
    }
}

impl<'a, P: PhysicalMemory> PhysicalMemory for EptLayer<'a, P> {
    fn read_phys(&self, phys_addr: u64, buf: &mut [u8]) -> Result<()> {
        // Handle reads that might cross page boundaries
        let mut offset = 0;
        while offset < buf.len() {
            let current_addr = phys_addr + offset as u64;
            let page_remaining = PAGE_SIZE as usize - (current_addr as usize & 0xFFF);
            let to_read = std::cmp::min(buf.len() - offset, page_remaining);

            let l1_addr = self.translate_l2(current_addr)?;
            self.l1
                .read_phys(l1_addr, &mut buf[offset..offset + to_read])?;
            offset += to_read;
        }
        Ok(())
    }

    fn phys_size(&self) -> u64 {
        self.l2_size
    }
}

/// Find all EPT candidates, ranked by quality (most non-zero translated pages first).
pub fn find_ept_candidates<P: PhysicalMemory>(l1: &P) -> Result<Vec<EptCandidate>> {
    let l1_size = l1.phys_size();

    log::info!(
        "EPT scan: searching {} MB for EPT PML4 tables...",
        l1_size / (1024 * 1024)
    );

    let mut page_buf = vec![0u8; PAGE_SIZE as usize];
    let mut candidates: Vec<EptCandidate> = Vec::new();

    let mut addr: u64 = 0;
    while addr < l1_size {
        if l1.read_phys(addr, &mut page_buf).is_err() {
            addr += PAGE_SIZE;
            continue;
        }

        if page_buf.iter().all(|&b| b == 0) {
            addr += PAGE_SIZE;
            continue;
        }

        let (valid, zero, invalid, max_l2) = score_ept_page(&page_buf, l1_size);

        if (1..=64).contains(&valid) && zero >= 400 && invalid == 0 {
            let (nonzero, sampled) = sample_nonzero_translated(l1, addr, l1_size, 64);

            let l2_size = std::cmp::min((max_l2 + 1) * (1u64 << 39), l1_size * 2);

            log::debug!(
                "EPT candidate at L1=0x{:x}: {} PML4E, {}/{} non-zero, l2={} MB",
                addr,
                valid,
                nonzero,
                sampled,
                l2_size / (1024 * 1024)
            );

            candidates.push(EptCandidate {
                pml4_addr: addr,
                l2_size,
                valid_pml4e: valid,
                nonzero_pages: nonzero,
                total_sampled: sampled,
            });
        }

        addr += PAGE_SIZE;
    }

    // Sort: prefer EPTs closer to Windows kernel (fewer PML4E = more specific).
    // A Windows kernel EPT typically has 1-4 PML4E covering 8-16 GB of L2 space.
    // Hypervisor-level EPTs have 8+ PML4E mapping all of L1 memory.
    // Primary: non-zero > 0 (must have data), then fewer PML4E preferred,
    // then non-zero count as tiebreaker.
    candidates.sort_by(|a, b| {
        // First: any non-zero data beats none
        let a_has = if a.nonzero_pages > 0 { 1u32 } else { 0 };
        let b_has = if b.nonzero_pages > 0 { 1u32 } else { 0 };
        b_has
            .cmp(&a_has)
            // Fewer PML4E = more likely Windows kernel EPT
            .then(a.valid_pml4e.cmp(&b.valid_pml4e))
            // More non-zero pages as tiebreaker
            .then(b.nonzero_pages.cmp(&a.nonzero_pages))
    });

    // Filter out zero-data candidates if we have any good ones
    let good_count = candidates.iter().filter(|c| c.nonzero_pages > 0).count();
    if good_count > 0 {
        candidates.retain(|c| c.nonzero_pages > 0);
    }

    log::info!(
        "EPT scan: {} candidates found{}",
        candidates.len(),
        if let Some(best) = candidates.first() {
            format!(
                ", best at L1=0x{:x} ({}/{} non-zero)",
                best.pml4_addr, best.nonzero_pages, best.total_sampled
            )
        } else {
            String::new()
        }
    );

    if candidates.is_empty() {
        return Err(GovmemError::SystemProcessNotFound);
    }

    Ok(candidates)
}

/// Convenience wrapper: returns the single best EPT root.
pub fn find_ept_root<P: PhysicalMemory>(l1: &P) -> Result<(u64, u64)> {
    let candidates = find_ept_candidates(l1)?;
    let best = candidates
        .first()
        .ok_or(GovmemError::SystemProcessNotFound)?;
    Ok((best.pml4_addr, best.l2_size))
}

#[inline]
fn read_entry(buf: &[u8], idx: u64) -> u64 {
    let off = (idx * 8) as usize;
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

/// Score a page as a potential EPT PML4 table.
fn score_ept_page(page: &[u8], l1_size: u64) -> (u32, u32, u32, u64) {
    let mut valid = 0u32;
    let mut zero = 0u32;
    let mut invalid = 0u32;
    let mut max_idx: u64 = 0;

    for i in 0..512u64 {
        let entry = read_entry(page, i);
        if entry == 0 {
            zero += 1;
            continue;
        }

        let rwx = entry & EPT_PRESENT_MASK;
        let phys = entry & EPT_ADDR_MASK;

        if rwx != 0 && phys < l1_size && phys != 0 {
            valid += 1;
            max_idx = i;
        } else {
            invalid += 1;
        }
    }

    (valid, zero, invalid, max_idx)
}

/// Sample translated pages to verify an EPT candidate maps non-zero data.
fn sample_nonzero_translated<P: PhysicalMemory>(
    l1: &P,
    pml4_addr: u64,
    l1_size: u64,
    max_samples: usize,
) -> (u32, u32) {
    let mut targets: Vec<u64> = Vec::with_capacity(max_samples);
    let mut pml4_buf = [0u8; PAGE_SIZE as usize];

    if l1.read_phys(pml4_addr, &mut pml4_buf).is_err() {
        return (0, 0);
    }

    'outer: for i in 0..512u64 {
        let pml4e = read_entry(&pml4_buf, i);
        if pml4e & EPT_PRESENT_MASK == 0 {
            continue;
        }
        let pdpt_addr = pml4e & EPT_ADDR_MASK;
        if pdpt_addr >= l1_size {
            continue;
        }

        let mut pdpt_buf = [0u8; PAGE_SIZE as usize];
        if l1.read_phys(pdpt_addr, &mut pdpt_buf).is_err() {
            continue;
        }

        for j in 0..512u64 {
            let pdpte = read_entry(&pdpt_buf, j);
            if pdpte & EPT_PRESENT_MASK == 0 {
                continue;
            }

            if pdpte & EPT_LARGE_PAGE != 0 {
                let base = pdpte & 0x000F_FFFF_C000_0000;
                if base < l1_size {
                    targets.push(base);
                }
                if targets.len() >= max_samples {
                    break 'outer;
                }
                continue;
            }

            let pd_addr = pdpte & EPT_ADDR_MASK;
            if pd_addr >= l1_size {
                continue;
            }

            let mut pd_buf = [0u8; PAGE_SIZE as usize];
            if l1.read_phys(pd_addr, &mut pd_buf).is_err() {
                continue;
            }

            for k in 0..512u64 {
                let pde = read_entry(&pd_buf, k);
                if pde & EPT_PRESENT_MASK == 0 {
                    continue;
                }

                if pde & EPT_LARGE_PAGE != 0 {
                    let base = pde & 0x000F_FFFF_FFE0_0000;
                    if base < l1_size {
                        targets.push(base);
                    }
                    if targets.len() >= max_samples {
                        break 'outer;
                    }
                    continue;
                }

                let pt_addr = pde & EPT_ADDR_MASK;
                if pt_addr >= l1_size {
                    continue;
                }

                let mut pt_buf = [0u8; PAGE_SIZE as usize];
                if l1.read_phys(pt_addr, &mut pt_buf).is_err() {
                    continue;
                }

                for l in 0..512u64 {
                    let pte = read_entry(&pt_buf, l);
                    if pte & EPT_PRESENT_MASK == 0 {
                        continue;
                    }
                    let target = pte & EPT_ADDR_MASK;
                    if target < l1_size {
                        targets.push(target);
                    }
                    if targets.len() >= max_samples {
                        break 'outer;
                    }
                }
            }
        }
    }

    let mut nonzero = 0u32;
    let total = targets.len() as u32;
    let mut page_buf = [0u8; PAGE_SIZE as usize];

    for &target in &targets {
        if l1.read_phys(target, &mut page_buf).is_ok() && !page_buf.iter().all(|&b| b == 0) {
            nonzero += 1;
        }
    }

    (nonzero, total)
}
