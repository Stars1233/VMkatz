use crate::error::{GovmemError, Result};
use crate::memory::{PhysicalMemory, VirtualMemory};
use crate::paging::translate::{PageTableWalker, ProcessMemory};
use crate::windows::eprocess::EprocessReader;
use crate::windows::offsets::EprocessOffsets;

/// PEB + 0x20 = ProcessParameters (RTL_USER_PROCESS_PARAMETERS*)
const PEB_PROCESS_PARAMETERS: u64 = 0x20;
/// RTL_USER_PROCESS_PARAMETERS + 0x60 = ImagePathName (UNICODE_STRING)
const PROCESS_PARAMS_IMAGE_PATH: u64 = 0x60;

/// A discovered Windows process.
#[derive(Debug)]
pub struct Process {
    pub pid: u64,
    pub name: String,
    pub dtb: u64,
    pub eprocess_phys: u64,
    pub peb_vaddr: u64,
}

/// Find the System process (PID 4) by scanning physical memory.
///
/// Scans page-by-page through physical address space looking for "System\0" at
/// the ImageFileName offset, then validates PID, DTB, and Flink fields.
pub fn find_system_process(
    phys: &impl PhysicalMemory,
    offsets: &EprocessOffsets,
) -> Result<Process> {
    let reader = EprocessReader::new(offsets);
    let pattern = b"System\0\0\0\0\0\0\0\0\0"; // 15 bytes for ImageFileName
    let phys_size = phys.phys_size();

    log::info!(
        "Scanning {} MB of physical memory for System process...",
        phys_size / (1024 * 1024)
    );

    // Scan physical memory page by page
    let mut page_addr: u64 = 0;
    let mut page_buf = vec![0u8; 4096];

    while page_addr < phys_size {
        if phys.read_phys(page_addr, &mut page_buf).is_err() {
            page_addr += 4096;
            continue;
        }

        // Search within this page for the pattern
        let mut off = 0usize;
        while off + pattern.len() <= page_buf.len() {
            if &page_buf[off..off + pattern.len()] != pattern {
                off += 1;
                continue;
            }

            let match_phys = page_addr + off as u64;

            // EPROCESS base = match location - ImageFileName offset
            if match_phys < offsets.image_file_name {
                off += 1;
                continue;
            }
            let eprocess_phys = match_phys - offsets.image_file_name;

            // Validate PID = 4
            let pid = match reader.read_pid(phys, eprocess_phys) {
                Ok(pid) => pid,
                Err(_) => { off += 1; continue; }
            };
            if pid != 4 {
                off += 1;
                continue;
            }

            // Validate DTB: physical base must be nonzero, within address space,
            // and only low 12 bits may differ (PCID)
            let dtb = match reader.read_dtb(phys, eprocess_phys) {
                Ok(dtb) => dtb,
                Err(_) => { off += 1; continue; }
            };
            let dtb_base = dtb & 0x000F_FFFF_FFFF_F000;
            if dtb_base == 0 || dtb_base >= phys_size {
                off += 1;
                continue;
            }

            // Validate Flink: should be a canonical kernel address (0xFFFF...)
            let flink = match reader.read_flink(phys, eprocess_phys) {
                Ok(f) => f,
                Err(_) => { off += 1; continue; }
            };
            if (flink >> 48) != 0xFFFF {
                off += 1;
                continue;
            }

            // Read PEB (will be 0 for System, that's expected)
            let peb = reader.read_peb(phys, eprocess_phys).unwrap_or(0);

            log::info!(
                "Found System process: eprocess_phys=0x{:x}, PID={}, DTB=0x{:x}, Flink=0x{:x}",
                eprocess_phys, pid, dtb, flink
            );

            return Ok(Process {
                pid,
                name: "System".to_string(),
                dtb,
                eprocess_phys,
                peb_vaddr: peb,
            });
        }

        page_addr += 4096;
    }

    Err(GovmemError::SystemProcessNotFound)
}

/// Walk the EPROCESS linked list starting from the System process.
/// Uses the kernel DTB for virtual-to-physical translation of ActiveProcessLinks pointers.
pub fn enumerate_processes(
    phys: &impl PhysicalMemory,
    system: &Process,
    offsets: &EprocessOffsets,
) -> Result<Vec<Process>> {
    let reader = EprocessReader::new(offsets);
    let walker = PageTableWalker::new(phys);
    let kernel_dtb = system.dtb;

    // Read System's ActiveProcessLinks.Flink
    let head_flink = reader.read_flink(phys, system.eprocess_phys)?;
    let mut processes = vec![];

    // Add System itself
    processes.push(Process {
        pid: system.pid,
        name: system.name.clone(),
        dtb: system.dtb,
        eprocess_phys: system.eprocess_phys,
        peb_vaddr: system.peb_vaddr,
    });

    let mut current_flink = head_flink;
    let mut visited = std::collections::HashSet::new();
    visited.insert(system.eprocess_phys + offsets.active_process_links);

    loop {
        if visited.contains(&current_flink) {
            break;
        }
        visited.insert(current_flink);

        // Translate the virtual Flink address to physical
        let flink_phys = match walker.translate(kernel_dtb, current_flink) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("Failed to translate Flink 0x{:x}: {}", current_flink, e);
                break;
            }
        };

        // EPROCESS base = Flink physical address - ActiveProcessLinks offset
        let eprocess_phys = flink_phys - offsets.active_process_links;

        // Read process info
        let pid = match reader.read_pid(phys, eprocess_phys) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("Failed to read PID at 0x{:x}: {}", eprocess_phys, e);
                break;
            }
        };

        // Read next Flink before potentially skipping this entry
        let next_flink = match reader.read_flink(phys, eprocess_phys) {
            Ok(f) => f,
            Err(e) => {
                log::warn!("Failed to read Flink at 0x{:x}: {}", eprocess_phys, e);
                break;
            }
        };

        // Skip PID 0 (System Idle Process) - has no valid DTB, PEB, or name
        if pid != 0 {
            let short_name = reader
                .read_image_name(phys, eprocess_phys)
                .unwrap_or_else(|_| "<unknown>".to_string());

            let dtb = reader.read_dtb(phys, eprocess_phys).unwrap_or(0);
            let peb = reader.read_peb(phys, eprocess_phys).unwrap_or(0);

            // Try full name from PEB if available (fixes 15-char truncation)
            let name = if peb != 0 && dtb != 0 {
                read_full_image_name(phys, dtb, peb).unwrap_or(short_name)
            } else {
                short_name
            };

            processes.push(Process {
                pid,
                name,
                dtb,
                eprocess_phys,
                peb_vaddr: peb,
            });
        }

        current_flink = next_flink;
    }

    Ok(processes)
}

/// Read the full image name from PEB → ProcessParameters → ImagePathName.
/// Returns just the filename (e.g. "fontdrvhost.exe") from the full NT path.
/// Uses the process's own DTB for virtual address translation.
fn read_full_image_name(phys: &impl PhysicalMemory, dtb: u64, peb: u64) -> Option<String> {
    let vmem = ProcessMemory::new(phys, dtb);

    // PEB + 0x20 → ProcessParameters pointer
    let params_ptr = vmem.read_virt_u64(peb + PEB_PROCESS_PARAMETERS).ok()?;
    if params_ptr == 0 || params_ptr < 0x10000 {
        return None;
    }

    // ProcessParameters + 0x60 → ImagePathName (UNICODE_STRING)
    let full_path = vmem
        .read_win_unicode_string(params_ptr + PROCESS_PARAMS_IMAGE_PATH)
        .ok()?;
    if full_path.is_empty() {
        return None;
    }

    // Extract just the filename from the path (handles both \ and / separators)
    let name = full_path
        .rsplit(|c| c == '\\' || c == '/')
        .next()
        .unwrap_or(&full_path);

    if name.is_empty() {
        return None;
    }

    Some(name.to_string())
}
