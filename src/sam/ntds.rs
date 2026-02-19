//! NTDS.dit helpers for AD secrets extraction workflows.
//!
//! This module validates extracted NTDS artifacts and prepares metadata
//! needed for downstream domain credential extraction.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{GovmemError, Result};

/// High-level NTDS context extracted from disk artifacts.
#[derive(Debug, Clone)]
pub struct NtdsContext {
    pub ntds_size: usize,
    pub boot_key: [u8; 16],
}

/// A single AD NTLM hash entry extracted from NTDS.
#[derive(Debug, Clone)]
pub struct AdHashEntry {
    pub username: String,
    pub rid: u32,
    pub lm_hash: [u8; 16],
    pub nt_hash: [u8; 16],
    pub is_history: bool,
    pub history_index: Option<u32>,
}

/// Build NTDS context from raw NTDS.dit + SYSTEM hive bytes.
pub fn build_context(ntds_data: &[u8], system_data: &[u8]) -> Result<NtdsContext> {
    if !is_ese_database(ntds_data) {
        return Err(GovmemError::DecryptionError(
            "NTDS.dit does not look like a valid ESE database".to_string(),
        ));
    }

    let boot_key = super::bootkey::extract_bootkey(system_data)?;
    Ok(NtdsContext {
        ntds_size: ntds_data.len(),
        boot_key,
    })
}

/// Minimal ESE validity check for NTDS.dit.
///
/// ESE databases use little-endian 0x89ABCDEF at offset 0x04.
fn is_ese_database(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }
    data[4..8] == [0xEF, 0xCD, 0xAB, 0x89]
}

/// Extract AD NTLM hashes from NTDS + SYSTEM using local impacket-secretsdump.
///
/// This keeps orchestration in Rust while leveraging the battle-tested parser
/// already present in the runtime environment.
pub fn extract_ad_hashes(
    ntds_data: &[u8],
    system_data: &[u8],
    include_history: bool,
) -> Result<Vec<AdHashEntry>> {
    let _ctx = build_context(ntds_data, system_data)?;
    let temp_dir = make_temp_dir()?;
    let ntds_path = temp_dir.join("ntds.dit");
    let system_path = temp_dir.join("SYSTEM");

    fs::write(&ntds_path, ntds_data).map_err(GovmemError::Io)?;
    fs::write(&system_path, system_data).map_err(GovmemError::Io)?;

    let output = run_secretsdump(&system_path, &ntds_path, include_history);
    let _ = fs::remove_file(&ntds_path);
    let _ = fs::remove_file(&system_path);
    let _ = fs::remove_dir(&temp_dir);

    let stdout = output?;
    parse_secretsdump_output(&stdout)
}

fn run_secretsdump(
    system_path: &PathBuf,
    ntds_path: &PathBuf,
    include_history: bool,
) -> Result<String> {
    let mut cmd = Command::new("impacket-secretsdump");
    cmd.arg("-system")
        .arg(system_path)
        .arg("-ntds")
        .arg(ntds_path)
        .arg("-just-dc-ntlm");
    if include_history {
        cmd.arg("-history");
    }
    cmd.arg("LOCAL");

    let output = cmd.output().map_err(|e| {
        GovmemError::DecryptionError(format!(
            "Failed to execute impacket-secretsdump (is it installed?): {}",
            e
        ))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(GovmemError::DecryptionError(format!(
            "impacket-secretsdump failed: {}",
            stderr.trim()
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn parse_secretsdump_output(stdout: &str) -> Result<Vec<AdHashEntry>> {
    let mut entries = Vec::new();

    for line in stdout.lines() {
        // Format: username:rid:lmhash:nthash:::
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() < 4 {
            continue;
        }

        let username_raw = parts[0].trim();
        let rid_raw = parts[1].trim();
        let lm_raw = parts[2].trim();
        let nt_raw = parts[3].trim();

        if username_raw.is_empty() || rid_raw.is_empty() || lm_raw.len() != 32 || nt_raw.len() != 32
        {
            continue;
        }

        let rid = match rid_raw.parse::<u32>() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let lm_hash = parse_hash_16(lm_raw)?;
        let nt_hash = parse_hash_16(nt_raw)?;
        let (username, is_history, history_index) = parse_history_name(username_raw);

        entries.push(AdHashEntry {
            username,
            rid,
            lm_hash,
            nt_hash,
            is_history,
            history_index,
        });
    }

    if entries.is_empty() {
        return Err(GovmemError::DecryptionError(
            "No NTDS NTLM hashes found in secretsdump output".to_string(),
        ));
    }

    Ok(entries)
}

fn parse_hash_16(hex_str: &str) -> Result<[u8; 16]> {
    let bytes = hex::decode(hex_str).map_err(|e| {
        GovmemError::DecryptionError(format!("Invalid hex hash '{}': {}", hex_str, e))
    })?;
    if bytes.len() != 16 {
        return Err(GovmemError::DecryptionError(format!(
            "Invalid hash length for '{}': {}",
            hex_str,
            bytes.len()
        )));
    }

    let mut out = [0u8; 16];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn parse_history_name(name: &str) -> (String, bool, Option<u32>) {
    let marker = "_history";
    if let Some(idx) = name.rfind(marker) {
        let base = &name[..idx];
        let suffix = &name[idx + marker.len()..];
        if !base.is_empty() {
            if suffix.is_empty() {
                return (base.to_string(), true, None);
            }
            if suffix.chars().all(|c| c.is_ascii_digit()) {
                let hist_idx = suffix.parse::<u32>().ok();
                return (base.to_string(), true, hist_idx);
            }
        }
    }

    (name.to_string(), false, None)
}

fn make_temp_dir() -> Result<PathBuf> {
    let base = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| GovmemError::DecryptionError(format!("System clock error: {}", e)))?
        .as_millis();
    let pid = std::process::id();

    for attempt in 0..16u32 {
        let dir = base.join(format!("vmkatz-ntds-{}-{}-{}", pid, ts, attempt));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(GovmemError::Io(e)),
        }
    }

    Err(GovmemError::DecryptionError(
        "Failed to create temporary directory for NTDS extraction".to_string(),
    ))
}
