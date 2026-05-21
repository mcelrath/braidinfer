//! BQNT file format: zero-copy quantized model storage.
//!
//! On-disk layout matches GPU kernel memory layout exactly.
//! See docs/bqnt_format.txt for full specification.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::quant::WeightFormat;

const MAGIC: u32 = 0x544E5142; // "BQNT" little-endian
const VERSION: u32 = 1;
const HEADER_SIZE: u64 = 32;
const ENTRY_SIZE: u64 = 48;
const ALIGNMENT: u64 = 65536; // 64KB

/// FNV-1a 64-bit hash of a tensor name.
pub fn fnv1a_64(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Format code in the file.
fn format_to_code(f: WeightFormat) -> u8 {
    match f {
        WeightFormat::Bf16 => 0,
        WeightFormat::PcG32Q4 => 1,
        WeightFormat::Rnf4G128 => 2,
    }
}

/// Format code back to enum.
pub fn code_to_format(code: u8) -> Option<WeightFormat> {
    match code {
        0 => Some(WeightFormat::Bf16),
        1 => Some(WeightFormat::PcG32Q4),
        2 => Some(WeightFormat::Rnf4G128),
        _ => None,
    }
}

/// One entry in the tensor table.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub name_hash: u64,
    pub format: u8,
    pub out_features: u32,
    pub in_features: u32,
    pub original_ndim: u32,
    pub data_offset: u64,
    pub data_bytes: u64,
}

/// Compute packed data size for a tensor at given format.
pub fn packed_size(format: WeightFormat, out_dim: usize, in_dim: usize) -> usize {
    match format {
        WeightFormat::Bf16 => out_dim * in_dim * 2,
        WeightFormat::PcG32Q4 => {
            let groups = (in_dim + 31) / 32;
            out_dim * groups * 20
        }
        WeightFormat::Rnf4G128 => {
            let groups = (in_dim + 127) / 128;
            out_dim * groups * 132
        }
    }
}

fn checked_packed_size(format: WeightFormat, out_dim: u32, in_dim: u32) -> io::Result<u64> {
    let out_dim = out_dim as u64;
    let in_dim = in_dim as u64;
    match format {
        WeightFormat::Bf16 => out_dim
            .checked_mul(in_dim)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| invalid_data("bf16 packed size overflows")),
        WeightFormat::PcG32Q4 => {
            let groups = in_dim
                .checked_add(31)
                .map(|x| x / 32)
                .ok_or_else(|| invalid_data("pcg32 packed group count overflows"))?;
            out_dim
                .checked_mul(groups)
                .and_then(|x| x.checked_mul(20))
                .ok_or_else(|| invalid_data("pcg32 packed size overflows"))
        }
        WeightFormat::Rnf4G128 => {
            let groups = in_dim
                .checked_add(127)
                .map(|x| x / 128)
                .ok_or_else(|| invalid_data("rnf4 packed group count overflows"))?;
            out_dim
                .checked_mul(groups)
                .and_then(|x| x.checked_mul(132))
                .ok_or_else(|| invalid_data("rnf4 packed size overflows"))
        }
    }
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

// --- Writer ---

/// Builder for writing a .bqnt file.
pub struct BqntWriter {
    writer: BufWriter<File>,
    entries: Vec<TensorEntry>,
    names: HashMap<u64, String>,
    current_offset: u64,
    max_tensors: usize,
}

impl BqntWriter {
    /// Create a new .bqnt file for writing.
    /// `max_tensors`: upper bound on the number of tensors to be written.
    /// The entry table is reserved at `align_up(HEADER_SIZE + max_tensors * ENTRY_SIZE, ALIGNMENT)`.
    /// Pass a count from a first-pass scan or a safe overestimate to avoid table/data overlap.
    pub fn create(path: &Path, max_tensors: usize) -> io::Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::with_capacity(1 << 20, file); // 1MB buffer
        Ok(Self {
            writer,
            entries: Vec::new(),
            names: HashMap::new(),
            current_offset: 0,
            max_tensors,
        })
    }

    /// Write a quantized tensor. `packed_data` must already be in the correct
    /// kernel-ready format (from quantize_rnf4_g128 or quantize_pc_g32_q4).
    pub fn write_tensor(
        &mut self,
        name: &str,
        format: WeightFormat,
        out_features: u32,
        in_features: u32,
        ndim: u32,
        packed_data: &[u8],
    ) -> io::Result<()> {
        let hash = fnv1a_64(name);
        // bd 6n01: detect FNV-1a hash collisions at write time. If two distinct
        // tensor names hash to the same u64, panic with both names so the user
        // can rename one. With FNV-1a 64-bit on typical tensor counts (≤2¹⁶)
        // collisions are astronomically unlikely (~2⁻³³ per pair) but possible.
        if let Some(existing) = self.names.get(&hash) {
            if existing != name {
                panic!(
                    "BQNT writer: FNV-1a hash collision detected. Names {existing:?} and \
                     {name:?} both hash to {hash:#018x}. Rename one of them or report \
                     this as a hash-function issue."
                );
            }
        }
        self.names.insert(hash, name.to_string());

        // Compute aligned offset — first tensor starts after header + table + alignment
        // For simplicity, we'll finalize offsets in finish()
        self.entries.push(TensorEntry {
            name_hash: hash,
            format: format_to_code(format),
            out_features,
            in_features,
            original_ndim: ndim,
            data_offset: 0, // filled in finish()
            data_bytes: packed_data.len() as u64,
        });

        // Store packed data temporarily — we'll write everything in finish()
        // For streaming large models, we write data immediately and record offset
        if self.current_offset == 0 {
            // Reserve space for header + tensor table (will be rewritten in finish).
            // data_start is determined by max_tensors passed at construction time.
            // CRITICAL: data_start must be >= HEADER_SIZE + actual_n_tensors * ENTRY_SIZE or
            // finish() will overwrite tensor data when writing the entry table.
            let data_start = align_up(
                HEADER_SIZE + self.max_tensors as u64 * ENTRY_SIZE,
                ALIGNMENT,
            );
            self.current_offset = data_start;
            self.writer.seek(SeekFrom::Start(data_start))?;
        }

        // Align current offset
        let aligned = align_up(self.current_offset, ALIGNMENT);
        if aligned > self.current_offset {
            let padding = vec![0u8; (aligned - self.current_offset) as usize];
            self.writer.write_all(&padding)?;
        }

        // Record actual offset
        let last = self.entries.last_mut().unwrap();
        last.data_offset = aligned;

        // Write packed data
        self.writer.write_all(packed_data)?;
        self.current_offset = aligned + packed_data.len() as u64;

        Ok(())
    }

    /// Finalize: write header, tensor table, and JSON metadata.
    pub fn finish(mut self, metadata_json: &str) -> io::Result<()> {
        // bd 6n01: inject the {hash -> name} table into the metadata JSON so
        // the reader can verify on lookup that a tensor name maps to the
        // expected hash entry (detects post-write corruption / future
        // hash-function changes / catastrophically improbable collisions).
        // Hex-encoded u64 keys so JSON stays parseable.
        let augmented_metadata = {
            let mut v: serde_json::Value = serde_json::from_str(metadata_json)
                .unwrap_or_else(|_| serde_json::json!({}));
            let mut name_table = serde_json::Map::new();
            for (hash, name) in &self.names {
                name_table.insert(format!("{hash:#018x}"), serde_json::Value::String(name.clone()));
            }
            if let Some(obj) = v.as_object_mut() {
                obj.insert("name_table".to_string(), serde_json::Value::Object(name_table));
            }
            v.to_string()
        };
        // Write JSON metadata at current position
        let metadata_offset = align_up(self.current_offset, 8);
        if metadata_offset > self.current_offset {
            let padding = vec![0u8; (metadata_offset - self.current_offset) as usize];
            self.writer.write_all(&padding)?;
        }
        let metadata_bytes = augmented_metadata.as_bytes();
        self.writer.write_all(metadata_bytes)?;

        // Seek back and write header
        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(&MAGIC.to_le_bytes())?;
        self.writer.write_all(&VERSION.to_le_bytes())?;
        self.writer
            .write_all(&(self.entries.len() as u32).to_le_bytes())?;
        self.writer
            .write_all(&(self.max_tensors as u32).to_le_bytes())?; // reserved_entries
        self.writer.write_all(&metadata_offset.to_le_bytes())?;
        self.writer
            .write_all(&(metadata_bytes.len() as u64).to_le_bytes())?;

        // Write tensor table
        for entry in &self.entries {
            self.writer.write_all(&entry.name_hash.to_le_bytes())?;
            self.writer.write_all(&[entry.format, 0, 0, 0])?;
            self.writer.write_all(&entry.out_features.to_le_bytes())?;
            self.writer.write_all(&entry.in_features.to_le_bytes())?;
            self.writer.write_all(&entry.original_ndim.to_le_bytes())?;
            self.writer.write_all(&entry.data_offset.to_le_bytes())?;
            self.writer.write_all(&entry.data_bytes.to_le_bytes())?;
            self.writer.write_all(&0u64.to_le_bytes())?; // reserved
        }

        self.writer.flush()?;
        Ok(())
    }
}

// --- Reader ---

/// Parsed .bqnt file header and tensor table.
#[derive(Debug)]
pub struct BqntFile {
    pub entries: HashMap<u64, TensorEntry>,
    pub n_tensors: usize,
    pub metadata_offset: u64,
    pub metadata_size: u64,
}

fn invalid_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn checked_end(offset: u64, size: u64, context: &str) -> io::Result<u64> {
    offset
        .checked_add(size)
        .ok_or_else(|| invalid_data(format!("{context} range overflows file address space")))
}

fn ranges_overlap(start_a: u64, end_a: u64, start_b: u64, end_b: u64) -> bool {
    start_a < end_b && start_b < end_a
}

impl BqntFile {
    /// Open and parse a .bqnt file header and tensor table.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < HEADER_SIZE {
            return Err(invalid_data(format!(
                "BQNT file too small: expected at least {HEADER_SIZE} bytes, got {file_len}"
            )));
        }
        let mut header = [0u8; 32];
        file.read_exact(&mut header)?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Not a BQNT file",
            ));
        }

        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported BQNT version {version}"),
            ));
        }

        let n_tensors = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
        // bytes 12-15: reserved_entries (added in v1.1). Zero in old files → default to 4096.
        // The writer always reserves this many entry slots before tensor data starts.
        let reserved_entries = {
            let v = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
            if v == 0 { 4096 } else { v }
        };
        let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap());
        let metadata_size = u64::from_le_bytes(header[24..32].try_into().unwrap());
        let table_bytes = (n_tensors as u64).checked_mul(ENTRY_SIZE).ok_or_else(|| {
            invalid_data(format!(
                "tensor table size overflows for {n_tensors} entries"
            ))
        })?;
        let table_end_exact = checked_end(HEADER_SIZE, table_bytes, "tensor table")?;
        // Use the aligned reserved size for overlap checks (writer always aligns to ALIGNMENT).
        let table_end = align_up(
            HEADER_SIZE + reserved_entries as u64 * ENTRY_SIZE,
            ALIGNMENT,
        );
        if table_end_exact > file_len {
            return Err(invalid_data(format!(
                "tensor table ends at {table_end_exact}, beyond file size {file_len}"
            )));
        }
        let metadata_end = checked_end(metadata_offset, metadata_size, "metadata")?;
        if metadata_end > file_len {
            return Err(invalid_data(format!(
                "metadata ends at {metadata_end}, beyond file size {file_len}"
            )));
        }
        if metadata_size != 0 && metadata_offset < table_end {
            return Err(invalid_data(format!(
                "metadata offset {metadata_offset} overlaps tensor table ending at {table_end}"
            )));
        }

        let mut entries = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let mut buf = [0u8; 48];
            file.read_exact(&mut buf)?;

            let name_hash = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let format = buf[8];
            let out_features = u32::from_le_bytes(buf[12..16].try_into().unwrap());
            let in_features = u32::from_le_bytes(buf[16..20].try_into().unwrap());
            let original_ndim = u32::from_le_bytes(buf[20..24].try_into().unwrap());
            let data_offset = u64::from_le_bytes(buf[24..32].try_into().unwrap());
            let data_bytes = u64::from_le_bytes(buf[32..40].try_into().unwrap());
            let format = code_to_format(format).ok_or_else(|| {
                invalid_data(format!(
                    "tensor {name_hash:#x} has unknown format code {}",
                    buf[8]
                ))
            })?;
            let expected_bytes = checked_packed_size(format, out_features, in_features)?;
            if data_bytes != expected_bytes {
                return Err(invalid_data(format!(
                    "tensor {name_hash:#x} size mismatch: header says {data_bytes} bytes, expected {expected_bytes}"
                )));
            }
            let data_end = checked_end(data_offset, data_bytes, "tensor data")?;
            if data_end > file_len {
                return Err(invalid_data(format!(
                    "tensor {name_hash:#x} ends at {data_end}, beyond file size {file_len}"
                )));
            }
            if data_offset < table_end {
                return Err(invalid_data(format!(
                    "tensor {name_hash:#x} data offset {data_offset} overlaps tensor table ending at {table_end}"
                )));
            }
            if metadata_size != 0
                && ranges_overlap(data_offset, data_end, metadata_offset, metadata_end)
            {
                return Err(invalid_data(format!(
                    "tensor {name_hash:#x} data range [{data_offset}, {data_end}) overlaps metadata range [{metadata_offset}, {metadata_end})"
                )));
            }
            if entries.contains_key(&name_hash) {
                return Err(invalid_data(format!(
                    "duplicate tensor hash {name_hash:#x}; hash-only lookup would be ambiguous"
                )));
            }

            entries.insert(
                name_hash,
                TensorEntry {
                    name_hash,
                    format: buf[8],
                    out_features,
                    in_features,
                    original_ndim,
                    data_offset,
                    data_bytes,
                },
            );
        }

        Ok(Self {
            entries,
            n_tensors,
            metadata_offset,
            metadata_size,
        })
    }

    /// Look up a tensor by name.
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.entries.get(&fnv1a_64(name))
    }

    /// Read JSON metadata from file.
    pub fn read_metadata(&self, path: &Path) -> io::Result<String> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(self.metadata_offset))?;
        let mut buf = vec![0u8; self.metadata_size as usize];
        file.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("braidinfer-{name}-{unique}.bqnt"))
    }

    #[test]
    fn rejects_duplicate_hash_entries() {
        // Build a valid 2-tensor file via BqntWriter, then mutate entry 1's
        // hash bytes to match entry 0's. The reader's per-entry checks (size,
        // data_offset >= table_end, no metadata overlap) all pass on the
        // writer-produced layout, so validation reaches the duplicate-hash
        // gate at bqnt.rs:402.
        let path = temp_path("dup-hash");
        let mut w = BqntWriter::create(&path, 2).unwrap();
        w.write_tensor("a", WeightFormat::Bf16, 1, 1, 2, &[0, 0]).unwrap();
        w.write_tensor("b", WeightFormat::Bf16, 1, 1, 2, &[0, 0]).unwrap();
        w.finish("{}").unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        // Entry table starts at HEADER_SIZE (32). Each entry is ENTRY_SIZE (48).
        // Copy entry 0's hash (bytes 32..40) into entry 1's hash slot (80..88).
        let hash0 = bytes[32..40].to_vec();
        bytes[80..88].copy_from_slice(&hash0);
        std::fs::write(&path, bytes).unwrap();

        let err = match BqntFile::open(&path) {
            Ok(_) => panic!("expected duplicate-hash BQNT to be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("duplicate tensor hash"),
            "{err}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_out_of_bounds_metadata() {
        let path = temp_path("bad-meta");
        let mut writer = BqntWriter::create(&path, 64).unwrap();
        writer
            .write_tensor("x", WeightFormat::Bf16, 1, 1, 2, &[0, 0])
            .unwrap();
        writer.finish("{}").unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let bad_offset = (bytes.len() as u64) + 16;
        bytes[16..24].copy_from_slice(&bad_offset.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let err = match BqntFile::open(&path) {
            Ok(_) => panic!("expected out-of-bounds metadata to be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("metadata ends"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_tensor_range_overlapping_metadata() {
        // Build a valid 1-tensor file via BqntWriter, then mutate entry 0's
        // data_offset (bytes 56..64) to point at metadata_offset (read from
        // header bytes 16..24). The reader's data_offset >= table_end check
        // still passes (both regions sit well past the aligned table), so
        // validation reaches the ranges_overlap gate at bqnt.rs:395.
        let path = temp_path("metadata-overlap");
        let mut w = BqntWriter::create(&path, 1).unwrap();
        w.write_tensor("x", WeightFormat::Bf16, 1, 1, 2, &[0, 0]).unwrap();
        w.finish("{}").unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let metadata_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        // Entry 0 data_offset is at offset HEADER_SIZE + 24 = 56.
        bytes[56..64].copy_from_slice(&metadata_offset.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let err = match BqntFile::open(&path) {
            Ok(_) => panic!("expected metadata-overlapping tensor to be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("overlaps metadata range"),
            "{err}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_overflowing_packed_size() {
        let path = temp_path("packed-size-overflow");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let metadata_offset = HEADER_SIZE + ENTRY_SIZE;
        bytes.extend_from_slice(&metadata_offset.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());

        bytes.extend_from_slice(&0x5678u64.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&metadata_offset.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());

        std::fs::write(&path, bytes).unwrap();
        let err = match BqntFile::open(&path) {
            Ok(_) => panic!("expected overflowed packed size to be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("packed size overflows"));
        let _ = std::fs::remove_file(path);
    }
}

// --- mmap-based loader ---

/// Memory-mapped .bqnt file for zero-copy GPU loading.
/// Holds the mmap and parsed tensor table. Tensor data is accessed
/// via raw byte slices that can be passed directly to hipMemcpyAsync.
pub struct MmapBqnt {
    mmap: Mmap,
    header: BqntFile,
    /// bd 6n01: hash → name table parsed from metadata JSON. Used by
    /// tensor_data to verify that a name lookup didn't return a collided
    /// entry. None for v1 BQNT files written before bd 6n01 landed.
    name_table: Option<HashMap<u64, String>>,
}

impl MmapBqnt {
    /// Open and mmap a .bqnt file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let header = BqntFile::open(path)?;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        // Parse name_table from metadata JSON if present.
        let name_table = {
            let start = header.metadata_offset as usize;
            let end = start + header.metadata_size as usize;
            if end <= mmap.len() && header.metadata_size > 0 {
                std::str::from_utf8(&mmap[start..end])
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                    .and_then(|v| v.get("name_table").cloned())
                    .and_then(|v| v.as_object().cloned())
                    .map(|obj| {
                        let mut m = HashMap::with_capacity(obj.len());
                        for (k, v) in obj {
                            let hash = u64::from_str_radix(k.trim_start_matches("0x"), 16).ok();
                            let name = v.as_str().map(|s| s.to_string());
                            if let (Some(h), Some(n)) = (hash, name) {
                                m.insert(h, n);
                            }
                        }
                        m
                    })
            } else {
                None
            }
        };
        Ok(Self { mmap, header, name_table })
    }

    /// Number of tensors in the file.
    pub fn n_tensors(&self) -> usize {
        self.header.n_tensors
    }

    /// Read model_name from JSON metadata.
    pub fn model_name(&self) -> Option<String> {
        // Read metadata directly from mmap
        let start = self.header.metadata_offset as usize;
        let end = start + self.header.metadata_size as usize;
        if end <= self.mmap.len() {
            let json_str = std::str::from_utf8(&self.mmap[start..end]).ok()?;
            let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
            v.get("model_name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Look up a tensor entry by name.
    pub fn entry(&self, name: &str) -> Option<&TensorEntry> {
        self.header.get(name)
    }

    /// Get raw packed bytes for a tensor — zero-copy from mmap.
    /// The returned slice is 64KB-aligned and byte-identical to what
    /// the GPU dequant kernel expects. Pass directly to hipMemcpyAsync.
    ///
    /// bd 6n01: if the file contains a `name_table` in its metadata,
    /// verify the resolved hash entry matches the requested name. Returns
    /// None (with a stderr warning) on collision; this should never fire
    /// in practice but is the cheap defense against silent corruption.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let entry = self.header.get(name)?;
        if let Some(name_table) = self.name_table.as_ref() {
            let hash = fnv1a_64(name);
            if let Some(stored) = name_table.get(&hash) {
                if stored != name {
                    eprintln!(
                        "WARN: BQNT name-table mismatch for {name:?}: hash {hash:#018x} \
                         maps to entry stored as {stored:?}. Likely FNV-1a collision. \
                         Returning None (treat as missing weight)."
                    );
                    return None;
                }
            }
        }
        let start = entry.data_offset as usize;
        let end = start + entry.data_bytes as usize;
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// Get raw packed bytes by hash — for hot-path loading without string hashing.
    pub fn tensor_data_by_hash(&self, hash: u64) -> Option<&[u8]> {
        let entry = self.header.entries.get(&hash)?;
        let start = entry.data_offset as usize;
        let end = start + entry.data_bytes as usize;
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// Iterate all tensors: yields (name_hash, entry, data_slice).
    pub fn iter_tensors(&self) -> impl Iterator<Item = (u64, &TensorEntry, &[u8])> {
        self.header
            .entries
            .iter()
            .filter_map(move |(&hash, entry)| {
                let start = entry.data_offset as usize;
                let end = start + entry.data_bytes as usize;
                if end <= self.mmap.len() {
                    Some((hash, entry, &self.mmap[start..end]))
                } else {
                    None
                }
            })
    }

    /// Read JSON metadata.
    pub fn metadata(&self) -> io::Result<String> {
        let start = self.header.metadata_offset as usize;
        let end = start + self.header.metadata_size as usize;
        if end <= self.mmap.len() {
            String::from_utf8(self.mmap[start..end].to_vec())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        } else {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Metadata out of bounds",
            ))
        }
    }
}
