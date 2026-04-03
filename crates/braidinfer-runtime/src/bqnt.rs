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
}

impl BqntWriter {
    /// Create a new .bqnt file for writing.
    pub fn create(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::with_capacity(1 << 20, file); // 1MB buffer
        Ok(Self {
            writer,
            entries: Vec::new(),
            names: HashMap::new(),
            current_offset: 0,
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
            // Reserve space for header + tensor table (will be rewritten in finish)
            let table_end = HEADER_SIZE + self.entries.len() as u64 * ENTRY_SIZE;
            // We don't know final n_tensors yet, so reserve generous space
            // Actually, write data first, header last via seeking
            let data_start = align_up(HEADER_SIZE + 4096 * ENTRY_SIZE, ALIGNMENT);
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
        // Write JSON metadata at current position
        let metadata_offset = align_up(self.current_offset, 8);
        if metadata_offset > self.current_offset {
            let padding = vec![0u8; (metadata_offset - self.current_offset) as usize];
            self.writer.write_all(&padding)?;
        }
        let metadata_bytes = metadata_json.as_bytes();
        self.writer.write_all(metadata_bytes)?;

        // Seek back and write header
        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(&MAGIC.to_le_bytes())?;
        self.writer.write_all(&VERSION.to_le_bytes())?;
        self.writer.write_all(&(self.entries.len() as u32).to_le_bytes())?;
        self.writer.write_all(&0u32.to_le_bytes())?; // reserved
        self.writer.write_all(&metadata_offset.to_le_bytes())?;
        self.writer.write_all(&(metadata_bytes.len() as u64).to_le_bytes())?;

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
pub struct BqntFile {
    pub entries: HashMap<u64, TensorEntry>,
    pub n_tensors: usize,
    pub metadata_offset: u64,
    pub metadata_size: u64,
}

impl BqntFile {
    /// Open and parse a .bqnt file header and tensor table.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut header = [0u8; 32];
        file.read_exact(&mut header)?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a BQNT file"));
        }

        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Unsupported BQNT version {version}")));
        }

        let n_tensors = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
        let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap());
        let metadata_size = u64::from_le_bytes(header[24..32].try_into().unwrap());

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

            entries.insert(name_hash, TensorEntry {
                name_hash, format, out_features, in_features,
                original_ndim, data_offset, data_bytes,
            });
        }

        Ok(Self { entries, n_tensors, metadata_offset, metadata_size })
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

// --- mmap-based loader ---

/// Memory-mapped .bqnt file for zero-copy GPU loading.
/// Holds the mmap and parsed tensor table. Tensor data is accessed
/// via raw byte slices that can be passed directly to hipMemcpyAsync.
pub struct MmapBqnt {
    mmap: Mmap,
    header: BqntFile,
}

impl MmapBqnt {
    /// Open and mmap a .bqnt file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let header = BqntFile::open(path)?;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap, header })
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
            v.get("model_name").and_then(|v| v.as_str()).map(|s| s.to_string())
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
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let entry = self.header.get(name)?;
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
        self.header.entries.iter().filter_map(move |(&hash, entry)| {
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
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Metadata out of bounds"))
        }
    }
}
