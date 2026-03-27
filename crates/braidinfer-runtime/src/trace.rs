use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub struct TraceWriter {
    writer: BufWriter<File>,
    count: u32,
}

impl TraceWriter {
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(b"BTRC")?;
        let version: u32 = 1;
        writer.write_all(&version.to_le_bytes())?;
        Ok(TraceWriter { writer, count: 0 })
    }

    pub fn write_checkpoint(&mut self, name: &str, data: &[f32]) {
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u32;
        let _ = self.writer.write_all(&name_len.to_le_bytes());
        let _ = self.writer.write_all(name_bytes);
        let num_elements = data.len() as u32;
        let _ = self.writer.write_all(&num_elements.to_le_bytes());
        for &val in data {
            let _ = self.writer.write_all(&val.to_le_bytes());
        }
        self.count += 1;
    }

    pub fn close(mut self) -> std::io::Result<()> {
        self.writer.write_all(&self.count.to_le_bytes())?;
        self.writer.flush()?;
        Ok(())
    }
}
