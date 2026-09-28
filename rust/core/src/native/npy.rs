//! 2-D `.npy` tables, memory-mapped: the text table alone is 620 MB of float16,
//! and only the rows a phrase uses are ever read.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, bail};
use half::f16;
use memmap2::Mmap;

pub enum Kind {
    F16,
    F32,
}

pub struct Table {
    map: Mmap,
    offset: usize,
    kind: Kind,
    pub rows: usize,
    pub cols: usize,
}

impl Table {
    pub fn open(path: &Path) -> anyhow::Result<Table> {
        let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 10 || &map[..6] != b"\x93NUMPY" {
            bail!("{} is not a .npy file", path.display());
        }
        let (hlen, start) = if map[6] == 1 {
            (u16::from_le_bytes([map[8], map[9]]) as usize, 10)
        } else {
            (u32::from_le_bytes([map[8], map[9], map[10], map[11]]) as usize, 12)
        };
        let header = std::str::from_utf8(&map[start..start + hlen])?;
        let kind = if header.contains("'<f2'") {
            Kind::F16
        } else if header.contains("'<f4'") {
            Kind::F32
        } else {
            bail!("{}: only little-endian float16/float32 tables", path.display());
        };
        if header.contains("'fortran_order': True") {
            bail!("{}: Fortran order is not supported", path.display());
        }
        let shape = header.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).context("shape")?;
        let dims: Vec<usize> = shape.split(',').filter_map(|d| d.trim().parse().ok()).collect();
        let (rows, cols) = match dims.as_slice() {
            [r, c] => (*r, *c),
            [n] => (1, *n),
            _ => bail!("{}: expected a 2-D table", path.display()),
        };
        Ok(Table { map, offset: start + hlen, kind, rows, cols })
    }

    /// One row as float32, added into `out` (the prompt sums embeddings).
    pub fn add_row(&self, row: usize, out: &mut [f32]) {
        assert!(row < self.rows, "row {row} of {}", self.rows);
        match self.kind {
            Kind::F16 => {
                let at = self.offset + row * self.cols * 2;
                for (o, c) in out.iter_mut().zip(self.map[at..at + self.cols * 2].chunks_exact(2)) {
                    *o += f16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            Kind::F32 => {
                let at = self.offset + row * self.cols * 4;
                for (o, c) in out.iter_mut().zip(self.map[at..at + self.cols * 4].chunks_exact(4)) {
                    *o += f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
        }
    }

    pub fn row(&self, row: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.cols];
        self.add_row(row, &mut out);
        out
    }

    pub fn all(&self) -> Vec<f32> {
        (0..self.rows).flat_map(|r| self.row(r)).collect()
    }
}
