//! Output files are written beside their destination and moved into place only
//! once complete, so a failed or interrupted run never leaves a truncated
//! archive where a reader would take it for a finished one.

use std::{
    ffi::OsString,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{Context, Result};

/// Files being written, so that a signal handler can remove them.
static UNFINISHED: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// A file written under a temporary name in its destination's directory.
///
/// It is renamed onto the destination by [`Self::persist`]; dropped without
/// that, it is removed.
pub(crate) struct PendingFile {
    path: PathBuf,
    destination: PathBuf,
    persisted: bool,
}

impl PendingFile {
    pub(crate) fn create(destination: &Path) -> Result<(Self, File)> {
        // An existing link is followed to the file it names, as writing in
        // place did; renaming onto the link would replace the link itself.
        let destination =
            &fs::canonicalize(destination).unwrap_or_else(|_| destination.to_path_buf());
        let name = destination
            .file_name()
            .with_context(|| format!("output {} has no file name", destination.display()))?;
        let mut temporary = OsString::from(".");
        temporary.push(name);
        temporary.push(format!(".{}.tmp", std::process::id()));
        let path = destination.with_file_name(temporary);
        // Registered before it exists, so that a signal arriving in between
        // still finds it.
        lock().push(path.clone());
        let pending = Self {
            path,
            destination: destination.to_path_buf(),
            persisted: false,
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&pending.path)
            .with_context(|| format!("failed to create output {}", destination.display()))?;
        Ok((pending, file))
    }

    pub(crate) fn persist(mut self) -> Result<()> {
        fs::rename(&self.path, &self.destination).with_context(|| {
            format!(
                "failed to move {} to {}",
                self.path.display(),
                self.destination.display()
            )
        })?;
        self.persisted = true;
        Ok(())
    }
}

impl Drop for PendingFile {
    fn drop(&mut self) {
        lock().retain(|path| path != &self.path);
        if !self.persisted {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Removes every output still being written. For a signal handler, before the
/// process exits.
pub fn remove_unfinished_outputs() {
    for path in lock().drain(..) {
        let _ = fs::remove_file(path);
    }
}

fn lock() -> std::sync::MutexGuard<'static, Vec<PathBuf>> {
    // A panic while holding the list cannot leave it inconsistent.
    UNFINISHED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("gpv2pmtiles-output-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn entries(directory: &Path) -> Vec<OsString> {
        let mut names = fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    #[test]
    fn a_persisted_file_replaces_its_destination() {
        let directory = scratch("persist");
        let destination = directory.join("out.pmtiles");
        fs::write(&destination, b"old").unwrap();
        let (pending, mut file) = PendingFile::create(&destination).unwrap();
        file.write_all(b"new").unwrap();
        // Until it is complete, the destination keeps what it held.
        assert_eq!(fs::read(&destination).unwrap(), b"old");
        drop(file);
        pending.persist().unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"new");
        assert_eq!(entries(&directory), ["out.pmtiles"]);
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_link_keeps_pointing_at_the_new_file() {
        let directory = scratch("link");
        let target = directory.join("target.pmtiles");
        let link = directory.join("link.pmtiles");
        fs::write(&target, b"old").unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let (pending, mut file) = PendingFile::create(&link).unwrap();
        file.write_all(b"new").unwrap();
        drop(file);
        pending.persist().unwrap();
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read(&target).unwrap(), b"new");
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn an_unfinished_file_leaves_nothing_behind() {
        let directory = scratch("drop");
        let destination = directory.join("out.pmtiles");
        let (pending, mut file) = PendingFile::create(&destination).unwrap();
        file.write_all(b"partial").unwrap();
        drop(file);
        drop(pending);
        assert!(entries(&directory).is_empty());
        assert!(!lock().iter().any(|path| path.starts_with(&directory)));
        fs::remove_dir_all(directory).unwrap();
    }
}
